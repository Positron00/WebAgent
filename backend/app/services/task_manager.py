"""
Task manager service for the WebAgent backend.
Manages workflow task execution and results.
"""
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import asyncio
import logging
import os
import traceback

from app.agents.supervisor import SupervisorAgent
from app.graph.workflows import get_agent_workflow
from app.models.task import TaskInfo, WorkflowState
from app.core.config import settings
from app.core.metrics import timing_decorator, track_task_start, track_task_completion
from app.security.errors import GuardDeclined, GuardStop
from app.core.loadEnvYAML import get_agents_config
from app.services import progress, result_store, run_origin, task_budget
from app.services.run_supervisor import RunSupervisor
from app.services.task_budget import BUDGET_MESSAGE, TaskBudgetExceeded
from app.core.log_context import current_task_id

logger = logging.getLogger(__name__)

# A saved "processing" record older than this is reported as interrupted.
TASK_TIMEOUT_MINUTES = int(os.getenv("WEBAGENT_TASK_TIMEOUT_MINUTES", "30"))

def _search_cap() -> int:
    """The per-task search cap (REC-109): agents.web_research.max_searches_per_task, 0 if unset."""
    try:
        return int(get_agents_config().web_research.max_searches_per_task)
    except Exception as exc:  # noqa: BLE001 - a missing config must not stop the run
        logger.warning(f"No web_research.max_searches_per_task ({exc}); searches are not capped")
        return 0


def _standalone_query(context: Optional[Dict[str, Any]]) -> Optional[str]:
    """The supervisor's standalone question (REC-110), when the plan has one. Never logged."""
    plan = (context or {}).get("research_plan")
    value = plan.get("standalone_query") if isinstance(plan, dict) else None
    return value if isinstance(value, str) and value.strip() else None


def _saved_time(value: Any) -> Optional[datetime]:
    """A saved record's timestamp (an ISO string on disk), or None."""
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value)) if value else None
    except ValueError:
        return None


class TaskManager:
    """
    Manages the execution and results of workflow tasks.
    """
    _instance = None
    
    def __new__(cls):
        """Singleton pattern to ensure only one task manager exists."""
        if cls._instance is None:
            cls._instance = super(TaskManager, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize the task manager."""
        self.active_tasks = {}  # task_id -> task_info
        self.task_results = {}  # task_id -> result
        self.task_errors = {}   # task_id -> error details
        self.supervisor = RunSupervisor(on_stop=self._mark_stopped)
        # task_id -> {"origin", "topic_id", "counter"} while the run executes in this worker.
        self._run_meta: Dict[str, Dict[str, Any]] = {}
        logger.info("Task Manager initialized")

    def start(
        self, task_id: str, message: str, context: Dict[str, Any], direct: bool,
        *, origin: Optional[str] = None, topic_id: Optional[str] = None, token_budget: Optional[int] = None,
    ) -> None:
        """Start a direct or workflow run in the background, under the run time limit.

        Must be called from a running event loop (an async endpoint). Call register_pending first.
        `origin="schedule"`, `topic_id` and `token_budget` mark a scheduled topic run (REC-108).
        """
        run = self.run_direct if direct else self.run_workflow
        if origin is None and topic_id is None and token_budget is None:
            self.supervisor.start(task_id, run(task_id, message, context))
        else:
            self.supervisor.start(
                task_id, run(task_id, message, context, origin=origin, topic_id=topic_id, token_budget=token_budget)
            )

    def _enter_run(
        self, task_id: str, origin: Optional[str], topic_id: Optional[str], token_budget: Optional[int],
        search_cap: int = 0,
    ) -> Tuple[Any, ...]:
        """Set the run's context variables: task ID, budget counter and (scheduled runs) topic_run."""
        scheduled = origin == "schedule"
        tokens = (
            current_task_id.set(task_id),
            task_budget.start(token_budget or settings.TASK_TOKEN_BUDGET, search_cap=search_cap),
            run_origin.topic_run.set(topic_id if scheduled else None),
        )
        self._run_meta[task_id] = {
            "origin": "schedule" if scheduled else "interactive",
            "topic_id": topic_id if scheduled else None,
            "counter": task_budget.current(),
        }
        return tokens

    def _exit_run(self, task_id: str, tokens: Tuple[Any, ...]) -> None:
        task_id_token, budget_token, topic_token = tokens
        run_origin.topic_run.reset(topic_token)
        task_budget.reset(budget_token)
        current_task_id.reset(task_id_token)
        self._run_meta.pop(task_id, None)

    def _origin(self, task_id: str) -> str:
        """Metrics origin label of a run in this worker: "interactive" or "schedule"."""
        return self._run_meta.get(task_id, {}).get("origin", "interactive")

    def _run_fields(self, task_id: str) -> Dict[str, Any]:
        """Fields every result gets: usage (REC-109); origin and topic_id for scheduled runs (REC-108)."""
        meta = self._run_meta.get(task_id) or {}
        fields: Dict[str, Any] = {"usage": task_budget.usage(meta.get("counter"))}
        if meta.get("origin") == "schedule":
            fields["origin"] = "schedule"
            fields["topic_id"] = meta.get("topic_id")
        return fields

    def _mark_stopped(self, task_id: str, code: str, message: str) -> None:
        """Show a run the supervisor stopped as an error in this worker's memory.

        The run's own TaskInfo object is changed in place: its `finally` writes that object back.
        """
        task_info = self.active_tasks.get(task_id)
        if task_info is not None:
            task_info.status = "error"
            task_info.error = message
            task_info.message = message
            task_info.current_step = "error"
            task_info.completed_at = datetime.now()
            duration = (task_info.completed_at - task_info.created_at).total_seconds()
            track_task_completion(task_id, duration, "error", self._origin(task_id))
        fields = self._run_fields(task_id)
        self.task_results[task_id] = {"error": message, "error_code": code, **fields}
        self.task_errors[task_id] = {"message": message, "error_code": code}
        self._add_fields_to_stop_record(task_id, code, fields)

    @staticmethod
    def _add_fields_to_stop_record(task_id: str, code: str, fields: Dict[str, Any]) -> None:
        """Add usage (and origin, topic_id) to the record the supervisor saved for a stopped run."""
        try:
            saved = result_store.load_result(task_id)
            if not saved or saved.get("error_code") != code:
                return
            saved.update(fields)
            saved["result"] = {**(saved.get("result") or {}), **fields}
            result_store.save_result(task_id, saved)
        except Exception as exc:  # noqa: BLE001 - the stop itself is already saved
            logger.warning(f"Could not add usage to the stopped task {task_id}: {exc}")
        
    def register_pending(self, task_id: str, message: str) -> None:
        """Record a task as processing before its background job starts, so early polls find it.

        The record is also saved to the result store, so a poll that reaches another worker
        process (uvicorn --workers N) sees "processing" instead of "not found".
        """
        created_at = datetime.now()
        self.active_tasks[task_id] = TaskInfo(
            task_id=task_id,
            status="processing",
            created_at=created_at,
            message="Task queued",
            query=message,
            current_step="queued",
        )
        result_store.save_result(
            task_id,
            {
                "task_id": task_id,
                "query": message,
                "status": "processing",
                "created_at": created_at,
                "result": None,
            },
        )

    @staticmethod
    def _status_from_saved(saved: Dict[str, Any]) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Status of a task held by another worker, read from its saved record."""
        status = saved.get("status", "completed")
        if status == "processing":
            # A worker that died mid-task leaves "processing" behind; don't let clients wait forever.
            try:
                age = datetime.now() - datetime.fromisoformat(str(saved.get("created_at")))
            except ValueError:
                age = timedelta(0)
            if age > timedelta(minutes=TASK_TIMEOUT_MINUTES):
                return "error", {"error": "Task was interrupted before it finished"}
            return "processing", None
        return status, TaskManager._result_for_client(saved)

    @staticmethod
    def _result_for_client(saved: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Merge routing metadata into the report returned to API clients."""
        result = saved.get("result")
        if not isinstance(result, dict):
            return result
        meta_keys = ("route", "decided_by", "p_agentic", "image_ignored", "triage")
        meta = {k: saved[k] for k in meta_keys if k in saved}
        return {**result, **meta} if meta else result

    @staticmethod
    def _result_meta(context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Route fields for a saved result.

        `context["triage"]` is the decision that chose the path; `context["shadow_triage"]` (shadow
        mode only) is what the classifier would have chosen and is what clients see as `triage`.
        """
        ctx = context or {}
        meta: Dict[str, Any] = {}
        decision = ctx.get("triage")
        if decision:
            meta["route"] = decision.get("route")
            meta["decided_by"] = decision.get("decided_by")
            if decision.get("p_agentic") is not None:
                meta["p_agentic"] = decision.get("p_agentic")
        elif ctx.get("origin") == "schedule" or run_origin.topic_run.get():
            # Scheduled runs skip triage and always research (recurring-research spec REC-108).
            meta["route"] = "agentic"
            meta["decided_by"] = "schedule"
        triage = ctx.get("shadow_triage") or decision
        if triage:
            meta["triage"] = triage
        if ctx.get("had_image"):
            meta["image_ignored"] = True
        return meta

    @staticmethod
    def _security_block(context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Copy the workflow security ledger for API clients (SEC-082)."""
        ctx = context or {}
        sec = ctx.get("security")
        if not isinstance(sec, dict):
            sec = {
                "tainted": False,
                "hops": [],
                "dropped": [],
                "shadow_flagged": [],
                "skipped": [],
                "approvals": [],
                "declined_actions_in_task": 0,
            }
        guard_cfg = getattr(settings, "SECURITY_GUARD", None)
        # The summary only hides itself when nothing was enforced, i.e. both checkpoints are in shadow.
        shadow = (guard_cfg.input_mode == "shadow" and guard_cfg.tool_mode == "shadow") if guard_cfg else True
        return {
            "tainted": bool(sec.get("tainted")),
            "shadow": shadow,
            "hops": list(sec.get("hops") or []),
            "dropped": list(sec.get("dropped") or []),
            "shadow_flagged": list(sec.get("shadow_flagged") or []),
            "skipped": list(sec.get("skipped") or []),
            "approvals": list(sec.get("approvals") or []),
            "declined_actions_in_task": int(sec.get("declined_actions_in_task", 0)),
        }

    @classmethod
    def _guard_failure(cls, stop: BaseException, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Result for a task a checkpoint stopped (SEC-046): `escalated` or `declined`, final."""
        decision = stop.decision  # type: ignore[attr-defined]
        code = "escalated" if decision.outcome == "escalate" else "declined"
        return {"error": decision.reason, "error_code": code, "security": cls._security_block(context)}

    def _budget_failure(self, task_id: str, task_info: TaskInfo, exc: TaskBudgetExceeded) -> None:
        """Mark a run that used its token budget (IMP-042) as failed with error_code "budget"."""
        logger.warning(f"Task {task_id} stopped: {exc}")
        task_info.status = "error"
        task_info.completed_at = datetime.now()
        task_info.message = BUDGET_MESSAGE
        task_info.error = BUDGET_MESSAGE
        task_info.current_step = "error"
        self.task_results[task_id] = {"error": BUDGET_MESSAGE, "error_code": "budget"}
        self.task_errors[task_id] = {"message": BUDGET_MESSAGE, "error_code": "budget"}
        track_task_completion(
            task_id, (task_info.completed_at - task_info.created_at).total_seconds(), "error", self._origin(task_id)
        )

    def _save_final_record(
        self, task_id: str, message: str, task_info: TaskInfo, start_time: datetime,
        context: Optional[Dict[str, Any]], extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save a finished run's record (skipped by result_store if a stop already wrote one).

        The result gets the route fields, `usage` and, for scheduled runs, `origin` and `topic_id`;
        `extra` adds fields such as `standalone_query`. GET /api/task returns them in `result`.
        """
        meta = self._result_meta(context)
        fields = self._run_fields(task_id)
        result = dict(self.task_results.get(task_id) or {})
        result.update(meta)
        result.update(fields)
        result.update(extra or {})
        self.task_results[task_id] = result
        payload = {
            "task_id": task_id,
            "query": message,
            "status": task_info.status,
            "created_at": task_info.created_at,
            "completed_at": task_info.completed_at or datetime.now(),
            "duration_seconds": round((datetime.now() - start_time).total_seconds(), 2),
            "tokens_used": task_budget.used(),
            "result": result,
            **meta,
            **fields,
        }
        if result.get("error_code"):
            payload["error_code"] = result["error_code"]
        result_store.save_final(task_id, **payload)

    async def run_direct(
        self, task_id: str, message: str, context: Optional[Dict[str, Any]] = None,
        *, origin: Optional[str] = None, topic_id: Optional[str] = None, token_budget: Optional[int] = None,
    ):
        """One text-only LLM call without building the LangGraph workflow."""
        run_tokens = self._enter_run(task_id, origin, topic_id, token_budget)
        start_time = datetime.now()
        ctx = context or {}
        task_info = TaskInfo(
            task_id=task_id,
            status="processing",
            created_at=datetime.now(),
            message="Direct response started",
            query=message,
            current_step="direct",
        )
        self.active_tasks[task_id] = task_info
        track_task_start(task_id, self._origin(task_id))
        try:
            supervisor = SupervisorAgent()
            content = await supervisor.answer_directly(
                message, ctx.get("previous_messages", []), image_attached=bool(ctx.get("had_image"))
            )
            report = {"title": "Response", "content": content, "type": "direct_response"}
            self.task_results[task_id] = report
            task_info.status = "completed"
            task_info.completed_at = datetime.now()
            task_info.message = "Task completed successfully"
            task_info.current_step = "completed"
            duration = (datetime.now() - start_time).total_seconds()
            track_task_completion(task_id, duration, "completed", self._origin(task_id))
        except (GuardStop, GuardDeclined) as stop:
            # GuardStop/GuardDeclined derive from BaseException, so they must be caught explicitly.
            task_info.status = "error"
            task_info.error = stop.decision.reason
            self.task_results[task_id] = self._guard_failure(stop, ctx)
            duration = (datetime.now() - start_time).total_seconds()
            track_task_completion(task_id, duration, "error", self._origin(task_id))
        except TaskBudgetExceeded as exc:
            self._budget_failure(task_id, task_info, exc)
        except Exception as e:
            logger.exception("Direct task %s failed: %s", task_id, e)
            task_info.status = "error"
            task_info.error = str(e)
            self.task_results[task_id] = {"error": str(e)}
            duration = (datetime.now() - start_time).total_seconds()
            track_task_completion(task_id, duration, "error", self._origin(task_id))
        finally:
            if task_id in self.active_tasks:
                self.active_tasks[task_id] = task_info
            if task_info.status in ("completed", "error"):
                self._save_final_record(task_id, message, task_info, start_time, ctx)
            self._exit_run(task_id, run_tokens)

    @timing_decorator
    async def run_workflow(
        self, task_id: str, message: str, context: Optional[Dict[str, Any]] = None,
        *, origin: Optional[str] = None, topic_id: Optional[str] = None, token_budget: Optional[int] = None,
    ):
        """
        Run a workflow task asynchronously.
        
        Args:
            task_id: The unique ID for this task
            message: The user's message/query
            context: Optional additional context
            origin, topic_id, token_budget: set for a scheduled topic run (REC-108); its budget
                is token_budget instead of the global one
        """
        run_tokens = self._enter_run(task_id, origin, topic_id, token_budget, search_cap=_search_cap())
        start_time = datetime.now()
        extra: Dict[str, Any] = {}
        run_config: Optional[Dict[str, Any]] = None
        if origin == "schedule":
            # LangSmith metadata, so scheduled runs can be told apart (REC-108).
            run_config = {"metadata": {"origin": "schedule", "topic_id": topic_id}}
        
        # Create task info
        task_info = TaskInfo(
            task_id=task_id,
            status="processing",
            created_at=datetime.now(),
            message="Task started",
            query=message,
            current_step="initializing"
        )
        
        # Store in active tasks
        self.active_tasks[task_id] = task_info
        
        # Track task start for metrics
        track_task_start(task_id, self._origin(task_id))
        
        try:
            # Get the workflow
            workflow = get_agent_workflow()
            
            # Initialize the state
            ctx = context or {}
            ctx = {**ctx, "task_id": task_id}
            # Create the security ledger up front and share the same dict with the state, so a task
            # a checkpoint stops still reports what happened before the stop (SEC-082).
            from app.security import ledger as security_ledger

            ctx["security"] = security_ledger.get_ledger(WorkflowState(query=message, context={}))
            # The user's own words, the reference for checking agent subgoals and tool calls
            # (checkpoint2-intent SEC-100); its embedding and harm are computed on first use.
            from app.security import intent as security_intent

            security_intent.record_intent(WorkflowState(query=message, context=ctx))
            inputs: Dict[str, Any] = {}
            if ctx.get("document_id"):
                inputs["document_id"] = ctx["document_id"]
            initial_state = WorkflowState(
                query=message,
                context=ctx,
                inputs=inputs,
                current_step="planning",
            )
            
            # Update task info
            task_info.current_step = "planning"
            self.active_tasks[task_id] = task_info
            
            # Run the workflow
            logger.info(f"Starting workflow for task {task_id}")
            result = await self._stream_with_progress(task_id, workflow, initial_state, run_config)
            final_state = (
                result
                if isinstance(result, WorkflowState)
                else WorkflowState.model_validate(result)
            )
            standalone = _standalone_query(final_state.context)
            if standalone is not None:
                extra["standalone_query"] = standalone

            # Calculate duration for metrics
            duration = (datetime.now() - start_time).total_seconds()

            # Log completion
            logger.info(f"Task {task_id} finished workflow in {duration:.2f} seconds")

            security = self._security_block(final_state.context)
            if final_state.final_report:
                task_info.status = "completed"
                task_info.completed_at = datetime.now()
                task_info.message = "Task completed successfully"
                task_info.current_step = "completed"
                logger.info(
                    f"Storing final report for task {task_id} "
                    f"(content length: {len(final_state.final_report.get('content', ''))})"
                )
                report = dict(final_state.final_report)
                report["security"] = security
                self.task_results[task_id] = report
                track_task_completion(task_id, duration, "completed", self._origin(task_id))
            else:
                task_info.status = "error"
                task_info.completed_at = datetime.now()
                error_msg = final_state.error or "Workflow ended without a report"
                task_info.message = f"Task failed: {error_msg}"
                task_info.error = error_msg
                task_info.current_step = "error"
                logger.warning(f"Workflow for task {task_id} ended without a final report")
                self.task_results[task_id] = {"error": error_msg, "security": security}
                self.task_errors[task_id] = {
                    "message": error_msg,
                    "workflow_error": final_state.error,
                }
                track_task_completion(task_id, duration, "error", self._origin(task_id))

        except (GuardStop, GuardDeclined) as stop:
            duration = (datetime.now() - start_time).total_seconds()
            task_info.status = "error"
            task_info.completed_at = datetime.now()
            task_info.message = stop.decision.reason
            task_info.error = stop.decision.reason
            task_info.current_step = "error"
            self.task_results[task_id] = self._guard_failure(stop, ctx)
            self.task_errors[task_id] = {
                "message": stop.decision.reason,
                "error_code": self.task_results[task_id]["error_code"],
                "policy_ids": list(stop.decision.policy_ids),
            }
            track_task_completion(task_id, duration, "error", self._origin(task_id))

        except TaskBudgetExceeded as exc:
            self._budget_failure(task_id, task_info, exc)

        except Exception as e:
            # Calculate duration for metrics
            duration = (datetime.now() - start_time).total_seconds()
            
            # Log the full exception
            logger.exception(f"Error in workflow for task {task_id}: {str(e)}")
            
            # Update task info
            task_info.status = "error"
            task_info.message = f"Task failed: {str(e)}"
            task_info.error = str(e)
            
            # Store detailed error information
            error_detail = {
                "error": str(e),
                "traceback": traceback.format_exc(),
                "timestamp": datetime.now().isoformat()
            }
            self.task_errors[task_id] = error_detail
            
            # Store the error message in results
            self.task_results[task_id] = {"error": str(e)}
            
            # Track error for metrics
            track_task_completion(task_id, duration, "error", self._origin(task_id))
            
        finally:
            # Always update the active tasks
            if task_id in self.active_tasks:
                self.active_tasks[task_id] = task_info
            if task_info.status in ("completed", "error"):
                self._save_final_record(task_id, message, task_info, start_time, context, extra)
            self._exit_run(task_id, run_tokens)

    @staticmethod
    async def _stream_with_progress(
        task_id: str, workflow: Any, initial_state: WorkflowState, config: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Run the graph, saving progress after each finished node; returns the final state.

        `config` (scheduled runs only) carries LangSmith metadata; interactive runs pass none.
        """
        final_values = None
        stream_kwargs: Dict[str, Any] = {"stream_mode": ["updates", "values"]}
        if config:
            stream_kwargs["config"] = config
        async for mode, chunk in workflow.astream(initial_state, **stream_kwargs):
            if mode == "values":
                final_values = chunk
            elif mode == "updates" and isinstance(chunk, dict) and chunk:
                node = next(iter(chunk))
                logs = progress.logs_of(chunk[node]) or progress.logs_of(final_values)
                progress.save(task_id, node, logs)
        if final_values is None:
            raise RuntimeError("Workflow ended without a state")
        return final_values
    
    def get_task_status(self, task_id: str) -> Tuple[str, Optional[Dict[str, Any]]]:
        """
        Get the status and result of a task.
        
        Args:
            task_id: The task ID
            
        Returns:
            Tuple of (status, result)
        """
        if task_id not in self.active_tasks and task_id not in self.task_results:
            saved = result_store.load_result(task_id)
            if saved is None:
                return "not_found", None
            return self._status_from_saved(saved)
        
        if task_id in self.active_tasks:
            task_info = self.active_tasks[task_id]
            return task_info.status, self.task_results.get(task_id)
        
        return "completed", self.task_results.get(task_id)
    
    def task_exists(self, task_id: str) -> bool:
        """Check if a task exists in this worker or in the shared result store."""
        return (
            task_id in self.active_tasks
            or task_id in self.task_results
            or result_store.load_result(task_id) is not None
        )
    
    def list_tasks(self) -> List[TaskInfo]:
        """List all tasks."""
        return list(self.active_tasks.values())
    
    def get_task_info(self, task_id: str) -> Optional[TaskInfo]:
        """Task info from this worker, or from the shared saved record (a task another worker ran,
        or one from before a restart)."""
        info = self.active_tasks.get(task_id)
        if info is not None:
            return info
        saved = result_store.load_result(task_id)
        if saved is None:
            return None
        status, result = self._status_from_saved(saved)
        error = result.get("error") if status == "error" and isinstance(result, dict) else None
        return TaskInfo(
            task_id=task_id,
            status=status,
            created_at=_saved_time(saved.get("created_at")) or datetime.now(),
            completed_at=_saved_time(saved.get("completed_at")),
            message=f"Task {status}",
            query=str(saved.get("query") or ""),
            error=error,
        )

    def get_task_error(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Error details from this worker, or from the shared saved record."""
        if task_id in self.task_errors:
            return self.task_errors[task_id]
        saved = result_store.load_result(task_id)
        if saved is None:
            return None
        status, result = self._status_from_saved(saved)
        if status != "error" or not isinstance(result, dict):
            return None
        details = {"message": result.get("error") or "Unknown error"}
        if result.get("error_code"):
            details["error_code"] = result["error_code"]
        return details
    
    def delete_task(self, task_id: str) -> bool:
        """Delete a task from this worker's memory and its result files (seen by every worker)."""
        self.active_tasks.pop(task_id, None)
        self.task_results.pop(task_id, None)
        self.task_errors.pop(task_id, None)
        result_store.delete_result(task_id)
        return True
