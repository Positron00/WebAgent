"""Task API compatible with the Next.js web client (/api/task)."""
import logging
from typing import Any, Dict, List, Literal, Optional
from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError
from starlette.concurrency import run_in_threadpool

from app.api.intake import GuardRefused, refusal_body, run_intake_guard  # noqa: F401 (refusal_body: guard.py)
from app.core.api_errors import error_body
from app.core.config import settings
from app.core.metrics import record_triage_decision
from app.core.scheduler_auth import scheduler_header_ok
from app.security import tokens
from app.services import result_store, run_origin, triage_collect
from app.services.routing import choose_route, resolve_mode
from app.services.task_manager import TaskManager
from app.services.triage import TriageDecision, classify

logger = logging.getLogger(__name__)
router = APIRouter()
task_manager = TaskManager()


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=20000)


class TaskRequest(BaseModel):
    query: str = Field(max_length=20000)
    image: Optional[str] = None
    agentic: Optional[bool] = None
    mode: Optional[Literal["off", "auto", "always"]] = None
    document_id: Optional[str] = None
    previous_messages: List[ChatTurn] = Field(default_factory=list, max_length=8)
    triage_override: bool = False
    consent_token: Optional[str] = None
    approval_tokens: List[str] = Field(default_factory=list)
    # Scheduler-only fields (REC-107): accepted only with a valid X-WebAgent-Scheduler header.
    origin: Optional[Literal["schedule"]] = None
    topic_id: Optional[str] = None
    task_id: Optional[str] = None
    token_budget: Optional[int] = None


SCHEDULER_FIELDS = ("origin", "topic_id", "task_id", "token_budget")


def _refuse(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"code": code, **error_body(code, message)})


def _uuid_or_none(value: Optional[str]) -> Optional[str]:
    try:
        return str(UUID(str(value)))
    except (TypeError, ValueError, AttributeError):
        return None


def _check_scheduler_fields(req: TaskRequest, header_ok: bool) -> Optional[JSONResponse]:
    """REC-107/REC-108 rules for a request that sets a scheduler field; a refusal, or None.

    The refusals name fields, never values (the token and the question stay out of replies and logs).
    """
    if not header_ok:  # also when WEBAGENT_SCHEDULER_TOKEN is unset
        return _refuse(422, "scheduler_fields_need_token",
                       "origin, topic_id, task_id and token_budget need the scheduler credential")
    if req.origin != "schedule":
        return _refuse(422, "scheduler_field_invalid", "Scheduler fields need origin 'schedule'")
    if req.mode != "always":
        return _refuse(422, "scheduler_field_invalid", "A scheduled task needs mode 'always'")
    if _uuid_or_none(req.topic_id) is None:
        return _refuse(422, "scheduler_field_invalid", "topic_id must be a UUID")
    if req.task_id is not None and _uuid_or_none(req.task_id) is None:
        return _refuse(422, "scheduler_field_invalid", "task_id must be a UUID")
    cap = settings.SCHEDULE.max_token_budget
    budget = req.token_budget
    if budget is None or isinstance(budget, bool) or not 1 <= budget <= cap:
        return _refuse(422, "scheduler_field_invalid", f"token_budget must be between 1 and {cap}")
    return None


def last_user_question(previous: List[ChatTurn]) -> Optional[str]:
    for turn in reversed(previous):
        if turn.role == "user" and turn.content.strip():
            return turn.content
    return None


def _intake_guard(req: TaskRequest, task_id: str, topic_id: Optional[str] = None) -> bool:
    history = [t.model_dump() for t in req.previous_messages]
    # A scheduled run's intake decisions are collected with origin and topic_id (REC-108).
    topic_token = run_origin.topic_run.set(topic_id)
    try:
        return run_intake_guard(req.query, history, bool(req.image), req.consent_token, task_id)
    finally:
        run_origin.topic_run.reset(topic_token)


def _approved_tools(req: TaskRequest) -> List[str]:
    """Tool actions approved for this re-run (SEC-051): tokens bound to the same question."""
    approved: List[str] = []
    binding = {"query_hash": tokens.query_hash(req.query)}
    for token in req.approval_tokens:
        payload = tokens.redeem(token, "approval", binding)
        if payload and payload.get("action"):
            approved.append(str(payload["action"]))
    return list(dict.fromkeys(approved))


@router.post("")
async def create_task(req: TaskRequest, request: Request):
    return await _create_task(req, scheduler_ok=scheduler_header_ok(request))


async def _create_task(req: TaskRequest, scheduler_ok: bool = False):
    scheduled = any(getattr(req, name) is not None for name in SCHEDULER_FIELDS)
    topic_id: Optional[str] = None
    if scheduled:
        refusal = _check_scheduler_fields(req, scheduler_ok)
        if refusal is not None:
            return refusal
        topic_id = _uuid_or_none(req.topic_id)
        task_id = _uuid_or_none(req.task_id) or str(uuid4())
        if req.task_id is not None and task_manager.task_exists(task_id):
            return _refuse(409, "task_exists", "A task with this task_id already exists")
        # A scheduled run has no previous turns (REC-108).
        req = req.model_copy(update={"previous_messages": []})
    else:
        task_id = str(uuid4())
    prev_q = last_user_question(req.previous_messages)
    has_image = bool(req.image)

    try:
        consented = await run_in_threadpool(_intake_guard, req, task_id, topic_id)
    except GuardRefused as refused:
        return JSONResponse(status_code=refused.status, content=refused.body)

    context: Dict[str, Any] = {
        "task_id": task_id,
        "previous_messages": [t.model_dump() for t in req.previous_messages],
    }
    if consented:
        context["consented"] = True
    approved = _approved_tools(req) if req.approval_tokens else []
    if approved:
        context["approved_tools"] = approved
    if req.document_id:
        context["document_id"] = req.document_id
    if has_image:
        context["had_image"] = True

    triage_info: Optional[Dict[str, Any]] = None
    use_direct = False

    if scheduled:
        # Always mode; triage is neither run nor collected (REC-108).
        context["agentic_toggle"] = True
        context["chat_mode"] = "always"
    elif req.triage_override:
        override = TriageDecision("agentic", None, "override", "Research in depth", "none", 0.0)
        record_triage_decision(override)
        triage_collect.record("override", req.query, None, prev_q)
        triage_info = override.to_dict()
        context["agentic_toggle"] = True
        context["chat_mode"] = "always"
    else:
        resolved = resolve_mode(req.mode, req.agentic)
        context["chat_mode"] = resolved
        decision = await run_in_threadpool(choose_route, resolved, req.query, prev_q, has_image)
        triage_info = decision.to_dict()
        use_direct = decision.route == "direct"
        context["agentic_toggle"] = not use_direct

        # Shadow mode labels requests the caller sent to the workflow itself; it never changes the route.
        cfg = getattr(settings, "TRIAGE", None)
        if cfg and cfg.mode == "shadow" and resolved == "always":
            try:
                shadow = await run_in_threadpool(classify, req.query, prev_q, has_image)
                triage_collect.record("shadow", req.query, shadow, prev_q)
                context["shadow_triage"] = {**shadow.to_dict(), "shadow": True}
            except Exception as exc:
                logger.warning("Shadow triage failed; running the task anyway: %s", exc)

    context["triage"] = triage_info

    task_manager.register_pending(task_id, req.query)
    if scheduled:
        logger.info(f"Scheduled task {task_id} started for topic {topic_id}")
        task_manager.start(
            task_id, req.query, context, direct=False,
            origin="schedule", topic_id=topic_id, token_budget=req.token_budget,
        )
    else:
        task_manager.start(task_id, req.query, context, direct=use_direct)
    return {"task_id": task_id, "status": "processing"}


async def start_legacy_task(query: Any, context: Any = None):
    """Start a task for the deprecated POST /api/v1/chat/ and /api/v1/tasks/create routes.

    They take the same path as /api/task in Always mode: intake guard, length limit, run
    supervisor. Caller context is not trusted (it could claim consent or approved tools);
    only a document ID is kept. Returns create_task's reply: a dict, or the refusal response.
    """
    document_id = context.get("document_id") if isinstance(context, dict) else None
    try:
        req = TaskRequest(
            query=query,
            mode="always",
            document_id=str(document_id) if document_id else None,
        )
    except ValidationError as exc:
        raise RequestValidationError(exc.errors()) from exc
    return await _create_task(req)


@router.get("/{task_id}")
async def task_status(task_id: str) -> Dict[str, Any]:
    status, result = task_manager.get_task_status(task_id)
    if status == "not_found":
        return {"status": "failed", "error": "Task not found"}
    client_status = "failed" if status == "error" else status
    body: Dict[str, Any] = {"status": client_status, "result": result, "error": (result or {}).get("error")}
    if isinstance(result, dict) and result.get("error_code"):
        body["error_code"] = result["error_code"]  # e.g. "escalated" (SEC-046); final for the web app
    if status == "processing":
        # Read from the file: the run may be on another worker (IMP-052).
        saved = result_store.load_result(task_id) or {}
        if saved.get("progress"):
            body["progress"] = saved["progress"]
    return body


@router.post("/{task_id}/cancel", status_code=202)
async def cancel_task(task_id: str) -> Dict[str, str]:
    """Ask the worker running a task to stop it (IMP-050); it stops within a few seconds."""
    saved = result_store.load_result(task_id)
    if saved is None:
        raise HTTPException(status_code=404, detail="Task not found")
    # A stale "processing" record (its worker died) counts as finished too.
    if TaskManager._status_from_saved(saved)[0] != "processing":
        raise HTTPException(status_code=409, detail="Task already finished")
    if not result_store.request_cancel(task_id):
        raise HTTPException(status_code=500, detail="Could not request the cancel")
    logger.info(f"Cancel requested for task {task_id}")
    return {"status": "cancelling"}
