"""Task API compatible with the Next.js web client (/api/task)."""
import logging
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.core.config import settings
from app.core.metrics import record_triage_decision
from app.services import triage_collect
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
    query: str
    image: Optional[str] = None
    agentic: Optional[bool] = None
    mode: Optional[Literal["off", "auto", "always"]] = None
    document_id: Optional[str] = None
    previous_messages: List[ChatTurn] = Field(default_factory=list, max_length=8)
    triage_override: bool = False


def last_user_question(previous: List[ChatTurn]) -> Optional[str]:
    for turn in reversed(previous):
        if turn.role == "user" and turn.content.strip():
            return turn.content
    return None


@router.post("")
async def create_task(req: TaskRequest, background_tasks: BackgroundTasks) -> Dict[str, str]:
    task_id = str(uuid4())
    prev_q = last_user_question(req.previous_messages)
    has_image = bool(req.image)

    context: Dict[str, Any] = {
        "previous_messages": [t.model_dump() for t in req.previous_messages],
    }
    if req.document_id:
        context["document_id"] = req.document_id
    if has_image:
        context["had_image"] = True

    triage_info: Optional[Dict[str, Any]] = None
    use_direct = False

    if req.triage_override:
        override = TriageDecision("agentic", None, "override", "Research in depth", "none", 0.0)
        record_triage_decision(override)
        triage_collect.record("override", req.query, None, prev_q)
        triage_info = override.to_dict()
        context["agentic_toggle"] = True
    else:
        resolved = resolve_mode(req.mode, req.agentic)
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
    if use_direct:
        background_tasks.add_task(task_manager.run_direct, task_id, req.query, context)
    else:
        background_tasks.add_task(task_manager.run_workflow, task_id, req.query, context)
    return {"task_id": task_id, "status": "processing"}


@router.get("/{task_id}")
async def task_status(task_id: str) -> Dict[str, Any]:
    status, result = task_manager.get_task_status(task_id)
    if status == "not_found":
        return {"status": "failed", "error": "Task not found"}
    client_status = "failed" if status == "error" else status
    return {"status": client_status, "result": result, "error": (result or {}).get("error")}
