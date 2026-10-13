"""Task API compatible with the Next.js web client (/api/task)."""
import logging
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.core.config import settings
from app.core.metrics import record_triage_decision
from app.models.task import WorkflowState
from app.security.errors import GuardStop
from app.security.hop_guard import check_hop, items_from_task_request
from app.security import tokens
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
    consent_token: Optional[str] = None
    approval_tokens: List[str] = Field(default_factory=list)


def last_user_question(previous: List[ChatTurn]) -> Optional[str]:
    for turn in reversed(previous):
        if turn.role == "user" and turn.content.strip():
            return turn.content
    return None


class GuardRefused(Exception):
    """The intake check refused the request; carries the HTTP status and body (SEC-036)."""

    def __init__(self, status: int, body: Dict[str, Any]):
        self.status = status
        self.body = body


def refusal_body(decision, consent_token: Optional[str] = None) -> Dict[str, Any]:
    """Top-level refusal body shared by /api/task and /api/guard/input (not wrapped in `detail`)."""
    code = {"consent": "consent_required", "escalate": "escalated"}.get(decision.outcome, "declined")
    body: Dict[str, Any] = {
        "code": code,
        "outcome": decision.outcome,
        "reason": decision.reason,
        "policy_id": decision.policy_ids[0] if decision.policy_ids else "",
        "policy_ids": list(decision.policy_ids),
    }
    if consent_token:
        body["consent_token"] = consent_token
    return body


def _intake_guard(req: TaskRequest, task_id: str) -> bool:
    """Checkpoint 1 on the user turn and history before the task exists (SEC-033).

    Returns whether the user consented (the consent token is redeemed here, once).
    Raises GuardRefused for consent, decline or escalate in act mode.
    """
    cfg = getattr(settings, "SECURITY_GUARD", None)
    if cfg is None or not cfg.enabled:
        return False
    history = [t.model_dump() for t in req.previous_messages]
    binding = tokens.consent_binding(req.query, history)
    consented = bool(req.consent_token and tokens.redeem(req.consent_token, "consent", binding))
    items = items_from_task_request(req.query, history, has_image=bool(req.image))
    state = WorkflowState(
        query=req.query,
        context={"task_id": task_id, "previous_messages": history, "consented": consented},
    )
    try:
        check_hop("intake", items, state)
    except GuardStop as exc:
        decision = exc.decision
        if decision.outcome == "consent":
            policy_id = decision.policy_ids[0] if decision.policy_ids else ""
            raise GuardRefused(409, refusal_body(decision, tokens.issue("consent", binding, policy_id)))
        raise GuardRefused(403, refusal_body(decision))
    return consented


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
async def create_task(req: TaskRequest, background_tasks: BackgroundTasks):
    task_id = str(uuid4())
    prev_q = last_user_question(req.previous_messages)
    has_image = bool(req.image)

    try:
        consented = await run_in_threadpool(_intake_guard, req, task_id)
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

    if req.triage_override:
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
    body: Dict[str, Any] = {"status": client_status, "result": result, "error": (result or {}).get("error")}
    if isinstance(result, dict) and result.get("error_code"):
        body["error_code"] = result["error_code"]  # e.g. "escalated" (SEC-046); final for the web app
    return body
