"""Direct chat guard endpoint (SEC-034)."""
from __future__ import annotations

import logging
from typing import List, Literal, Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.core.config import settings
from app.security.errors import GuardStop
from app.security.hop_guard import check_hop, items_from_task_request
from app.security import tokens
from app.security.types import SEVERITY
from app.models.task import WorkflowState
from app.api.endpoints.task import refusal_body

logger = logging.getLogger(__name__)
router = APIRouter()


class GuardMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str = Field(max_length=100_000)


class GuardInputRequest(BaseModel):
    messages: List[GuardMessage] = Field(min_length=1, max_length=60)
    has_image: bool = False
    consent_token: Optional[str] = None


class GuardInputResponse(BaseModel):
    outcome: Literal["allow", "verify", "consent", "decline", "escalate"]
    reason: str
    policy_ids: List[str]
    consent_token: Optional[str] = None
    decision_id: str


def _last_user(messages: List[GuardMessage]) -> str:
    for msg in reversed(messages):
        if msg.role == "user" and msg.content.strip():
            return msg.content
    raise HTTPException(status_code=422, detail="No user message found")


def _history(messages: List[GuardMessage], user_text: str) -> List[dict]:
    rows: List[dict] = []
    for msg in messages:
        if msg.role == "system":
            continue
        if msg.role == "user" and msg.content == user_text:
            continue
        rows.append({"role": msg.role, "content": msg.content})
    return rows


@router.post("/input", response_model=GuardInputResponse)
async def guard_input(req: GuardInputRequest):
    """Checkpoint 1 for the web app's direct chat (SEC-034).

    200 for allow/verify; 409 (consent) or 403 (decline/escalate) with a top-level body that has
    the response fields plus `code` (SEC-036), so both the guard client and apiClient can read it.
    """
    decision_id = f"chat-{uuid4()}"
    user_text = _last_user(req.messages)
    history = _history(req.messages, user_text)
    binding = tokens.consent_binding(user_text, history)
    consented = bool(req.consent_token and tokens.redeem(req.consent_token, "consent", binding))

    items = items_from_task_request(user_text, history, has_image=req.has_image)
    state = WorkflowState(
        query=user_text,
        context={"task_id": decision_id, "previous_messages": history, "consented": consented},
    )
    try:
        hop = await run_in_threadpool(check_hop, "web_chat", items, state, chat_id=decision_id)
    except GuardStop as exc:
        decision = exc.decision
        token = None
        status = 403
        if decision.outcome == "consent":
            status = 409
            token = tokens.issue("consent", binding, decision.policy_ids[0] if decision.policy_ids else "")
        body = {**refusal_body(decision, token), "decision_id": decision_id}
        return JSONResponse(status_code=status, content=body)

    if not hop.decisions:
        return GuardInputResponse(outcome="allow", reason="Allowed.", policy_ids=[], decision_id=decision_id)
    worst = max(hop.decisions, key=lambda d: SEVERITY.index(d.outcome))
    # Shadow mode records the would-be outcome (metrics, logs, collection) but the effective
    # outcome is allow; a 200 only ever carries allow or verify (SEC-034, SEC-062).
    outcome = "allow" if worst.shadow or worst.outcome not in ("allow", "verify") else worst.outcome
    return GuardInputResponse(
        outcome=outcome,
        reason=worst.reason,
        policy_ids=list(worst.policy_ids),
        decision_id=decision_id,
    )
