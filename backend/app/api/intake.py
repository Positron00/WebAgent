"""Checkpoint 1 intake guard shared by /api/task and /api/v1/chat/completions (SEC-033, IMP-020)."""
from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.models.task import WorkflowState
from app.security import tokens
from app.security.errors import GuardStop
from app.security.hop_guard import check_hop, items_from_task_request


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


def run_intake_guard(
    query: str,
    history: List[Dict[str, str]],
    has_image: bool,
    consent_token: Optional[str],
    task_id: str,
) -> bool:
    """Checkpoint 1 on the user turn and history, before the task exists.

    Returns whether the user consented (the consent token is redeemed here, once).
    Raises GuardRefused (409 consent, 403 decline or escalate) in act mode.
    """
    cfg = getattr(settings, "SECURITY_GUARD", None)
    if cfg is None or not cfg.enabled:
        return False
    binding = tokens.consent_binding(query, history)
    consented = bool(consent_token and tokens.redeem(consent_token, "consent", binding))
    items = items_from_task_request(query, history, has_image=has_image)
    state = WorkflowState(
        query=query,
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
