"""Checkpoint 2: guard tool actions before they run (SEC-040–SEC-047)."""
from __future__ import annotations

import asyncio
import time
import logging
from typing import Any, Dict, List, Optional, Set

from app.core.config import settings
from app.core.metrics import (GUARD_ALIGNMENT, GUARD_EGRESS_OVERLAP, GUARD_INTENT_ALIGNMENT,
                              GUARD_RISK_ABOVE_INTENT, GUARD_TOOL_SECONDS, record_guard_decision)
from app.models.task import WorkflowState
from app.security import alignment_model, code_checks, egress, intent, patterns, signals
from app.security.decision import decide
from app.security.errors import GuardStop
from app.security import judge as guard_judge
from app.security import ledger
from app.security import records
from app.security import tokens
from app.security.thresholds import DEFAULT_THRESHOLDS
from app.security.types import Decision, ToolAction
from app.utils.code_sandbox import sandbox_mode

logger = logging.getLogger(__name__)


def _guard_cfg():
    return getattr(settings, "SECURITY_GUARD", None)


def _resource_id(state: WorkflowState) -> str:
    return str(state.context.get("task_id") or "task-unknown")


def _approved_tools(state: WorkflowState) -> List[str]:
    ledger_data = ledger.get_ledger(state)
    from_ctx = state.context.get("approved_tools") or []
    from_ledger = ledger_data.get("approved_tools") or []
    return list(dict.fromkeys([*from_ctx, *from_ledger]))


def _scored_text(action: ToolAction, args: Dict[str, Any], scored_text: Optional[str]) -> str:
    """The text checked against the intent: the query, or the coding assistant's explanation."""
    if action in ("tool::web_search", "tool::internal_retrieve"):
        return str(args.get("query") or "")
    if action == "tool::execute_code":
        return str(scored_text or "")
    return ""  # document extraction: an ID, checked by provenance (SEC-042)


CODE_CALL_CHARS = 1000  # the classifier reads code calls as the code itself, as in its training rows


def _intent_fields(action: ToolAction, text: str, state: WorkflowState, code: Optional[str] = None) -> Dict[str, Any]:
    """Checkpoint2-intent SEC-103 – SEC-108: the call against the user's intent."""
    fields: Dict[str, Any] = {
        "declared": intent.declared_for(state, action),
        "low_alignment_calls": ledger.low_alignment_calls(state),
    }
    if action == "tool::web_search":
        fields["egress_overlap"] = egress.overlap(state, text)
    if not text.strip():
        fields["intent_harm"] = signals.intent_harm(state)
        return fields
    # Phase A also fills the intent's harm the first time (in parallel); phase B scores one pair.
    sig = signals.score_text(state, text, action=action.split("::", 1)[-1],
                             classifier_text=code[:CODE_CALL_CHARS] if code else None)
    fields.update(
        intent_harm=signals.intent_harm(state),
        signals_available=sig.available,
        classifier_available=sig.classifier,
        intent_alignment=sig.intent_alignment,
        arg_harm=sig.harm,
        risk_above_intent=signals.call_risk(state, sig),
        task_alignment=sig.task_alignment,
        _p_aligned=sig.p_aligned,
    )
    return fields


def _build_context(
    action: ToolAction,
    args: Dict[str, Any],
    state: WorkflowState,
    *,
    tool_mode: str,
    scored_text: Optional[str] = None,
) -> Dict[str, Any]:
    pattern_hits: Set[str] = set()
    code_findings: Set[str] = set()
    align = 100

    user_turn = state.query
    extra = _intent_fields(action, _scored_text(action, args, scored_text), state,
                           code=str(args.get("code") or "") if action == "tool::execute_code" else None)
    task_alignment = extra.pop("task_alignment", None)
    if action in ("tool::web_search", "tool::internal_retrieve"):
        query = str(args.get("query") or "")
        pattern_hits = patterns.scan_args(query, user_turn=user_turn)
        # The same encode gave the task-text alignment; recompute only if the signals were unavailable.
        align = task_alignment if extra.get("signals_available", True) and task_alignment is not None \
            else alignment_for_action(action, args, state)
    elif action == "tool::document_extract":
        doc_id = args.get("document_id")
        task_doc = state.inputs.get("document_id")
        pattern_hits = patterns.scan_args(
            "",
            document_id=str(doc_id) if doc_id is not None else None,
            task_document_id=str(task_doc) if task_doc is not None else None,
        )
    elif action == "tool::execute_code":
        code = str(args.get("code") or "")
        code_findings = set(code_checks.find_findings(code))

    return {
        "source": "tool",
        "pattern_hits": sorted(pattern_hits),
        "alignment": align,
        "code_findings": sorted(code_findings),
        "tainted": ledger.is_tainted(state),
        "approved_tools": _approved_tools(state),
        "declined_actions_in_task": ledger.declined_actions(state),
        "mode": str(state.context.get("chat_mode") or "unknown"),
        "env": settings.WEBAGENT_ENV,
        "code_execution_enabled": bool(getattr(settings, "ENABLE_CODE_EXECUTION", False)),
        "sandbox": sandbox_mode(),
        "consented": bool(state.context.get("consented")),
        # A gated classifier's own intent thresholds (SEC-126), like checkpoint 1's ContextDefender ones.
        **alignment_model.current_thresholds(),
        **extra,
    }


# Verify rules on the intent signals: with the classifier, its uncertain band picks judge or side (SEC-127).
INTENT_VERIFY_RULES = frozenset({"off-intent-search-verify", "call-risk-above-intent-verify",
                                 "undeclared-tool-verify", "rewrite-drift-verify"})


def resolve_verify(state: WorkflowState, action: str, args: Dict[str, Any], decision: Decision,
                   p_aligned: Optional[float]) -> str:
    """"allow" or "decline" for a verify. With the judge on and the classifier's score present, a
    verify from the intent rules only goes to the judge inside the uncertain band (SEC-127)."""
    cfg = _guard_cfg()
    ids = set(decision.policy_ids)
    if cfg is not None and cfg.judge and p_aligned is not None and ids and ids <= INTENT_VERIFY_RULES:
        side = alignment_model.band_outcome(p_aligned)
        if side != "judge":
            return side
    return "allow" if guard_judge.is_aligned(state, action, args) else "decline"


# Collected with each checkpoint 2 decision (numbers and flags only; SEC-113).
_INTENT_FIELDS = ("intent_alignment", "arg_harm", "intent_harm", "risk_above_intent", "egress_overlap",
                  "declared", "low_alignment_calls", "signals_available", "classifier_available")


def alignment_for_action(action: str, args: Dict[str, Any], state: WorkflowState) -> int:
    from app.security import alignment

    return alignment.alignment_for_action(action, args, state)


def _observe_intent_signals(action: str, context: Dict[str, Any], state: WorkflowState) -> None:
    """Metrics for the intent signals (SEC-112) and the drift counter (SEC-108)."""
    if not context.get("signals_available", True):
        return
    if action in ("tool::web_search", "tool::internal_retrieve", "tool::execute_code"):
        GUARD_INTENT_ALIGNMENT.labels(action=action).observe(context.get("intent_alignment", 100))
        GUARD_RISK_ABOVE_INTENT.labels(action=action).observe(context.get("risk_above_intent", 0))
    if action == "tool::web_search":
        GUARD_EGRESS_OVERLAP.observe(context.get("egress_overlap", 0))
    if action in ("tool::web_search", "tool::internal_retrieve") and \
            context.get("intent_alignment", 100) < int(context.get("drift_align_at", DEFAULT_THRESHOLDS["drift_align_at"])):
        ledger.increment_low_alignment_calls(state)


def _log_decision(
    decision: Decision,
    action: ToolAction,
    state: WorkflowState,
    *,
    effective: str,
) -> None:
    extra = ""
    if settings.WEBAGENT_ENV == "dev":
        extra = f' query="{state.query[:200]}"'
    logger.info(
        "guard checkpoint=2 action=%s outcome=%s effective=%s policies=%s task=%s shadow=%s%s",
        action,
        decision.outcome,
        effective,
        ",".join(decision.policy_ids) or "-",
        _resource_id(state),
        decision.shadow,
        extra,
    )


def _record_skip(state: WorkflowState, action: ToolAction, args: Dict[str, Any], decision: Decision) -> None:
    entry = {
        "action": action,
        "args": {k: (v[:200] if isinstance(v, str) else v) for k, v in args.items()},
        "outcome": decision.outcome,
        "policy_ids": list(decision.policy_ids),
        "reason": decision.reason,
        "shadow": decision.shadow,
    }
    ledger.record_skipped(state, entry)
    if decision.outcome == "consent":
        policy_id = decision.policy_ids[0] if decision.policy_ids else ""
        approvals = ledger.get_ledger(state)["approvals"]
        if any(a.get("action") == action for a in approvals):
            return  # one approval per tool per task (SEC-051)
        approvals.append(
            {
                "action": action,
                "policy_ids": list(decision.policy_ids),
                "reason": decision.reason,
                # Bound to the task's question, not its id: "Approve and re-run" creates a new task.
                "approval_token": tokens.issue(
                    "approval", {"query_hash": tokens.query_hash(state.query)}, policy_id, action=action
                ),
            }
        )


async def guard_tool(
    action: ToolAction,
    args: Dict[str, Any],
    state: WorkflowState,
    *,
    principal: str = "agent",
    scored_text: Optional[str] = None,
) -> bool:
    """
    Run checkpoint 2 for a tool call.

    Returns True when the tool should run. Raises :class:`GuardStop` on escalate. Timed for SNFR-01.
    """
    started = time.perf_counter()
    try:
        return await _guard_tool(action, args, state, principal=principal, scored_text=scored_text)
    finally:
        GUARD_TOOL_SECONDS.labels(action=action).observe(time.perf_counter() - started)


async def _guard_tool(
    action: ToolAction,
    args: Dict[str, Any],
    state: WorkflowState,
    *,
    principal: str,
    scored_text: Optional[str] = None,
) -> bool:
    cfg = _guard_cfg()
    if cfg is None or not cfg.enabled:
        return True

    tool_mode = cfg.tool_mode
    shadow = tool_mode == "shadow"
    # Alignment encodes text and the judge may wait 3 s: keep both off the event loop.
    context = await asyncio.to_thread(_build_context, action, args, state, tool_mode=tool_mode,
                                      scored_text=scored_text)
    p_aligned = context.pop("_p_aligned", None)
    decision = decide(
        principal,
        action,
        _resource_id(state),
        context,
        shadow=shadow,
    )
    record_guard_decision(2, action, decision)
    if action in ("tool::web_search", "tool::internal_retrieve"):
        # The distribution real searches need before any alignment rule can act on untainted tasks.
        GUARD_ALIGNMENT.labels(tainted=str(bool(context.get("tainted"))).lower()).observe(context.get("alignment", 100))
    _observe_intent_signals(action, context, state)
    ledger.note_call(state, f"{action}: {_scored_text(action, args, None) or args.get('document_id', '')}")
    records.record_collection(
        checkpoint=2,
        action=action,
        decision=decision,
        state_context=state.context,
        args=args,
        context={k: context.get(k) for k in ("alignment", "tainted", "pattern_hits", "code_findings",
                                               "approved_tools", "declined_actions_in_task", *_INTENT_FIELDS)},
    )

    effective = decision.outcome
    if decision.error:
        # SEC-061: fail closed for code and documents, open for search and retrieval.
        effective = "decline" if action in ("tool::execute_code", "tool::document_extract") else "allow"
    if shadow:
        effective = "allow"

    if effective == "verify":
        effective = await asyncio.to_thread(resolve_verify, state, action, args, decision, p_aligned)

    _log_decision(decision, action, state, effective=effective)

    if effective == "allow":
        return True

    if shadow:
        return True

    if effective == "escalate":
        records.record_escalation(
            hop=principal,
            action=action,
            decision=decision,
            state_context=state.context,
            args={k: (v[:200] if isinstance(v, str) else v) for k, v in args.items()},
            scores={"alignment": context.get("alignment"), "pattern_hits": context.get("pattern_hits"),
                    "code_findings": context.get("code_findings")},
        )
        raise GuardStop(decision)

    if effective in ("decline", "consent"):
        _record_skip(state, action, args, decision)
        if effective == "decline":
            ledger.increment_declined_actions(state)
        return False

    return False
