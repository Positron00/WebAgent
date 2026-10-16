"""The user's intent and the plan check (checkpoint2-intent spec SEC-100 – SEC-102, SEC-107).

The intent is the user's own text, fixed when the research workflow starts. Agent-written subgoals
(the supervisor's rewrite and plan questions, the senior research agent's follow-up questions)
are decided against it before they act. New rules are shadow-only, so in shadow mode nothing is
removed; in act mode a declined rewrite falls back to the user's turn and a declined subgoal is
dropped and counted.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.core.config import settings
from app.core.metrics import record_guard_decision
from app.models.task import WorkflowState
from app.security import ledger, records, signals
from app.security import judge as guard_judge
from app.security.decision import decide
from app.security.errors import GuardStop
from app.security.types import Decision

logger = logging.getLogger(__name__)

INTENT_MAX_CHARS = 2000
PREVIOUS_USER_TURNS = 3
_INTENT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "security" / "intent.json"
_PLAN_FLAGS = {
    "tool::web_search": "requires_web_search",
    "tool::internal_retrieve": "requires_internal_knowledge",
    "tool::document_extract": "requires_document_extraction",
}


def _cfg():
    return getattr(settings, "SECURITY_GUARD", None)


def plan_check_enabled() -> bool:
    cfg = _cfg()
    return bool(cfg and cfg.enabled and getattr(cfg, "plan_check", True))


def build_intent_text(query: str, previous: Optional[List[Dict[str, Any]]]) -> str:
    turns = [str(t.get("content") or "") for t in (previous or []) if isinstance(t, dict) and t.get("role") == "user"]
    text = "\n".join([*turns[-PREVIOUS_USER_TURNS:], query or ""]).strip()
    return text[-INTENT_MAX_CHARS:]


def record_intent(state: WorkflowState) -> None:
    """SEC-100: store the intent once, when the research workflow's state is built. Never raises.

    Only the text is stored here; its embedding and harm score are computed on first use and cached
    in the same record (``signals.intent_embedding`` / ``intent_harm``), so starting a task costs
    nothing and a task that never calls a tool never loads a model."""
    cfg = _cfg()
    if cfg is None or not cfg.enabled:
        return
    text = build_intent_text(state.query, state.context.get("previous_messages"))
    ledger.get_ledger(state)["intent"] = {"text": text, "turn": state.query or ""}


def intent_text(state: Optional[WorkflowState]) -> str:
    if state is None:
        return ""
    record = ledger.get_ledger(state).get("intent") or {}
    return record.get("text") or build_intent_text(state.query, state.context.get("previous_messages"))


@lru_cache(maxsize=1)
def _code_patterns() -> Tuple[re.Pattern, ...]:
    try:
        data = json.loads(_INTENT_CONFIG.read_text(encoding="utf-8"))
        return tuple(re.compile(p, re.IGNORECASE) for p in data.get("code_request", []))
    except (OSError, ValueError, re.error) as exc:
        logger.warning("intent.json unreadable (%s); no code-request keywords", exc)
        return ()


def code_requested(text: str) -> bool:
    return any(p.search(text or "") for p in _code_patterns())


def declared_for(state: WorkflowState, action: str) -> bool:
    """SEC-107: whether the plan (or, for code, the intent) asked for this tool; true without a plan."""
    if action == "tool::execute_code":
        return code_requested(intent_text(state))
    plan = state.context.get("research_plan")
    flag = _PLAN_FLAGS.get(action)
    if not isinstance(plan, dict) or flag is None:
        return True
    return bool(plan.get(flag, True))


def _decide_text(state: WorkflowState, action: str, text: str, origin: str, principal: str) -> Tuple[str, Decision]:
    """Decide one rewrite or subgoal; returns (effective outcome, decision). Raises GuardStop on escalate."""
    cfg = _cfg()
    shadow = cfg.tool_mode == "shadow"
    sig = signals.score_text(state, text)
    context: Dict[str, Any] = {
        "source": "plan",
        "subgoal_origin": origin,
        "signals_available": sig.available,
        "intent_harm": signals.intent_harm(state),
        "tainted": ledger.is_tainted(state),
        "declined_actions_in_task": ledger.declined_actions(state),
        "low_alignment_calls": ledger.low_alignment_calls(state),
        "env": settings.WEBAGENT_ENV,
    }
    if action == "plan::rewrite":
        context["rewrite_alignment"] = sig.intent_alignment
    else:
        context["intent_alignment"] = sig.intent_alignment
        context["subgoal_harm"] = sig.harm
    resource = str(state.context.get("task_id") or "task-unknown")
    decision = decide(principal, action, resource, context, shadow=shadow)
    record_guard_decision(2, action, decision)
    dev = settings.WEBAGENT_ENV == "dev"
    records.record_collection(
        checkpoint=2, action=action, decision=decision, state_context=state.context,
        args={"text": text[:500]} if dev else {},
        context={k: context.get(k) for k in ("subgoal_origin", "signals_available", "intent_harm",
                                              "rewrite_alignment", "intent_alignment", "subgoal_harm")
                 if k in context},
    )
    effective = decision.outcome
    if decision.error or shadow:
        effective = "allow"  # shadow mode, or a decision error: the plan check fails open
    if effective == "verify":
        effective = "allow" if guard_judge.is_aligned(state, action, {"text": text}) else "decline"
    logger.info(
        "guard plan action=%s origin=%s outcome=%s effective=%s policies=%s shadow_policies=%s task=%s shadow=%s%s",
        action, origin, decision.outcome, effective, ",".join(decision.policy_ids) or "-",
        ",".join(decision.shadow_policy_ids) or "-", resource, shadow,
        f' text="{text[:200]}"' if dev else "",
    )
    if effective == "escalate":
        records.record_escalation(hop=principal, action=action, decision=decision, state_context=state.context,
                                  args={"text": text[:200]} if dev else {},
                                  scores={k: context.get(k) for k in ("intent_alignment", "subgoal_harm",
                                                                      "rewrite_alignment") if k in context})
        raise GuardStop(decision)
    return effective, decision


def _filter(state: WorkflowState, items: List[Any], origin: str, principal: str) -> List[Any]:
    kept: List[Any] = []
    for item in items:
        if not isinstance(item, str) or not item.strip():
            kept.append(item)
            continue
        effective, decision = _decide_text(state, "plan::subgoal", item, origin, principal)
        if effective == "allow":
            kept.append(item)
            continue
        entry: Dict[str, Any] = {"action": "plan::subgoal", "origin": origin, "outcome": decision.outcome,
                                 "policy_ids": list(decision.policy_ids), "reason": decision.reason,
                                 "shadow": False}
        if settings.WEBAGENT_ENV == "dev":
            entry["text"] = item[:200]
        ledger.record_skipped(state, entry)
        ledger.increment_declined_actions(state)
    return kept


async def check_plan(state: WorkflowState, principal: str = "supervisor") -> None:
    """SEC-101, SEC-102: the supervisor's rewrite and plan questions, after `research_plan` is set."""
    cfg = _cfg()
    if cfg is not None and cfg.enabled:
        # Score the intent now, during the supervisor step, so checkpoint 2's first call doesn't.
        await asyncio.to_thread(signals.warm_intent, state)
    if not plan_check_enabled():
        return
    plan = state.context.get("research_plan")
    if not isinstance(plan, dict):
        return
    try:
        rewrite = (plan.get("standalone_query") or "").strip()
        if rewrite and rewrite != (state.query or "").strip():
            effective, _ = await asyncio.to_thread(_decide_text, state, "plan::rewrite", rewrite, "plan", principal)
            if effective != "allow":
                plan["standalone_query"] = ""  # searches fall back to the user's turn
        questions = list(plan.get("questions") or [])
        if questions:
            plan["questions"] = await asyncio.to_thread(_filter, state, questions, "plan", principal)
    except Exception as exc:  # noqa: BLE001 - fail open; GuardStop (a BaseException) still propagates
        logger.warning("Plan check failed; plan unchanged: %s", exc)


async def check_follow_ups(state: WorkflowState, questions: List[Any], principal: str = "senior_research") -> List[Any]:
    """SEC-102: the senior research agent's follow-up questions, before they are stored."""
    if not plan_check_enabled() or not questions:
        return questions
    try:
        return await asyncio.to_thread(_filter, state, list(questions), "follow_up", principal)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Follow-up check failed; questions unchanged: %s", exc)
        return questions
