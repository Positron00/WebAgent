"""Checkpoint 1: score context items before each LLM call (SEC-030–SEC-039)."""
from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.core.config import settings
from app.core.metrics import GUARD_HOP_SECONDS, record_guard_decision
from app.models.task import WorkflowState
from app.security import context_defender, judge, ledger, patterns, records
from app.security.decision import decide
from app.security.errors import GuardStop
from app.security.guarded_llm import mark_hop
from app.security.types import UNTRUSTED, ContextItem, Decision, HopResult, ItemScores, Source

logger = logging.getLogger(__name__)

_TASK_ITEM_CACHE: Dict[str, List[ContextItem]] = {}

_USER_SOURCES = frozenset({"user_turn", "history"})

# Verify policies for user turns the LLM judge settles (SEC-035a): axis, outcome if flagged, reason.
_JUDGED_POLICIES = {
    "user-harm-verify": ("harm", "decline", "This request looks harmful."),
    "user-jailbreak-verify": ("jailbreak", "consent", "This may try to change how the assistant behaves."),
}


def _judge_verify(item: ContextItem, decision: Decision, state: Optional[WorkflowState]) -> Optional[Decision]:
    """Ask the judge about a user turn the policies sent to verify.

    Returns the judged decision (decline, consent or allow), or None when nothing was judged: no
    judged policy matched, the judge is off, or it failed. Then verify stands (continue, tainted).
    """
    consented = bool(state is not None and state.context.get("consented"))
    judged_any = False
    for policy_id, (axis, outcome, reason) in _JUDGED_POLICIES.items():  # harm first
        if policy_id not in decision.policy_ids:
            continue
        flagged = judge.judge_user_turn(
            item.text, axis,
            moderation_cutoff=context_defender.current_moderation_cutoff() if axis == "harm" else None,
        )
        if flagged is None:
            return None
        judged_any = True
        if flagged and not (outcome == "consent" and consented):
            return Decision(outcome=outcome, policy_ids=(policy_id,), reason=reason,
                            shadow=decision.shadow, latency_ms=decision.latency_ms)
    if not judged_any:
        return None
    return Decision(outcome="allow", policy_ids=decision.policy_ids, reason="Cleared by the judge.",
                    shadow=decision.shadow, latency_ms=decision.latency_ms)


def _guard_cfg():
    return getattr(settings, "SECURITY_GUARD", None)


def _resource_id(state: Optional[WorkflowState], *, chat_id: Optional[str] = None) -> str:
    if chat_id:
        return f'Chat::"{chat_id}"'
    if state is None:
        return 'Task::"task-unknown"'
    task_id = state.context.get("task_id") or "task-unknown"
    return f'Task::"{task_id}"'


def _task_cache_key(state: WorkflowState) -> str:
    prev = state.context.get("previous_messages") or []
    blob = f"{state.query}\0{prev!r}\0{state.context.get('consented')}"
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def task_items(state: Optional[WorkflowState]) -> List[ContextItem]:
    """User turn and history items for a task (cached by content hash)."""
    if state is None:
        return []
    key = _task_cache_key(state)
    cached = _TASK_ITEM_CACHE.get(key)
    if cached is not None:
        return list(cached)
    items: List[ContextItem] = [ContextItem("user_turn", state.query)]
    for turn in state.context.get("previous_messages") or []:
        role = turn.get("role") if isinstance(turn, dict) else getattr(turn, "role", "user")
        content = turn.get("content") if isinstance(turn, dict) else getattr(turn, "content", "")
        if content:
            items.append(ContextItem("history", content, ref=str(role)))
    _TASK_ITEM_CACHE[key] = items
    return list(items)


def items_from_task_request(
    query: str,
    previous_messages: Sequence[Dict[str, str]],
    *,
    has_image: bool = False,
) -> List[ContextItem]:
    items: List[ContextItem] = [ContextItem("user_turn", query)]
    for turn in previous_messages:
        items.append(ContextItem("history", turn["content"], ref=turn.get("role", "user")))
    if has_image:
        items.append(ContextItem("image", "", ref="attachment"))
    return items


def _scores_context(
    item: ContextItem,
    scores: ItemScores,
    state: Optional[WorkflowState],
    *,
    input_mode: str,
    has_image: bool,
) -> Dict[str, Any]:
    return {
        "source": item.source,
        "injection": scores.axes.get("injection", 0),
        "harm": scores.axes.get("harm", 0),
        "sensitive_data": scores.axes.get("sensitive_data", 0),
        "jailbreak": scores.axes.get("jailbreak", 0),
        "threat_risk": scores.threat_risk,
        "top_axis": scores.top_axis,
        "scores_available": scores.scores_available,
        "pattern_hits": sorted(scores.pattern_hits),
        "tainted": ledger.is_tainted(state) or bool(item.tainted),
        "consented": bool(state and state.context.get("consented")),
        "has_image": has_image or item.source == "image",
        "declined_actions_in_task": ledger.declined_actions(state),
        "mode": str(state.context.get("chat_mode") or "unknown") if state else "unknown",
        "env": settings.WEBAGENT_ENV,
        **context_defender.current_thresholds(),
    }


_INJECTION_PATTERNS = frozenset({"injection_phrase", "role_marker", "hidden_text"})


def _flagged_chunks(
    agent: str,
    item: ContextItem,
    scores: ItemScores,
    base_ctx: Dict[str, Any],
    resource: str,
) -> Optional[List[int]]:
    """Re-run the policy per chunk (SEC-036); return the indices whose own decision is decline or
    escalate, or None when chunks can't be matched to the text (then the whole item goes)."""
    tokenizer = context_defender.current_tokenizer()
    texts = context_defender.chunk_text(item.text, tokenizer)
    chunk_scores = list(scores.chunks)
    if not chunk_scores or len(texts) != len(chunk_scores):
        return None
    flagged: List[int] = []
    for idx, (text, axes) in enumerate(zip(texts, chunk_scores)):
        ctx = {**base_ctx, **{a: axes.get(a, 0) for a in context_defender.AXES},
               "pattern_hits": sorted(patterns.scan(text))}
        if decide(agent, "converse", resource, ctx).outcome in ("decline", "escalate"):
            flagged.append(idx)
    return flagged


def _trim_item(agent: str, item: ContextItem, scores: ItemScores, base_ctx: Dict[str, Any], resource: str) -> Optional[ContextItem]:
    """Drop the flagged chunks; drop the whole item when more than half are flagged (SEC-036)."""
    flagged = _flagged_chunks(agent, item, scores, base_ctx, resource)
    n_chunks = max(1, len(scores.chunks))
    if flagged is None or not flagged or len(flagged) > n_chunks / 2:
        # No chunk stands out on its own (the item as a whole was flagged), or most of it is bad.
        return None
    kept = context_defender.text_without_chunks(item.text, flagged, context_defender.current_tokenizer())
    return ContextItem(item.source, kept, ref=item.ref, tainted=item.tainted) if kept else None


def _log_decision(decision: Decision, item: ContextItem, scores: ItemScores, agent: str, state: Optional[WorkflowState]) -> None:
    extra = ""
    if settings.WEBAGENT_ENV == "dev":
        extra = f' text="{item.text[:120]}"'
    task = state.context.get("task_id") if state else "-"
    top = scores.top_axis
    logger.info(
        "guard checkpoint=1 agent=%s action=converse outcome=%s policies=%s source=%s top=%s:%s task=%s shadow=%s error=%s%s",
        agent,
        decision.outcome,
        ",".join(decision.policy_ids) or "-",
        item.source,
        top,
        scores.axes.get(top, 0) if top != "none" else 0,
        task,
        decision.shadow,
        decision.error,
        extra,
    )


def _drop_entry(item: ContextItem, decision: Decision, partial: bool = False) -> Dict[str, Any]:
    entry = {
        "source": item.source,
        "ref": item.ref,
        "policy_id": decision.policy_ids[0] if decision.policy_ids else "",
        "reason": decision.reason,
    }
    if partial:
        entry["partial"] = True
    return entry


def _shadow_flag_entry(item: ContextItem, decision: Decision) -> Dict[str, Any]:
    """A would-be drop in shadow mode: no reason or text, only what the notice email needs."""
    return {
        "source": item.source,
        "ref": item.ref,
        "policy_id": decision.policy_ids[0] if decision.policy_ids else "",
    }


def check_hop(
    agent: str,
    items: List[ContextItem],
    state: Optional[WorkflowState],
    *,
    chat_id: Optional[str] = None,
) -> HopResult:
    """Checkpoint 1 for one hop, timed for SNFR-01 (also when it stops the task)."""
    started = time.perf_counter()
    try:
        return _check_hop(agent, items, state, chat_id=chat_id)
    finally:
        GUARD_HOP_SECONDS.labels(agent=agent).observe(time.perf_counter() - started)


def _item_record(item: ContextItem, scores: ItemScores) -> Dict[str, Any]:
    """What the shadow-week review needs about one judged item; the page URL only for web results
    (public), no text outside dev (SNFR-09)."""
    row: Dict[str, Any] = {
        "source": item.source,
        "axes": dict(scores.axes),
        "top_axis": scores.top_axis,
        "threat_risk": scores.threat_risk,
        "pattern_hits": sorted(scores.pattern_hits),
        "scores_available": scores.scores_available,
        "chunks": len(scores.chunks),
        "tainted_input": bool(item.tainted),
        "model_version": scores.model_version,
    }
    if item.source == "web_result" and item.ref:
        row["ref"] = item.ref
    if settings.WEBAGENT_ENV == "dev":
        row["text"] = item.text[:2000]
    return row


def _check_hop(
    agent: str,
    items: List[ContextItem],
    state: Optional[WorkflowState],
    *,
    chat_id: Optional[str] = None,
) -> HopResult:
    """Checkpoint 1 for one hop (SEC-031). Raises GuardStop for a refused user turn (act mode)."""
    cfg = _guard_cfg()
    if cfg is None or not cfg.enabled:
        mark_hop(agent)
        return HopResult(items=list(items), decisions=[], tainted=ledger.is_tainted(state), dropped=[])

    input_mode = cfg.input_mode
    shadow = input_mode == "shadow"
    resource = _resource_id(state, chat_id=chat_id)
    has_image = any(i.source == "image" for i in items)
    state_ctx: Dict[str, Any] = state.context if state is not None else {"task_id": chat_id}

    scores_list = context_defender.score(items)
    kept: List[ContextItem] = []
    decisions: List[Decision] = []
    dropped: List[Dict[str, Any]] = []
    shadow_flagged: List[Dict[str, Any]] = []
    hop_tainted = ledger.is_tainted(state) or bool(state and state.context.get("consented"))

    for item, scores in zip(items, scores_list):
        ctx = _scores_context(item, scores, state, input_mode=input_mode, has_image=has_image)
        decision = decide(agent, "converse", resource, ctx, shadow=shadow)
        record_guard_decision(1, "converse", decision)
        records.record_collection(checkpoint=1, action="converse", decision=decision, state_context=state_ctx,
                                  item=_item_record(item, scores))
        decisions.append(decision)
        _log_decision(decision, item, scores, agent, state)

        effective = "allow" if shadow else decision.outcome
        if decision.error and not shadow:
            # SEC-061: a policy error fails open for the user's own words, closed for untrusted text.
            effective = "allow" if item.source in _USER_SOURCES else "decline"

        if item.source in _USER_SOURCES:
            if decision.outcome == "verify" and not decision.error:
                judged = _judge_verify(item, decision, state)
                if judged is not None:
                    # Recorded in shadow mode too, so shadow data shows what the judge would do.
                    record_guard_decision(1, "converse", judged)
                    decisions.append(judged)
                    _log_decision(judged, item, scores, agent, state)
                    effective = "allow" if shadow else judged.outcome
                    decision = judged
            if effective in ("consent", "decline", "escalate"):
                if effective == "escalate":
                    records.record_escalation(hop=agent, action="converse", decision=decision,
                                              state_context=state_ctx, scores=dict(scores.axes))
                raise GuardStop(decision)
            if effective == "verify":
                hop_tainted = True
            kept.append(item)
            continue

        if item.source == "image" or item.source not in UNTRUSTED and item.source != "agent_output":
            kept.append(item)
            continue

        if shadow:
            # Recurring-research B0 (REC-135): shadow mode keeps the item, but records what act mode
            # would have dropped, so notice emails can leave those sources' domains out.
            would_be = "decline" if decision.outcome == "consent" else decision.outcome
            if would_be in ("decline", "escalate"):
                shadow_flagged.append(_shadow_flag_entry(item, decision))
        if effective == "consent":
            effective = "decline"  # the policy lint forbids this; treat it as decline if it happens
        if effective == "allow":
            kept.append(item)
        elif effective == "verify":
            hop_tainted = True
            kept.append(item)
        elif effective == "decline":
            hop_tainted = True
            trimmed = None if decision.error else _trim_item(agent, item, scores, ctx, resource)
            if trimmed is None:
                dropped.append(_drop_entry(item, decision))
            else:
                dropped.append(_drop_entry(item, decision, partial=True))
                kept.append(trimmed)
        elif effective == "escalate":
            hop_tainted = True
            dropped.append(_drop_entry(item, decision))
            records.record_escalation(hop=agent, action="converse", decision=decision,
                                      state_context=state_ctx, scores=dict(scores.axes))

    if state is not None:
        ledger.add_hop(state, {
            "agent": agent,
            "tainted": hop_tainted,
            "dropped": dropped,
            "outcomes": [d.outcome for d in decisions],
            "shadow": shadow,
        })
        if dropped:
            ledger.get_ledger(state)["dropped"].extend(dropped)
        if shadow_flagged:
            ledger.get_ledger(state)["shadow_flagged"].extend(shadow_flagged)

    mark_hop(agent, decisions)
    return HopResult(items=kept, decisions=decisions, tainted=hop_tainted, dropped=dropped)


async def acheck_hop(
    agent: str,
    items: List[ContextItem],
    state: Optional[WorkflowState],
    *,
    chat_id: Optional[str] = None,
) -> HopResult:
    """Run check_hop off the event loop, then mark the hop in the caller's context: a context
    variable set inside the worker thread wouldn't be visible to the agent's model call."""
    hop = await asyncio.to_thread(check_hop, agent, items, state, chat_id=chat_id)
    mark_hop(agent, hop.decisions)
    return hop


def kept_text(hop: HopResult, ref: Optional[str], default: str = "") -> str:
    """Text of the kept item with this ref, or a note when checkpoint 1 left it out."""
    for item in hop.items:
        if item.ref == ref and item.source != "user_turn" and item.source != "history":
            return item.text
    if any(d.get("ref") == ref for d in hop.dropped):
        return "[Left out by the security check.]"
    return default
