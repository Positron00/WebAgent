"""Checkpoint 2: guard tool actions before they run (SEC-040–SEC-047)."""
from __future__ import annotations

import asyncio
import time
import logging
from typing import Any, Dict, List, Optional, Set

from app.core.config import settings
from app.core.metrics import GUARD_ALIGNMENT, GUARD_TOOL_SECONDS, record_guard_decision
from app.models.task import WorkflowState
from app.security import alignment, code_checks, patterns
from app.security.decision import decide
from app.security.errors import GuardStop
from app.security import judge as guard_judge
from app.security import ledger
from app.security import records
from app.security import tokens
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


def _build_context(
    action: ToolAction,
    args: Dict[str, Any],
    state: WorkflowState,
    *,
    tool_mode: str,
) -> Dict[str, Any]:
    pattern_hits: Set[str] = set()
    code_findings: Set[str] = set()
    align = 100

    user_turn = state.query
    if action in ("tool::web_search", "tool::internal_retrieve"):
        query = str(args.get("query") or "")
        pattern_hits = patterns.scan_args(query, user_turn=user_turn)
        align = alignment.alignment_for_action(action, args, state)
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
    }


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
) -> bool:
    """
    Run checkpoint 2 for a tool call.

    Returns True when the tool should run. Raises :class:`GuardStop` on escalate. Timed for SNFR-01.
    """
    started = time.perf_counter()
    try:
        return await _guard_tool(action, args, state, principal=principal)
    finally:
        GUARD_TOOL_SECONDS.labels(action=action).observe(time.perf_counter() - started)


async def _guard_tool(
    action: ToolAction,
    args: Dict[str, Any],
    state: WorkflowState,
    *,
    principal: str,
) -> bool:
    cfg = _guard_cfg()
    if cfg is None or not cfg.enabled:
        return True

    tool_mode = cfg.tool_mode
    shadow = tool_mode == "shadow"
    # Alignment encodes text and the judge may wait 3 s: keep both off the event loop.
    context = await asyncio.to_thread(_build_context, action, args, state, tool_mode=tool_mode)
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
    records.record_collection(
        checkpoint=2,
        action=action,
        decision=decision,
        state_context=state.context,
        args=args,
        context={k: context.get(k) for k in ("alignment", "tainted", "pattern_hits", "code_findings",
                                               "approved_tools", "declined_actions_in_task")},
    )

    effective = decision.outcome
    if decision.error:
        # SEC-061: fail closed for code and documents, open for search and retrieval.
        effective = "decline" if action in ("tool::execute_code", "tool::document_extract") else "allow"
    if shadow:
        effective = "allow"

    if effective == "verify":
        if await asyncio.to_thread(guard_judge.is_aligned, state, action, args):
            effective = "allow"
        else:
            effective = "decline"

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
