"""Security ledger stored on ``WorkflowState.context[\"security\"]``."""
from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

from app.models.task import WorkflowState

_LEDGER_KEY = "security"


def _empty_ledger() -> Dict[str, Any]:
    return {
        "tainted": False,
        "declined_actions_in_task": 0,
        "hops": [],
        "dropped": [],
        "skipped": [],
        "approvals": [],
    }


def get_ledger(state: WorkflowState) -> Dict[str, Any]:
    """Return the security ledger dict, creating it on the state if missing."""
    ledger = state.context.get(_LEDGER_KEY)
    if not isinstance(ledger, dict):
        ledger = _empty_ledger()
        state.context[_LEDGER_KEY] = ledger
    for key, default in _empty_ledger().items():
        ledger.setdefault(key, copy.deepcopy(default) if isinstance(default, list) else default)
    return ledger


def is_tainted(state: Optional[WorkflowState]) -> bool:
    if state is None:
        return False
    return bool(get_ledger(state).get("tainted"))


def declined_actions(state: Optional[WorkflowState]) -> int:
    if state is None:
        return 0
    return int(get_ledger(state).get("declined_actions_in_task", 0))


def add_hop(state: WorkflowState, hop: Dict[str, Any]) -> None:
    ledger = get_ledger(state)
    ledger["hops"].append(hop)
    if hop.get("tainted"):
        ledger["tainted"] = True


def record_skipped(state: WorkflowState, entry: Dict[str, Any]) -> None:
    get_ledger(state)["skipped"].append(entry)


def increment_declined_actions(state: WorkflowState, count: int = 1) -> None:
    ledger = get_ledger(state)
    ledger["declined_actions_in_task"] = int(ledger.get("declined_actions_in_task", 0)) + count


def merge(into: WorkflowState, branches: List[WorkflowState]) -> None:
    """Union hops, OR taint, sum declined tool/user declines (SEC-039)."""
    ledger = get_ledger(into)
    for branch in branches:
        other = branch.context.get(_LEDGER_KEY)
        if not isinstance(other, dict):
            continue
        ledger["hops"].extend(other.get("hops", []))
        ledger["dropped"].extend(other.get("dropped", []))
        ledger["skipped"].extend(other.get("skipped", []))
        ledger["approvals"].extend(other.get("approvals", []))
        ledger["tainted"] = bool(ledger.get("tainted")) or bool(other.get("tainted"))
        ledger["declined_actions_in_task"] = int(ledger.get("declined_actions_in_task", 0)) + int(
            other.get("declined_actions_in_task", 0)
        )


def output_item(agent: str, text: str, tainted: bool) -> Dict[str, Any]:
    """Descriptor for wrapping agent output as a context item (checkpoint 1)."""
    return {"source": "agent_output", "text": text, "ref": agent, "tainted": tainted}
