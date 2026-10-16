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


def low_alignment_calls(state: Optional[WorkflowState]) -> int:
    """Search and retrieval calls far from the user's intent so far (checkpoint2-intent SEC-108)."""
    if state is None:
        return 0
    return int(get_ledger(state).get("low_alignment_calls", 0))


def increment_low_alignment_calls(state: WorkflowState) -> None:
    ledger = get_ledger(state)
    ledger["low_alignment_calls"] = int(ledger.get("low_alignment_calls", 0)) + 1


RECENT_CALLS = 3


def note_call(state: WorkflowState, line: str) -> None:
    """Keep the last few checked calls for the judge's context (SEC-111). Not copied into results."""
    calls = get_ledger(state).setdefault("recent_calls", [])
    calls.append(line[:300])
    del calls[:-RECENT_CALLS]


def recent_calls(state: Optional[WorkflowState]) -> List[str]:
    if state is None:
        return []
    return list(get_ledger(state).get("recent_calls") or [])


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
        # Checkpoint2-intent SEC-115: drift counts add up; egress hashes are unioned (same salt per task).
        ledger["low_alignment_calls"] = int(ledger.get("low_alignment_calls", 0)) + int(
            other.get("low_alignment_calls", 0)
        )
        other_egress = other.get("egress")
        if isinstance(other_egress, dict) and other_egress.get("hashes"):
            mine = ledger.setdefault("egress", {"salt": other_egress.get("salt"), "hashes": []})
            if mine.get("salt") == other_egress.get("salt"):
                seen = set(mine["hashes"])
                mine["hashes"].extend(h for h in other_egress["hashes"] if h not in seen)


def output_item(agent: str, text: str, tainted: bool) -> Dict[str, Any]:
    """Descriptor for wrapping agent output as a context item (checkpoint 1)."""
    return {"source": "agent_output", "text": text, "ref": agent, "tainted": tainted}
