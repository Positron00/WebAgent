"""Shared types for the security guard."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List, Literal, Optional, Tuple

Source = Literal[
    "user_turn",
    "history",
    "agent_output",
    "web_result",
    "retrieved_doc",
    "document",
    "tool_output",
    "image",
    "unlabelled",
]
UNTRUSTED = frozenset(
    {"web_result", "retrieved_doc", "document", "tool_output", "unlabelled"}
)
Outcome = Literal["allow", "verify", "consent", "decline", "escalate"]
SEVERITY: Tuple[Outcome, ...] = ("allow", "verify", "consent", "decline", "escalate")
ToolAction = Literal[
    "tool::web_search",
    "tool::internal_retrieve",
    "tool::document_extract",
    "tool::execute_code",
]
# The plan check: the supervisor's rewrite and agent-written subgoals (checkpoint2-intent spec).
PlanAction = Literal["plan::rewrite", "plan::subgoal"]


@dataclass(frozen=True)
class ContextItem:
    source: Source
    text: str
    ref: Optional[str] = None
    tainted: bool = False


@dataclass(frozen=True)
class ItemScores:
    axes: Dict[str, int]
    top_axis: str
    threat_risk: int
    pattern_hits: FrozenSet[str]
    chunks: Tuple[Dict[str, int], ...]
    scores_available: bool
    model_version: str


@dataclass(frozen=True)
class Decision:
    outcome: Outcome
    policy_ids: Tuple[str, ...]
    reason: str
    shadow: bool
    latency_ms: float
    # True when policy evaluation itself failed; callers then fail open or closed per SEC-061.
    error: bool = False
    # Shadow-only policies that matched (@shadow("true")): recorded, never applied (SEC-012a).
    shadow_policy_ids: Tuple[str, ...] = ()


@dataclass
class HopResult:
    items: List[ContextItem]
    decisions: List[Decision]
    tainted: bool
    dropped: List[Dict[str, Any]]


@dataclass(frozen=True)
class ToolRequest:
    action: ToolAction
    args: Dict[str, Any]
