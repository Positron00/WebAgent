"""Typed research plan shared across supervisor, routers, and research agents."""
from typing import Any, Dict, List

from pydantic import BaseModel, Field


class ResearchPlan(BaseModel):
    requires_web_search: bool = True
    requires_internal_knowledge: bool = False
    requires_document_extraction: bool = False
    questions: List[str] = Field(default_factory=list)
    # The user's query rewritten to make sense without the chat history
    # (e.g. "what about Europe?" -> "EV adoption rates in Europe"). Empty if not provided.
    standalone_query: str = ""


def normalize_research_plan(raw: Dict[str, Any]) -> ResearchPlan:
    """Map legacy supervisor keys to the canonical research plan fields."""
    if not raw:
        return ResearchPlan()

    return ResearchPlan(
        requires_web_search=raw.get(
            "requires_web_search",
            raw.get("needs_web_research", True),
        ),
        requires_internal_knowledge=raw.get(
            "requires_internal_knowledge",
            raw.get("needs_internal_research", False),
        ),
        requires_document_extraction=raw.get(
            "requires_document_extraction",
            raw.get("needs_document_extraction", False),
        ),
        questions=raw.get("questions") or [],
        standalone_query=str(raw.get("standalone_query") or "").strip(),
    )


def research_query(state: Any) -> str:
    """Text to search with: the supervisor's standalone rewrite of a follow-up, else the raw query."""
    plan = normalize_research_plan(state.context.get("research_plan", {}))
    return plan.standalone_query or state.query or state.context.get("query", "")
