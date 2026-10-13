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


def without_missing_document(plan: ResearchPlan, has_document: bool) -> ResearchPlan:
    """Drop a document-extraction step the model asked for when no document was attached.

    The supervisor model sometimes sets `requires_document_extraction` on plain questions; the
    extraction step then fails with "Document ID is required". Research the question instead.
    It also sometimes plans a plain question as document extraction with every step false; the
    workflow then ends without a report, so a plan with nothing to do is researched on the web too.
    """
    if not (plan.requires_web_search or plan.requires_internal_knowledge or plan.requires_document_extraction):
        return plan.model_copy(update={"requires_web_search": True})
    if not plan.requires_document_extraction or has_document:
        return plan
    return plan.model_copy(
        update={
            "requires_document_extraction": False,
            "requires_web_search": plan.requires_web_search or not plan.requires_internal_knowledge,
        }
    )


def research_query(state: Any) -> str:
    """Text to search with: the supervisor's standalone rewrite of a follow-up, else the raw query."""
    plan = normalize_research_plan(state.context.get("research_plan", {}))
    return plan.standalone_query or state.query or state.context.get("query", "")
