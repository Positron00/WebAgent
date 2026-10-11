"""Resolve API client mode (off / auto / always) for phase 5."""
from typing import Literal, Optional

from app.services.triage import TriageDecision, classify

Mode = Literal["off", "auto", "always"]


def resolve_mode(mode: Optional[str], legacy_flag: Optional[bool]) -> Mode:
    if mode is not None:
        if mode in ("off", "auto", "always"):
            return mode
        return "off"
    if legacy_flag is not None:
        return "always" if legacy_flag else "off"
    return "off"


def choose_route(
    mode: Mode,
    query: str,
    previous_user_question: Optional[str],
    has_image: bool,
) -> TriageDecision:
    from app.core.metrics import record_triage_decision

    if mode == "off":
        decision = TriageDecision("direct", None, "mode", "Off mode", "none", 0.0)
        record_triage_decision(decision)
        return decision
    if mode == "always":
        decision = TriageDecision("agentic", None, "mode", "Always mode", "none", 0.0)
        record_triage_decision(decision)
        return decision
    return classify(query, previous_user_question, has_image)
