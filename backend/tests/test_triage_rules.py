"""Rule table (REQ-013): rules run before the model, in order; near-misses go to the model."""
import pytest

from app.services.triage import match_rule

GREETING = ("direct", "Greeting or short acknowledgement")
RESEARCH = ("agentic", "Explicit research request")


@pytest.mark.parametrize(
    "query,has_image,expected",
    [
        ("compare CRM tools", True, ("direct", "Has an image")),
        ("research EU AI rules", True, ("direct", "Has an image")),  # image rule comes first
        ("", False, GREETING),
        ("   ", False, GREETING),
        ("thanks", False, GREETING),
        ("Thanks!", False, GREETING),
        ("thank you.", False, GREETING),
        ("Good morning", False, GREETING),
        ("research EU AI regulation changes", False, RESEARCH),
        ("Please research battery recycling", False, RESEARCH),
        ("find studies on intermittent fasting", False, RESEARCH),
        ("Write a report on solar subsidies", False, RESEARCH),
        ("do a deep dive into RISC-V adoption", False, RESEARCH),
        ("give me a literature review of CRISPR safety", False, RESEARCH),
        # Known false match, kept on purpose: "research" used as a noun still routes agentic.
        ("research shows coffee is healthy, true?", False, RESEARCH),
    ],
)
def test_rules(query, has_image, expected):
    assert match_rule(query, has_image) == expected


@pytest.mark.parametrize(
    "query",
    [
        "hi, compare the top CRM tools for small teams",
        "thanks to whom was penicillin discovered?",
        "investigate why this test fails",
        "can you research this for me",  # "research" not at the start
        "researching is fun",  # word boundary: not the word "research"
    ],
)
def test_near_misses_go_to_the_model(query):
    assert match_rule(query, False) is None
