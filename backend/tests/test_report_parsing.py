"""Tests for parsing LLM output in the Team Manager and Senior Research agents."""
import json

from app.agents.senior_research import SeniorResearchAgent
from app.agents.team_manager import TeamManagerAgent

REPORT = {
    "needs_analysis": False,
    "needs_coding": False,
    "summary": "Lists are mutable; tuples are not.",
    "recommendations": ["Use tuples for fixed records"],
    "analysis": "## Details\n\nTuples are hashable.",
    "next_steps": "Read the docs.",
    "sources": [{"url": "https://docs.python.org", "title": "Python docs"}],
}


def test_parse_plain_json():
    assert TeamManagerAgent._parse_report_json(json.dumps(REPORT)) == REPORT


def test_parse_json_in_code_fence():
    text = f"```json\n{json.dumps(REPORT)}\n```"
    assert TeamManagerAgent._parse_report_json(text) == REPORT


def test_parse_json_surrounded_by_prose():
    text = f"Here is the report:\n{json.dumps(REPORT)}\nHope this helps."
    assert TeamManagerAgent._parse_report_json(text) == REPORT


def test_parse_json_followed_by_another_brace_block():
    # Valid JSON, then trailing text containing braces ("Extra data" with a greedy regex)
    text = f"{json.dumps(REPORT)}\n\nNote: see {{the docs}} for details."
    assert TeamManagerAgent._parse_report_json(text) == REPORT


def test_markdown_does_not_duplicate_an_inner_summary():
    report = {
        "summary": "Outer summary.",
        "analysis": "## Executive Summary\n\nInner summary.\n\n## Findings\n\nDetails.",
    }
    markdown = TeamManagerAgent._build_report_markdown("q", report, "")
    assert markdown.count("Executive Summary") == 1
    assert "Outer summary." not in markdown
    assert "## Analysis" not in markdown


def test_unparseable_output_keeps_full_text():
    text = "# Report\n\n" + "A long markdown answer. " * 50
    parsed = TeamManagerAgent._parse_report_json(text)
    assert parsed["analysis"] == text.strip()
    assert parsed["needs_analysis"] is False
    assert "Error" not in parsed["summary"]


def test_markdown_skips_empty_sections():
    report = {"analysis": "Only an analysis.", "summary": "", "recommendations": []}
    markdown = TeamManagerAgent._build_report_markdown("q", report, "")
    assert "## Analysis" in markdown
    assert "## Executive Summary" not in markdown
    assert "## Recommendations" not in markdown


def test_markdown_does_not_duplicate_a_complete_report():
    report = {
        "summary": "Short summary.",
        "analysis": "# Lists vs Tuples\n\n## Executive Summary\n\nFull report.",
        "recommendations": ["Use tuples"],
    }
    markdown = TeamManagerAgent._build_report_markdown("q", report, "")
    assert markdown.startswith("# Lists vs Tuples")
    assert markdown.count("## Executive Summary") == 1
    assert "Comprehensive Analysis" not in markdown
    assert "- Use tuples" in markdown


def test_markdown_uses_fallback_when_reply_has_no_summary_or_analysis():
    report = {"executive_summary": "unexpected key", "needs_analysis": False}
    markdown = TeamManagerAgent._build_report_markdown(
        "q", report, "", fallback_body="## Findings\n\nTuples are immutable."
    )
    assert "Tuples are immutable." in markdown
    assert markdown != "# Comprehensive Analysis: q"


def test_markdown_ignores_fallback_when_analysis_exists():
    report = {"analysis": "Real analysis."}
    markdown = TeamManagerAgent._build_report_markdown("q", report, "", fallback_body="Synthesis.")
    assert "Real analysis." in markdown
    assert "Synthesis." not in markdown


def test_markdown_accepts_string_recommendation():
    report = {"analysis": "x", "recommendations": "Use tuples"}
    markdown = TeamManagerAgent._build_report_markdown("q", report, "")
    assert "- Use tuples" in markdown


def test_evaluation_with_list_research_status():
    text = "```evaluation\nscore: high\nresearch_status: [complete]\n```"
    evaluation = SeniorResearchAgent()._parse_evaluation(text)
    assert evaluation["research_status"] == "complete"
    assert evaluation["requires_additional_research"] is False


def test_evaluation_with_list_status_needing_research():
    text = "```evaluation\nscore: low\nresearch_status: [incomplete]\n```"
    evaluation = SeniorResearchAgent()._parse_evaluation(text)
    assert evaluation["requires_additional_research"] is True
