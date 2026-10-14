"""A failed web search is reported, not mistaken for "nothing found" (IMP-055), and the
agent's logs give the question's length, not its text, outside dev (IMP-070)."""
from unittest.mock import patch

import pytest

from app.agents import web_research as web_research_module
from app.core.config import settings
from app.models.task import WorkflowState
from app.services.web_search import SearchError

MARKER = "zebra-marker-question about tides"


class _LLM:
    async def ainvoke(self, prompt, *args, **kwargs):
        return "## Answer\nTides follow the moon [1]."


def _agent():
    with patch.object(web_research_module, "get_agent_llm", return_value=_LLM()):
        return web_research_module.WebResearchAgent()


def _state(**context):
    return WorkflowState(query=MARKER, context={"research_plan": {"requires_web_search": True}, **context})


@pytest.mark.asyncio
async def test_search_error_is_recorded_in_the_report_and_logs(monkeypatch, caplog):
    monkeypatch.setattr(settings, "WEBAGENT_ENV", "prod")
    calls = []

    async def failing_search(**kwargs):
        calls.append(kwargs["query"])
        raise SearchError("timed out after 30 s")

    with patch.object(web_research_module, "perform_web_search", failing_search), caplog.at_level("DEBUG"):
        state = await _agent().run(_state())

    report = state.reports["web_research"]
    assert report["search_errors"] == [{"query_chars": len(MARKER), "error": "timed out after 30 s"}]
    assert report["analysis"].startswith("Web search failed")
    assert "Web search failed: timed out after 30 s" in state.logs
    assert len(calls) == 1  # a broken search isn't retried with a simplified query
    own_lines = [r.getMessage() for r in caplog.records if r.name == web_research_module.logger.name]
    assert any("chars" in line for line in own_lines)
    assert not any("zebra" in line for line in own_lines)


@pytest.mark.asyncio
async def test_follow_up_search_error_keeps_the_first_results():
    async def search(**kwargs):
        if kwargs["query"] == "second question":
            raise SearchError("Tavily 500")
        return [{"title": "Tides", "url": "https://example.com/tides", "content": "Moon."}]

    feedback = {"research_questions": ["first question", "second question"]}
    with patch.object(web_research_module, "perform_web_search", search):
        state = await _agent().run(_state(research_iteration=2, research_feedback=feedback))

    report = state.reports["web_research"]
    assert report["sources"][0]["url"] == "https://example.com/tides"
    assert report["search_errors"] == [{"query_chars": len("second question"), "error": "Tavily 500"}]


@pytest.mark.asyncio
async def test_no_search_errors_key_when_searches_work():
    async def search(**kwargs):
        return [{"title": "Tides", "url": "https://example.com/tides", "content": "Moon."}]

    with patch.object(web_research_module, "perform_web_search", search):
        state = await _agent().run(_state())
    assert "search_errors" not in state.reports["web_research"]
