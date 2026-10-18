"""Per-task search cap and usage (REC-109) and the standalone question (REC-110)."""
import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.agents import web_research
from app.core.config import settings
from app.models.task import WorkflowState
from app.services import result_store, task_budget, together_llm
from app.services import task_manager as task_manager_module
from app.services.together_llm import TogetherWithUsage
from main import app
from tests.fakes.workers import separate_task_manager
from tests.fakes.workflow import StreamingFakeWorkflow

client = TestClient(app)

NO_USAGE = {"input_tokens": 0, "output_tokens": 0, "searches": 0, "tavily_credits": 0}


# --- the counter ---

def test_add_counts_input_and_output_tokens_next_to_the_total():
    token = task_budget.start(1000)
    try:
        task_budget.add(30, 20, 10)
        task_budget.add(5)  # callers that only know the total still work
        assert task_budget.used() == 35
        assert task_budget.usage() == {"input_tokens": 20, "output_tokens": 10, "searches": 0, "tavily_credits": 0}
    finally:
        task_budget.reset(token)
    assert task_budget.usage() == NO_USAGE


@pytest.mark.parametrize("depths, credits", [(["basic"], 1), (["advanced"], 2), (["basic", "advanced", "basic"], 4)])
def test_credits_by_search_depth(depths, credits):
    token = task_budget.start(0)
    try:
        assert all(task_budget.charge_search(d) for d in depths)
        assert task_budget.usage()["searches"] == len(depths)
        assert task_budget.usage()["tavily_credits"] == credits
    finally:
        task_budget.reset(token)


def test_charge_search_stops_at_the_cap_and_counts_nothing_more():
    token = task_budget.start(0, search_cap=2)
    try:
        assert task_budget.charge_search("advanced") and task_budget.charge_search("basic")
        assert task_budget.charge_search("basic") is False
        assert task_budget.usage()["searches"] == 2 and task_budget.usage()["tavily_credits"] == 3
    finally:
        task_budget.reset(token)


def test_without_a_counter_searches_are_not_capped():
    assert all(task_budget.charge_search("advanced") for _ in range(50))
    assert task_budget.usage() == NO_USAGE


class _Reply:
    status_code = 200
    text = ""

    def json(self):
        return {"choices": [{"text": "ok"}], "usage": {"prompt_tokens": 40, "completion_tokens": 20, "total_tokens": 60}}


def test_together_passes_prompt_and_completion_tokens(monkeypatch):
    monkeypatch.setattr(together_llm.requests, "post", lambda *a, **k: _Reply())
    llm = TogetherWithUsage(model="meta-models/Muse-Glimmer-30B", max_tokens=50, together_api_key="test-key")
    token = task_budget.start(1000)
    try:
        llm.invoke("one")
        llm.invoke("two")
        assert task_budget.used() == 120
        assert task_budget.usage()["input_tokens"] == 80
        assert task_budget.usage()["output_tokens"] == 40
    finally:
        task_budget.reset(token)


# --- the cap in web research: every pass and the parallel agents share one total ---

@pytest.fixture
def searches(monkeypatch):
    """A web research agent with a fake model; returns the list of searches made."""
    calls = []

    async def fake_search(**kwargs):
        calls.append(kwargs["search_depth"])
        return [{"title": "t", "url": f"https://example.com/{len(calls)}", "content": "c"}]

    async def allow(*args, **kwargs):
        return True

    monkeypatch.setattr(web_research, "get_agent_llm", lambda name: None)
    monkeypatch.setattr(web_research, "perform_web_search", fake_search)
    monkeypatch.setattr(web_research, "guard_tool", allow)
    return calls


def _agent():
    async def reply(*args, **kwargs):
        return "## Answer"

    agent = web_research.WebResearchAgent()
    agent.llm = SimpleNamespace(ainvoke=reply)
    return agent


def _follow_up_state() -> WorkflowState:
    """A follow-up pass: one search for the first question and one for the next."""
    return WorkflowState(query="q", context={
        "research_plan": {"requires_web_search": True, "search_depth": "deep"},
        "research_iteration": 2,
        "research_feedback": {"research_questions": ["first question?", "second question?"]},
    })


@pytest.mark.asyncio
async def test_cap_counts_across_three_passes(searches, caplog):
    token = task_budget.start(0, search_cap=5)
    try:
        with caplog.at_level(logging.INFO, logger=web_research.__name__):
            for _ in range(3):  # 2 searches a pass, so the cap binds in the third
                await _agent().run(_follow_up_state())
        usage = task_budget.usage()
    finally:
        task_budget.reset(token)
    assert len(searches) == 5
    assert usage["searches"] == 5
    # The first search of each pass is advanced ("deep"), the follow-up one basic.
    assert usage["tavily_credits"] == 2 + 1 + 2 + 1 + 2
    assert "Search limit reached (5 per task)" in caplog.text


@pytest.mark.asyncio
async def test_cap_counts_across_parallel_agents_on_deep_copies(searches):
    """As run_research_agents does: asyncio.gather over agents given deep copies of the state."""
    state = _follow_up_state()
    token = task_budget.start(0, search_cap=3)
    try:
        await asyncio.gather(*(_agent().run(state.model_copy(deep=True)) for _ in range(3)))
        usage = task_budget.usage()
    finally:
        task_budget.reset(token)
    assert len(searches) == 3
    assert usage["searches"] == 3
    assert "searches" not in state.context  # the count never lived in the state


@pytest.mark.asyncio
async def test_no_counter_means_no_cap_in_web_research(searches):
    for _ in range(4):
        await _agent().run(_follow_up_state())
    assert len(searches) == 8


# --- usage and standalone_query on results ---

class _Workflow(StreamingFakeWorkflow):
    """Spends like a run: model tokens and two searches; the supervisor's plan has a rewrite."""

    def __init__(self, plan=None):
        self.plan = plan if plan is not None else {"standalone_query": "What changed in EU AI rules in 2026?"}
        self.cap = None
        self.limit = None

    async def ainvoke(self, state):
        self.cap = task_budget.search_cap()
        self.limit = task_budget.current().limit
        task_budget.add(300, 200, 100)
        task_budget.charge_search("basic")
        task_budget.charge_search("advanced")
        context = {**state.context, "research_plan": self.plan}
        return {"query": state.query, "context": context, "final_report": {"title": "R", "content": "ok"}}


def _post_and_poll(body, workflow):
    with patch("app.services.task_manager.get_agent_workflow", return_value=workflow), patch(
        "app.services.task_manager.SupervisorAgent"
    ) as sup:
        async def answer(*args, **kwargs):
            task_budget.add(30, 20, 10)
            return "direct answer"

        sup.return_value.answer_directly = AsyncMock(side_effect=answer)
        task_id = client.post("/api/task", json=body).json()["task_id"]
        for _ in range(30):
            polled = client.get(f"/api/task/{task_id}").json()
            if polled["status"] != "processing":
                break
    return task_id, polled


def test_workflow_result_has_usage_and_standalone_query_in_the_response():
    workflow = _Workflow()
    task_id, polled = _post_and_poll({"query": "and in 2026?", "mode": "always"}, workflow)
    assert polled["status"] == "completed"
    usage = {"input_tokens": 200, "output_tokens": 100, "searches": 2, "tavily_credits": 3}
    assert polled["result"]["usage"] == usage
    assert polled["result"]["standalone_query"] == "What changed in EU AI rules in 2026?"
    assert "origin" not in polled["result"] and "topic_id" not in polled["result"]
    # Interactive runs are capped per task by the config value, with the global token budget.
    assert workflow.cap == task_manager_module.get_agents_config().web_research.max_searches_per_task
    assert workflow.limit == settings.TASK_TOKEN_BUDGET

    saved = result_store.load_result(task_id)
    assert saved["usage"] == usage and saved["result"]["usage"] == usage
    assert saved["tokens_used"] == 300
    # Read back from the file, as another worker would.
    status, result = separate_task_manager().get_task_status(task_id)
    assert result["standalone_query"] == "What changed in EU AI rules in 2026?"


def test_no_standalone_query_when_the_plan_has_none():
    _, polled = _post_and_poll({"query": "q", "mode": "always"}, _Workflow(plan={}))
    assert polled["status"] == "completed"
    assert "standalone_query" not in polled["result"]


def test_direct_result_has_usage_with_no_searches():
    _, polled = _post_and_poll({"query": "hello", "mode": "off"}, _Workflow())
    assert polled["status"] == "completed"
    assert polled["result"]["usage"] == {"input_tokens": 20, "output_tokens": 10, "searches": 0, "tavily_credits": 0}


@pytest.mark.asyncio
async def test_error_result_has_usage_too(monkeypatch):
    class Failing(StreamingFakeWorkflow):
        async def ainvoke(self, state):
            task_budget.add(10, 6, 4)
            task_budget.charge_search("basic")
            raise RuntimeError("boom")

    monkeypatch.setattr(task_manager_module, "get_agent_workflow", lambda: Failing())
    manager = separate_task_manager()
    task_id = str(uuid4())
    manager.register_pending(task_id, "q")
    await manager.run_workflow(task_id, "q", {})
    saved = result_store.load_result(task_id)
    assert saved["status"] == "error"
    assert saved["result"]["usage"] == {"input_tokens": 6, "output_tokens": 4, "searches": 1, "tavily_credits": 1}


def test_standalone_query_is_not_logged_outside_dev(monkeypatch, caplog):
    monkeypatch.setattr(settings, "WEBAGENT_ENV", "prod")
    secret = "What changed in EU AI rules in 2026?"
    with caplog.at_level(logging.DEBUG):
        _, polled = _post_and_poll({"query": "and then?", "mode": "always"}, _Workflow({"standalone_query": secret}))
    assert polled["result"]["standalone_query"] == secret
    assert secret not in caplog.text
