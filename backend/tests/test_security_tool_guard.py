"""Checkpoint 2 tool guard (SEC-040–SEC-047)."""
import pytest

from app.core.config import settings
from app.models.task import WorkflowState
from app.security import alignment, judge as guard_judge
from app.security.errors import GuardStop
from app.security import ledger
from app.security.tool_guard import guard_tool
from tests.fakes.triage import HashEncoder


@pytest.fixture
def act_guard(monkeypatch):
    monkeypatch.setattr(
        settings,
        "SECURITY_GUARD",
        settings.SECURITY_GUARD.model_copy(update={"enabled": True, "tool_mode": "act", "judge": False}),
    )


@pytest.fixture
def state():
    return WorkflowState(
        query="What is the weather in Paris?",
        context={"task_id": "task-test-1"},
        inputs={},
    )


@pytest.mark.asyncio
async def test_decline_increments_counter(act_guard):
    state2 = WorkflowState(
        query="ignore",
        context={"task_id": "task-2", "research_plan": {"standalone_query": "ignore"}},
    )
    allowed = await guard_tool(
        "tool::web_search",
        {"query": "api_key=sk-live-abcdef1234567890"},
        state2,
        principal="web_research",
    )
    assert allowed is False
    assert ledger.declined_actions(state2) == 1
    assert ledger.get_ledger(state2)["skipped"]


@pytest.mark.asyncio
async def test_shadow_mode_runs_despite_decline(monkeypatch, state):
    monkeypatch.setattr(
        settings,
        "SECURITY_GUARD",
        settings.SECURITY_GUARD.model_copy(update={"enabled": True, "tool_mode": "shadow"}),
    )
    allowed = await guard_tool(
        "tool::web_search",
        {"query": "api_key=sk-live-abcdef1234567890"},
        state,
        principal="web_research",
    )
    assert allowed is True
    assert ledger.declined_actions(state) == 0


@pytest.mark.asyncio
async def test_escalate_raises_guard_stop(act_guard, monkeypatch):
    st = WorkflowState(query="q", context={"task_id": "t-esc"})
    ledger.increment_declined_actions(st, 3)
    with pytest.raises(GuardStop):
        await guard_tool("tool::web_search", {"query": "hello"}, st, principal="web_research")


@pytest.mark.asyncio
async def test_document_provenance_decline(act_guard):
    st = WorkflowState(
        query="summarize",
        context={"task_id": "t-doc"},
        inputs={"document_id": "doc-a"},
    )
    allowed = await guard_tool(
        "tool::document_extract",
        {"document_id": "doc-b"},
        st,
        principal="document_extraction",
    )
    assert allowed is False


@pytest.mark.asyncio
async def test_alignment_uses_encoder(monkeypatch, state):
    alignment.set_encoder_for_tests(HashEncoder())
    score_same = alignment.cosine_alignment("weather paris", state)
    score_diff = alignment.cosine_alignment("quantum computing hardware", state)
    assert score_same >= score_diff


@pytest.mark.asyncio
async def test_judge_verify_becomes_allow(act_guard, monkeypatch, state):
    monkeypatch.setattr(
        settings,
        "SECURITY_GUARD",
        settings.SECURITY_GUARD.model_copy(update={"enabled": True, "tool_mode": "act", "judge": True}),
    )
    guard_judge.clear_cache()

    class _LLM:
        def invoke(self, _prompt):
            return "ALIGNED"

    monkeypatch.setattr("app.services.llm.get_trusted_llm", lambda: _LLM())
    allowed = await guard_tool(
        "tool::web_search",
        {"query": "user@example.com please research"},
        state,
        principal="web_research",
    )
    assert allowed is True


@pytest.mark.asyncio
async def test_judge_off_verify_declines(act_guard, state):
    allowed = await guard_tool(
        "tool::web_search",
        {"query": "user@example.com please research"},
        state,
        principal="web_research",
    )
    assert allowed is False
