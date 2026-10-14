"""One failed research agent doesn't fail the run; both failing does (IMP-043)."""
import pytest

from app.graph import workflows
from app.models.task import WorkflowState
from app.security.errors import GuardStop
from app.security.types import Decision


def _working(name, text):
    async def agent(state):
        state.update_with_agent_output(name, {"findings": text})
        return state
    return agent


def _failing(exc):
    async def agent(state):
        raise exc
    return agent


def _state():
    plan = {"requires_web_search": True, "requires_internal_knowledge": True}
    return WorkflowState(query="q", context={"research_plan": plan})


def _use(monkeypatch, web, internal):
    monkeypatch.setattr(workflows, "get_web_research_agent", lambda: web)
    monkeypatch.setattr(workflows, "get_internal_research_agent", lambda: internal)


@pytest.mark.asyncio
async def test_failed_web_research_keeps_the_internal_report(monkeypatch):
    _use(monkeypatch, _failing(RuntimeError("Tavily down")), _working("internal_research", "from docs"))

    state = await workflows.run_research_agents(_state())

    assert state.reports["internal_research"] == {"findings": "from docs"}
    assert "web_research" not in state.reports
    assert state.context["research_errors"] == [{"agent": "web_research", "error": "Tavily down"}]
    assert any("web_research failed: RuntimeError" in line for line in state.logs)
    assert any("Research phase completed" in line for line in state.logs)


@pytest.mark.asyncio
async def test_both_failing_fails_the_run_with_both_messages(monkeypatch):
    _use(monkeypatch, _failing(RuntimeError("Tavily down")), _failing(ValueError("index missing")))

    with pytest.raises(RuntimeError) as caught:
        await workflows.run_research_agents(_state())

    assert str(caught.value) == "web_research: Tavily down; internal_research: index missing"


@pytest.mark.asyncio
async def test_a_guard_stop_still_ends_the_run(monkeypatch):
    decision = Decision(outcome="escalate", policy_ids=("p",), reason="stopped by policy",
                        shadow=False, latency_ms=0.0)
    _use(monkeypatch, _failing(GuardStop(decision)), _working("internal_research", "from docs"))

    with pytest.raises(GuardStop):
        await workflows.run_research_agents(_state())
