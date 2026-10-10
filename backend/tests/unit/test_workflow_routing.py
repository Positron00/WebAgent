"""Routers and the research node read the same keys the agents write."""
from unittest.mock import patch

import pytest
from langgraph.graph import END

from app.graph.workflows import (
    run_research_agents,
    senior_research_router,
    supervisor_router,
    team_manager_router,
)
from app.models.task import WorkflowState


def _state(**kwargs) -> WorkflowState:
    return WorkflowState(query="q", **kwargs)


def test_supervisor_router_accepts_legacy_plan_keys():
    state = _state(context={"agentic_toggle": True, "research_plan": {"needs_web_research": True}})
    assert supervisor_router(state) == "research"


def test_supervisor_router_ends_when_agentic_off():
    assert supervisor_router(_state(context={"agentic_toggle": False})) == END


def test_senior_router_follows_agent_flags():
    assert senior_research_router(_state(context={"continue_research": True})) == "research"
    assert senior_research_router(_state(context={"research_complete": True})) == "team_manager"


def test_team_manager_router_does_not_repeat_finished_steps():
    reports = {"team_manager": {"needs_analysis": True, "needs_coding": True}}
    assert team_manager_router(_state(reports=reports)) == "data_analysis"

    reports["data_analysis"] = {"analysis": "done"}
    assert team_manager_router(_state(reports=reports)) == "coding_assistant"

    reports["coding_review"] = {"fixed_code": ""}
    assert team_manager_router(_state(reports=reports)) == END


@pytest.mark.asyncio
async def test_research_node_counts_iterations_and_replaces_findings():
    calls = []

    async def web_agent(state):
        calls.append(state.context["research_iteration"])
        state.update_with_agent_output("web_research", {"analysis": f"pass {len(calls)}"})
        return state

    with patch("app.graph.workflows.get_web_research_agent", return_value=web_agent):
        state = _state(context={"research_plan": {"requires_web_search": True}})
        state = await run_research_agents(state)
        state.reports["senior_research"] = {"synthesis": "first"}
        state = await run_research_agents(state)

    assert calls == [1, 2]
    assert state.reports["web_research"]["analysis"] == "pass 2"
