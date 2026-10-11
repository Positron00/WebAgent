"""Run the real LangGraph workflow with stand-in agents to check state flows between nodes."""
from unittest.mock import patch

from app.graph import workflows
from app.models.task import WorkflowState

seen_by_senior = []


async def supervisor(state):
    state.context["research_plan"] = {
        "requires_web_search": True,
        "requires_internal_knowledge": False,
    }
    state.update_with_agent_output("supervisor", {"requirements": []})
    return state


async def web_research(state):
    state.update_with_agent_output("web_research", {"findings": "web findings"})
    return state


async def senior_research(state):
    iteration = state.context.get("research_iteration", 1)
    seen_by_senior.append((iteration, sorted(state.reports)))
    state.update_with_agent_output("senior_research", {"synthesis": f"pass {iteration}"})
    if iteration < 3:
        state.context["continue_research"] = True
    else:
        state.context["continue_research"] = False
        state.context["research_complete"] = True
    return state


async def team_manager(state):
    synthesis = state.reports["senior_research"]["synthesis"]
    state.mark_completed({"content": synthesis})
    return state


async def noop(state):
    return state


AGENTS = {
    "get_supervisor_agent": supervisor,
    "get_web_research_agent": web_research,
    "get_internal_research_agent": noop,
    "get_senior_research_agent": senior_research,
    "get_team_manager_agent": team_manager,
    "get_document_extraction_agent": noop,
    "get_data_analysis_agent": noop,
    "get_coding_assistant_agent": noop,
    "get_coding_review_agent": noop,
}


async def test_reports_and_iterations_flow_between_nodes():
    seen_by_senior.clear()
    factories = {name: (lambda agent=agent: agent) for name, agent in AGENTS.items()}
    with patch.multiple(workflows, **factories):
        graph = workflows._build_agent_workflow_impl()
        result = await graph.ainvoke(
            WorkflowState(query="q", context={"agentic_toggle": True}),
            {"recursion_limit": 25},
        )

    final = result if isinstance(result, WorkflowState) else WorkflowState.model_validate(result)

    # Each research pass counts up, and earlier agents' reports are visible downstream.
    assert [iteration for iteration, _ in seen_by_senior] == [1, 2, 3]
    assert seen_by_senior[0][1] == ["supervisor", "web_research"]
    assert "senior_research" in seen_by_senior[1][1]

    assert final.final_report["content"] == "pass 3"
    assert "Starting research phase (iteration 3)" in final.final_report["logs"]
