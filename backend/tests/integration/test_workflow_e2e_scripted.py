"""End-to-end workflow test with mocked agents (no API keys)."""
from unittest.mock import patch

import pytest

from app.agents.base_agent import BaseAgent
from app.graph import workflows as workflows_module
from app.graph.workflows import build_agent_workflow
from app.models.task import WorkflowState


@pytest.mark.asyncio
async def test_agentic_workflow_produces_final_report():
    plan = {
        "requires_web_search": True,
        "requires_internal_knowledge": False,
        "questions": ["What is the answer?"],
    }

    async def supervisor(state: WorkflowState) -> WorkflowState:
        state.context["agentic_toggle"] = True
        state.context["research_plan"] = plan
        state.update_with_agent_output("supervisor", {"research_plan": plan})
        return state

    async def web_research(state: WorkflowState) -> WorkflowState:
        report = {
            "analysis": "Web findings about the example topic.",
            "sources": [{"title": "Example", "url": "https://example.com"}],
        }
        state.update_with_agent_output("web_research", report)
        return state

    async def senior_research(state: WorkflowState) -> WorkflowState:
        state.update_with_agent_output(
            "senior_research",
            {
                "synthesis": "## Summary\nSynthetic answer for the user.",
                "research_status": "complete",
            },
        )
        state.context["research_complete"] = True
        state.context["continue_research"] = False
        return state

    async def team_manager(state: WorkflowState) -> WorkflowState:
        state.update_with_agent_output(
            "team_manager",
            {"needs_analysis": False, "needs_coding": False},
        )
        state.mark_completed(
            {
                "title": "Report",
                "content": "Final markdown report for the user.",
                "type": "comprehensive_report",
            }
        )
        return state

    async def noop(state: WorkflowState) -> WorkflowState:
        return state

    workflows_module._workflow_instance = None
    with patch.object(BaseAgent, "trace_chain", lambda self, chain: chain), patch(
        "app.graph.workflows.get_supervisor_agent", return_value=supervisor
    ), patch("app.graph.workflows.get_web_research_agent", return_value=web_research), patch(
        "app.graph.workflows.get_internal_research_agent",
        return_value=noop,
    ), patch(
        "app.graph.workflows.get_senior_research_agent", return_value=senior_research
    ), patch(
        "app.graph.workflows.get_team_manager_agent", return_value=team_manager
    ), patch(
        "app.graph.workflows.get_document_extraction_agent", return_value=noop
    ), patch("app.graph.workflows.get_data_analysis_agent", return_value=noop), patch(
        "app.graph.workflows.get_coding_assistant_agent", return_value=noop
    ), patch("app.graph.workflows.get_coding_review_agent", return_value=noop
    ):
        # Keep the patches active while the graph runs: the research node looks up
        # the research agents at run time, not only when the graph is built.
        workflow = build_agent_workflow()
        state = WorkflowState(
            query="What is an example?",
            context={"agentic_toggle": True},
        )
        result = await workflow.ainvoke(state)
    final = (
        result
        if isinstance(result, WorkflowState)
        else WorkflowState.model_validate(result)
    )

    assert final.final_report is not None
    assert final.final_report.get("content")
    # The stubbed research agent ran, not the real one (which would call Tavily and the LLM).
    assert final.reports["web_research"]["analysis"] == "Web findings about the example topic."
