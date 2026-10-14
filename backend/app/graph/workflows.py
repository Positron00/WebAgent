"""
LangGraph workflow definitions for WebAgent backend.

This module contains the workflow definitions for the WebAgent backend,
including the agent workflow graph and routing logic.
"""
import logging
import json
from typing import Dict, List, Optional, Any, cast
import os
import time
import asyncio

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from langchain.callbacks.tracers import LangChainTracer

from app.models.task import WorkflowState
from app.models.research_plan import normalize_research_plan
from app.agents.supervisor import get_supervisor_agent
from app.agents.web_research import get_web_research_agent
from app.agents.internal_research import get_internal_research_agent
from app.agents.senior_research import get_senior_research_agent
from app.agents.data_analysis import get_data_analysis_agent
from app.agents.coding_assistant import get_coding_assistant_agent
from app.agents.coding_review import get_coding_review_agent
from app.agents.team_manager import get_team_manager_agent
from app.agents.document_extraction_agent import get_document_extraction_agent
from app.core.config import settings
from app.core.metrics import timing_decorator

# Configure logging
logger = logging.getLogger(__name__)

# Singleton workflow instance
_workflow_instance = None

# Routers must not modify state: LangGraph hands them a throwaway copy and keeps
# only the returned branch name. Write logs and state changes inside nodes.

def supervisor_router(state: WorkflowState) -> str:
    """
    Determine the next step after the supervisor agent.
    
    Args:
        state: The current workflow state
        
    Returns:
        String indicating the next node
    """
    # Check if agentic toggle is off
    if not state.context.get("agentic_toggle", False):
        logger.info("Agentic toggle is off, ending workflow")
        return END

    if state.error:
        logger.warning(f"Error detected in workflow: {state.error}")
        return END

    plan = normalize_research_plan(state.context.get("research_plan", {}))

    if plan.requires_document_extraction:
        logger.info("Routing to document extraction")
        return "document_extraction"

    if plan.requires_web_search or plan.requires_internal_knowledge:
        logger.info("Routing to research agents")
        return "research"

    logger.info("No further processing needed, ending workflow")
    return END

def senior_research_router(state: WorkflowState) -> str:
    """
    Determine the next step after the senior research agent.
    
    Args:
        state: The current workflow state
        
    Returns:
        String indicating the next node
    """
    # Check if there's an error
    if state.error:
        logger.warning(f"Error detected in workflow: {state.error}")
        return END

    if state.context.get("continue_research"):
        logger.info("Senior Research requesting more research")
        return "research"

    if state.context.get("research_complete"):
        logger.info("Senior Research passing to Team Manager")
        return "team_manager"

    logger.info("Senior Research finalized, ending workflow")
    return END

def team_manager_router(state: WorkflowState) -> str:
    """
    Determine the next step after the team manager agent.
    
    Args:
        state: The current workflow state
        
    Returns:
        String indicating the next node
    """
    # Check if there's an error
    if state.error:
        logger.warning(f"Error detected in workflow: {state.error}")
        return END

    team_manager_output = state.reports.get("team_manager", {})
    needs_analysis = team_manager_output.get("needs_analysis", False) or state.context.get(
        "needs_analysis", False
    )
    needs_coding = team_manager_output.get("needs_coding", False)

    logger.info(
        f"Team Manager assessment - needs analysis: {needs_analysis}, needs coding: {needs_coding}"
    )

    if needs_analysis and "data_analysis" not in state.reports:
        logger.info("Team Manager requesting data analysis")
        return "data_analysis"
    if needs_coding and "coding_review" not in state.reports:
        logger.info("Team Manager requesting coding assistance")
        return "coding_assistant"

    logger.info("Team Manager finished processing, ending workflow")
    return END

async def run_research_agents(
    state: WorkflowState, config: Optional[RunnableConfig] = None
) -> WorkflowState:
    """
    Run multiple research agents in parallel.
    
    Args:
        state: The current workflow state
        config: Optional configuration for the runnable
        
    Returns:
        The updated workflow state
    """
    plan = normalize_research_plan(state.context.get("research_plan", {}))
    requires_web = plan.requires_web_search
    requires_internal = plan.requires_internal_knowledge

    has_previous_pass = "senior_research" in state.reports
    iteration = state.context.get("research_iteration", 0) + 1 if has_previous_pass else 1
    state.context["research_iteration"] = iteration
    state.context["continue_research"] = False

    state.add_log(f"Starting research phase (iteration {iteration})")
    logger.info(f"Running research agents - web: {requires_web}, internal: {requires_internal}")

    agents = []
    names = []
    if requires_web:
        logger.info("Starting web research agent")
        agents.append(get_web_research_agent())
        names.append("web_research")
    if requires_internal:
        logger.info("Starting internal research agent")
        agents.append(get_internal_research_agent())
        names.append("internal_research")

    if not agents:
        state.add_log("No research agents triggered")
        logger.warning("No research agents were triggered, check the research plan")
        return state

    results = await asyncio.gather(
        *(agent(state.model_copy(deep=True)) for agent in agents), return_exceptions=True
    )
    failures = [(name, r) for name, r in zip(names, results) if isinstance(r, BaseException)]
    for name, exc in failures:
        # Guard stops, cancellation and other BaseException-only errors must still end the run.
        if not isinstance(exc, Exception):
            raise exc
        logger.error(f"{name} failed: {type(exc).__name__}: {exc}", exc_info=exc)
        state.add_log(f"{name} failed: {type(exc).__name__}")
        state.context.setdefault("research_errors", []).append({"agent": name, "error": str(exc)[:300]})
    ok = [r for r in results if not isinstance(r, BaseException)]
    if not ok:
        raise RuntimeError("; ".join(f"{name}: {exc}" for name, exc in failures))

    from app.security import ledger as security_ledger

    security_ledger.merge(state, ok)

    agent_report_names = {"web_research", "internal_research"}
    for result in ok:
        for name, report in result.reports.items():
            if name in agent_report_names:
                state.reports[name] = report
            else:
                state.reports.setdefault(name, report)
        for history_item in result.history:
            if history_item not in state.history:
                state.history.append(history_item)

    state.add_log("Research phase completed")
    logger.info("Research agents completed")
    return state

def _build_agent_workflow_impl(config=None, workflow=None) -> StateGraph:
    """
    Implementation of build_agent_workflow that allows for testing with mocks.
    
    Args:
        config: Configuration options for the workflow. Can be a dict or bool.
               If bool, treated as verbose flag.
        workflow: A pre-existing StateGraph instance to use (for testing).
    
    Returns:
        StateGraph: The compiled agent workflow graph.
    """
    # Handle different parameter types
    verbose = False
    if isinstance(config, bool):
        verbose = config
    elif isinstance(config, dict) and config:
        verbose = config.get("verbose", False)
    
    logger.info("Building agent workflow graph")
    start_time = time.time()
    
    # Get all agent instances
    logger.debug("Initializing agent instances")
    supervisor = get_supervisor_agent()
    document_extraction = get_document_extraction_agent()
    senior_research = get_senior_research_agent()
    data_analysis = get_data_analysis_agent()
    coding_assistant = get_coding_assistant_agent()
    coding_review = get_coding_review_agent()
    team_manager = get_team_manager_agent()
    logger.debug(f"All agent instances initialized in {time.time() - start_time:.2f} seconds")
    
    # Create a new graph or use the provided one
    logger.debug("Creating StateGraph with WorkflowState")
    if workflow is None:
        workflow = StateGraph(WorkflowState)
    
    # Add agent nodes to the graph
    logger.debug("Adding agent nodes to graph")
    workflow.add_node("supervisor", supervisor)
    workflow.add_node("document_extraction", document_extraction)
    workflow.add_node("senior_research", senior_research)
    workflow.add_node("data_analysis", data_analysis)
    workflow.add_node("coding_assistant", coding_assistant)
    workflow.add_node("coding_review", coding_review)
    workflow.add_node("team_manager", team_manager)
    
    # Add special node for parallel research execution
    logger.debug("Adding parallel research execution node")
    workflow.add_node("research", run_research_agents)
    
    # Set up the workflow edges
    logger.debug("Configuring graph edges and routes")
    # From start to supervisor
    workflow.add_edge("__start__", "supervisor")
    
    # From supervisor to different paths based on complexity
    workflow.add_conditional_edges(
        "supervisor",
        supervisor_router,
        {
            "document_extraction": "document_extraction",
            "research": "research",
            END: END
        }
    )
    logger.debug("Added conditional edges from supervisor")
    
    # From document extraction directly to END (simple case)
    workflow.add_edge("document_extraction", END)
    
    # From parallel research to senior research
    workflow.add_edge("research", "senior_research")
    
    # From senior research to either more research or team manager
    workflow.add_conditional_edges(
        "senior_research",
        senior_research_router,
        {
            "research": "research",
            "team_manager": "team_manager",
            END: END
        }
    )
    logger.debug("Added conditional edges from senior research")
    
    # From team manager to specialized agents or end
    workflow.add_conditional_edges(
        "team_manager",
        team_manager_router,
        {
            "data_analysis": "data_analysis",
            "coding_assistant": "coding_assistant",
            END: END
        }
    )
    logger.debug("Added conditional edges from team manager")
    
    # From specialized agents back to team manager
    workflow.add_edge("data_analysis", "team_manager")
    workflow.add_edge("coding_assistant", "coding_review")
    workflow.add_edge("coding_review", "team_manager")
    
    # Configure LangSmith tracing if enabled
    if settings.LANGCHAIN_TRACING_V2:
        logger.info("LangSmith tracing enabled for workflow")
        logger.debug(f"Using LangSmith project: {os.environ.get('LANGCHAIN_PROJECT', 'default')}")
    
    # Compile the workflow - ensure compile() is called explicitly
    logger.info("Compiling workflow graph")
    compiled_time = time.time()
    compiled_workflow = workflow.compile()
    
    build_time = time.time() - start_time
    compile_time = time.time() - compiled_time
    
    logger.info(f"Agent workflow built and compiled successfully in {build_time:.2f}s (compile: {compile_time:.2f}s)")
    if verbose:
        logger.debug(f"Workflow nodes: {list(workflow.nodes)}")
    
    return compiled_workflow

@timing_decorator
def build_agent_workflow(config=None) -> StateGraph:
    """
    Build the agent workflow graph.
    
    Args:
        config: Configuration options for the workflow. Can be a dict or bool.
               If bool, treated as verbose flag.
    
    Returns:
        StateGraph: The compiled agent workflow graph.
    """
    return _build_agent_workflow_impl(config)

@timing_decorator
def get_agent_workflow() -> StateGraph:
    """
    Get the agent workflow instance (singleton pattern).
    
    Returns:
        StateGraph: The compiled workflow graph
    """
    global _workflow_instance
    if _workflow_instance is None:
        logger.info("Creating new workflow instance (singleton)")
        _workflow_instance = build_agent_workflow()
    else:
        logger.debug("Returning existing workflow instance")
    return _workflow_instance
