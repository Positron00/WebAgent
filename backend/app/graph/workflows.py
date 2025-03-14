"""
LangGraph workflow definitions for WebAgent backend.

This module contains the workflow definitions for the WebAgent backend,
including the agent workflow graph and routing logic.
"""
import logging
from typing import Dict, List, Optional, Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph

from app.models.task import WorkflowState
from app.agents.supervisor import get_supervisor_agent
from app.agents.web_research import get_web_research_agent
from app.agents.internal_research import get_internal_research_agent
from app.agents.senior_research import get_senior_research_agent
from app.agents.data_analysis import get_data_analysis_agent
from app.agents.coding_assistant import get_coding_assistant_agent
from app.agents.team_manager import get_team_manager_agent
from app.core.config import settings
from app.core.metrics import timing_decorator

# Configure logging
logger = logging.getLogger(__name__)

# Singleton workflow instance
_workflow_instance = None

def get_research_router():
    """
    Get the research router agent.
    
    This is a simple router that directs research requests to the appropriate agent.
    It's implemented as a function returning a dict with routing logic rather than a full agent class.
    
    Returns:
        A dict with routing logic for research requests
    """
    return {
        "name": "research_router",
        "description": "Routes research requests to web, internal, or senior research agents",
        "route": lambda state: state.context.get("research_route", "web_research")
    }

def research_router(state: WorkflowState) -> str:
    """
    Determine which research agent to route to.
    
    Args:
        state: The current workflow state
        
    Returns:
        String indicating the next node
    """
    # Check if there's an error
    if state.error:
        logger.warning(f"Error detected in workflow: {state.error}")
        return END
        
    # Check if the workflow is completed
    if state.completed:
        logger.info("Workflow is marked as completed, ending")
        return END
        
    plan = state.context.get("research_plan", {})
    
    # Debug logging
    logger.debug(f"Research plan: {plan}")
    logger.debug(f"Current step: {state.current_step}")
    
    # Route based on the research plan from the supervisor
    if plan.get("requires_web_search", False):
        logger.info("Web research requested")
        return "web_research"
    elif plan.get("requires_internal_knowledge", False):
        logger.info("Internal research requested")
        return "internal_research"
    else:
        # Default to senior research if no specific research plan exists
        logger.info("No specific research type specified, routing to senior research")
        return "senior_research"

@timing_decorator
def build_agent_workflow(verbose: bool = False) -> StateGraph:
    """
    Build the agent workflow graph.
    
    Returns:
        StateGraph: The compiled agent workflow graph.
    """
    logger.info("Building agent workflow graph")
    
    # Get all agent instances
    supervisor = get_supervisor_agent()
    web_research = get_web_research_agent()
    internal_research = get_internal_research_agent()
    senior_research = get_senior_research_agent()
    data_analysis = get_data_analysis_agent()
    coding_assistant = get_coding_assistant_agent()
    team_manager = get_team_manager_agent()
    
    # Create a new graph
    workflow = StateGraph(WorkflowState)
    
    # Add all agent nodes to the graph
    workflow.add_node("supervisor", supervisor)
    workflow.add_node("web_research", web_research)
    workflow.add_node("internal_research", internal_research) 
    workflow.add_node("senior_research", senior_research)
    workflow.add_node("data_analysis", data_analysis)
    workflow.add_node("coding_assistant", coding_assistant)
    workflow.add_node("team_manager", team_manager)
    
    # Set up direct edges
    # From start to supervisor
    workflow.add_edge("__start__", "supervisor")
    
    # From supervisor to research agents based on routing decision
    workflow.add_conditional_edges(
        "supervisor",
        lambda state: research_router(state),
        {
            "web_research": "web_research",
            "internal_research": "internal_research",
            "senior_research": "senior_research"
        }
    )
    
    # From research agents back to senior research for further analysis
    workflow.add_edge("web_research", "senior_research")
    workflow.add_edge("internal_research", "senior_research")
    
    # From senior research to all possible next steps
    # Using direct edges to avoid branch name conflicts
    workflow.add_edge("senior_research", "supervisor")  # For additional research
    workflow.add_edge("senior_research", "data_analysis")  # For data analysis
    workflow.add_edge("senior_research", "coding_assistant")  # For code generation
    workflow.add_edge("senior_research", "team_manager")  # For final report
    workflow.add_edge("senior_research", END)  # For error cases
    
    # From specialized agents back to team manager for final report
    workflow.add_edge("data_analysis", "team_manager")
    workflow.add_edge("coding_assistant", "team_manager")
    
    # Final step completes the workflow
    workflow.add_edge("team_manager", END)
    
    # Compile the workflow
    compiled_workflow = workflow.compile()
    
    logger.info("Agent workflow built and compiled successfully")
    return compiled_workflow

@timing_decorator
def get_agent_workflow() -> StateGraph:
    """
    Get the agent workflow instance (singleton pattern).
    
    Returns:
        StateGraph: The compiled workflow graph
    """
    global _workflow_instance
    if _workflow_instance is None:
        _workflow_instance = build_agent_workflow()
    return _workflow_instance 