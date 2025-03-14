"""
LangGraph workflow definitions for WebAgent backend.

This module contains the workflow definitions for the WebAgent backend,
including the agent workflow graph and routing logic.
"""
import logging
from typing import Dict, List, Optional, Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph

from backend.app.models.task import WorkflowState
from backend.app.agents.supervisor import get_supervisor_agent
from backend.app.agents.web_research import get_web_research_agent
from backend.app.agents.internal_research import get_internal_research_agent
from backend.app.agents.senior_research import get_senior_research_agent
from backend.app.agents.data_analysis import get_data_analysis_agent
from backend.app.agents.coding_assistant import get_coding_assistant_agent
from backend.app.agents.team_manager import get_team_manager_agent
from backend.app.core.config import settings

# Configure logging
logger = logging.getLogger(__name__)

# Singleton instance of the agent workflow
_workflow_instance = None

def build_agent_workflow() -> StateGraph:
    """
    Build the agent workflow graph.
    
    Returns:
        StateGraph: The compiled agent workflow graph.
    """
    logger.info("Building agent workflow graph")
    
    # Initialize agents
    supervisor = get_supervisor_agent()
    web_research = get_web_research_agent()
    internal_research = get_internal_research_agent()
    senior_research = get_senior_research_agent()
    data_analysis = get_data_analysis_agent()
    coding_assistant = get_coding_assistant_agent()
    team_manager = get_team_manager_agent()
    
    # Create a new graph
    workflow = StateGraph(WorkflowState)
    
    # Initialize the workflow graph
    workflow.add_node("supervisor", supervisor.run)
    workflow.add_node("web_research", web_research.run)
    workflow.add_node("internal_research", internal_research.run)
    workflow.add_node("senior_research", senior_research.run)
    workflow.add_node("data_analysis", data_analysis.run)
    workflow.add_node("coding_assistant", coding_assistant.run)
    workflow.add_node("team_manager", team_manager.run)
    
    # Define the research routing logic
    def research_router(state: WorkflowState) -> List[str]:
        """
        Determine which research agents should run based on the research plan.
        
        Args:
            state: The current workflow state
            
        Returns:
            List of agent nodes to execute
        """
        # Get the research plan from the supervisor's output
        research_plan = state.context.get("research_plan", {})
        
        # Check if this is a follow-up research request from senior research
        if state.context.get("continue_research", False):
            logger.info("Processing follow-up research request")
            next_agents = state.context.get("next_research_agents", [])
            if next_agents:
                logger.info(f"Routing to additional research agents: {next_agents}")
                return next_agents
        
        # Initial research routing
        next_agents = []
        
        # Determine which research agents to use
        if research_plan.get("requires_web_search", False):
            next_agents.append("web_research")
            
        if research_plan.get("requires_internal_knowledge", False):
            next_agents.append("internal_research")
            
        # If document extraction is required, handle it in the supervisor
        # and move directly to senior_research for analysis
        if research_plan.get("requires_document_extraction", False):
            logger.info("Document extraction required - handled by supervisor")
            
        # If no specific research agents are needed, skip to senior_research
        if not next_agents:
            logger.info("No specific research agents required. Proceeding to senior research.")
            
        return next_agents or ["senior_research"]
    
    # Define the research checkpoint condition for when all research is complete
    def research_checkpoint(state: WorkflowState) -> str:
        """
        Check if all required research is complete before proceeding.
        
        Args:
            state: The current workflow state
            
        Returns:
            Next node name
        """
        # Check if additional research is needed
        if state.context.get("continue_research", False):
            logger.info("Additional research requested by Senior Research Agent")
            return "research_router"
        
        # Default to team_manager as the next step
        return "team_manager"
    
    # Define the specialized agent router
    def specialized_agent_router(state: WorkflowState) -> str:
        """
        Route to specialized agents based on the research findings.
        
        Args:
            state: The current workflow state
            
        Returns:
            The next node to route to
        """
        # Check if there's an error
        if state.error:
            return "end"
            
        # Check if data analysis is needed
        if "data_analysis" in state.context.get("next_steps", []):
            return "data_analysis"
            
        # Check if coding assistance is needed
        if "coding_assistant" in state.context.get("next_steps", []):
            return "coding_assistant"
            
        # Default to team_manager as the next step
        return "team_manager"
    
    # Set up initial edges
    workflow.add_edge("__start__", "supervisor")
    workflow.add_edge("supervisor", "research_router")
    
    # Research edges
    workflow.add_conditional_edges(
        "research_router",
        research_router,
        {
            "web_research": lambda x: "web_research" in x,
            "internal_research": lambda x: "internal_research" in x,
            "senior_research": lambda x: "senior_research" in x
        }
    )
    
    # Research completion edges
    for agent in ["web_research", "internal_research"]:
        workflow.add_edge(agent, "senior_research")
    
    # Senior research edge with conditional routing for research loop
    workflow.add_conditional_edges(
        "senior_research",
        research_checkpoint,
        {
            "research_router": lambda x: x == "research_router",
            "team_manager": lambda x: x == "team_manager",
        }
    )
    
    # Senior research edge with conditional routing for specialized agents
    workflow.add_conditional_edges(
        "senior_research",
        specialized_agent_router,
        {
            "data_analysis": lambda x: x == "data_analysis",
            "coding_assistant": lambda x: x == "coding_assistant",
            "team_manager": lambda x: x == "team_manager",
            "end": lambda x: x == "end"
        }
    )
    
    # Add final edges
    workflow.add_edge("data_analysis", "team_manager")
    workflow.add_edge("coding_assistant", "team_manager")
    workflow.add_edge("team_manager", "end")
    
    # Compile the workflow
    workflow.compile()
    
    return workflow

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