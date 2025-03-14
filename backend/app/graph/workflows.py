"""
LangGraph workflow definitions for the WebAgent backend.
"""
from typing import Any, Dict, List, Optional, Tuple
import logging
import os

from langgraph.graph import StateGraph, START, END

from app.models.task import WorkflowState
from app.agents.supervisor import get_supervisor_agent
from app.agents.web_research import get_web_research_agent
from app.agents.internal_research import get_internal_research_agent
from app.agents.senior_research import get_senior_research_agent
from app.agents.data_analysis import get_data_analysis_agent
from app.agents.coding_assistant import get_coding_assistant_agent
from app.agents.team_manager import get_team_manager_agent
from app.core.setup_langsmith import get_langchain_tracer, create_langsmith_tags
from app.core.loadEnvYAML import get_langsmith_config

logger = logging.getLogger(__name__)

def build_agent_workflow():
    """
    Build the agent workflow graph.
    
    Returns:
        StateGraph: The agent workflow graph
    """
    # Set up loggers
    logger = logging.getLogger(__name__)
    logger.info("Building agent workflow graph")
    
    # Get agent instances
    supervisor_agent = get_supervisor_agent()
    web_research_agent = get_web_research_agent()
    internal_research_agent = get_internal_research_agent()
    senior_research_agent = get_senior_research_agent()
    data_analysis_agent = get_data_analysis_agent()
    coding_assistant_agent = get_coding_assistant_agent()
    team_manager_agent = get_team_manager_agent()
    advanced_agent = get_advanced_agent()
    
    # Initialize the workflow graph
    workflow = StateGraph(WorkflowState)
    
    # Add all nodes to the graph
    workflow.add_node("supervisor", supervisor_agent.run)
    workflow.add_node("web_research", web_research_agent.run)
    workflow.add_node("internal_research", internal_research_agent.run)
    workflow.add_node("senior_research", senior_research_agent.run)
    workflow.add_node("data_analysis", data_analysis_agent.run)
    workflow.add_node("coding_assistant", coding_assistant_agent.run)
    workflow.add_node("team_manager", team_manager_agent.run)
    workflow.add_node("advanced_agent", advanced_agent.run)
    
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
        
        # Default to advanced_agent as the next step
        return "advanced_agent"
    
    # Set up initial edges
    workflow.add_edge("__start__", "supervisor")
    workflow.add_edge("supervisor", research_router)
    
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
    
    # Senior research edge with conditional routing
    workflow.add_conditional_edges(
        "senior_research",
        research_checkpoint,
        {
            "research_router": lambda x: x == "research_router",
            "advanced_agent": lambda x: x == "advanced_agent",
        }
    )
    
    # Advanced agent edge
    workflow.add_edge("advanced_agent", "end")
    
    # Compile the workflow
    workflow.compile()
    logger.info("Agent workflow graph compiled successfully")
    
    return workflow

# Singleton instance of the workflow
_workflow_instance = None

def get_agent_workflow():
    """
    Get the agent workflow instance (singleton pattern).
    """
    global _workflow_instance
    if _workflow_instance is None:
        _workflow_instance = build_agent_workflow()
    return _workflow_instance 