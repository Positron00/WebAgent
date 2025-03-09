"""
LangGraph workflow definitions for the WebAgent backend.
"""
from typing import Any, Dict, List, Optional, Tuple
import logging

from langgraph.graph import StateGraph, START, END

from app.models.task import WorkflowState
from app.agents.supervisor import get_supervisor_agent
from app.agents.web_research import get_web_research_agent
from app.agents.internal_research import get_internal_research_agent
from app.agents.senior_research import get_senior_research_agent
from app.core.setup_langsmith import get_langchain_tracer, create_langsmith_tags
from app.core.loadEnvYAML import get_langsmith_config

logger = logging.getLogger(__name__)

def build_agent_workflow():
    """
    Build the multi-agent workflow graph.
    
    This function creates a graph of agent nodes and defines the flow between them.
    The graph includes Supervisor, Web Research, Internal Research, and Senior Research agents.
    """
    # Create state graph with our WorkflowState
    graph = StateGraph(WorkflowState)
    
    # Initialize the agents
    supervisor_agent = get_supervisor_agent()
    web_research_agent = get_web_research_agent()
    internal_research_agent = get_internal_research_agent()
    senior_research_agent = get_senior_research_agent()
    
    # Add the agent nodes to the graph
    graph.add_node("supervisor", supervisor_agent.run)
    graph.add_node("web_research", web_research_agent.run)
    graph.add_node("internal_research", internal_research_agent.run)
    graph.add_node("senior_research", senior_research_agent.run)
    
    # Define the decision function based on the supervisor's analysis
    def research_router(state: WorkflowState) -> List[str]:
        """
        Route the workflow based on the research plan.
        
        Args:
            state: The current workflow state
            
        Returns:
            List of next agent nodes to execute
        """
        # Get the research plan from the supervisor
        research_plan = state.context.get("research_plan", {})
        
        # Determine which agents to call based on the research plan
        next_agents = []
        
        # Check if web search is required
        if research_plan.get("requires_web_search", False):
            next_agents.append("web_research")
            
        # Check if internal knowledge is required
        if research_plan.get("requires_internal_knowledge", False):
            next_agents.append("internal_research")
        
        # If no specific research is required, use both by default
        if not next_agents:
            logger.info("No specific research requirements found. Using both web and internal research.")
            next_agents = ["web_research", "internal_research"]
            
        return next_agents
    
    # Define the checkpoint condition to check when all research is complete
    def research_checkpoint(state: WorkflowState) -> str:
        """
        Check if all required research is complete.
        
        Args:
            state: The current workflow state
            
        Returns:
            Next node name
        """
        # Check if both web and internal research have completed or been skipped
        web_completed = "web_research" in state.reports
        internal_completed = "internal_research" in state.reports
        
        # If at least one research agent has reported, proceed to senior research
        if web_completed or internal_completed:
            return "senior_research"
        
        # If no research is complete yet, wait (this shouldn't happen with synchronous execution)
        return None
    
    # Define the workflow routes
    # START -> supervisor
    graph.add_edge(START, "supervisor")
    
    # supervisor -> research agents (conditional)
    graph.add_conditional_edges(
        "supervisor",
        research_router,
        {
            "web_research": "web_research",
            "internal_research": "internal_research"
        }
    )
    
    # Research agents -> research_checkpoint
    graph.add_edge("web_research", research_checkpoint)
    graph.add_edge("internal_research", research_checkpoint)
    
    # research_checkpoint -> senior_research -> END
    graph.add_edge("senior_research", END)
    
    # If there's an error in the supervisor, end the workflow
    graph.add_edge("supervisor", END, condition=lambda state: state.error is not None)
    
    # Compile the graph
    workflow = graph.compile()
    
    # Add LangSmith tracing if enabled
    langsmith_config = get_langsmith_config()
    if langsmith_config.tracing_enabled:
        tracer = get_langchain_tracer()
        if tracer:
            tags = create_langsmith_tags(agent_name="research_workflow")
            logger.info(f"Enabled LangSmith tracing for research workflow with project: {langsmith_config.project_name}")
            workflow = workflow.with_tracing(tracer=tracer, tags=tags)
    
    logger.info("Agent workflow graph built and compiled successfully.")
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