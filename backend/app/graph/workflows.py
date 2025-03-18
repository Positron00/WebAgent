"""
LangGraph workflow definitions for WebAgent backend.

This module contains the workflow definitions for the WebAgent backend,
including the agent workflow graph and routing logic.
"""
import logging
from typing import Dict, List, Optional, Any, cast

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from langsmith.run_trees.run_tree import RunTree

from app.models.task import WorkflowState
from app.agents.supervisor import get_supervisor_agent
from app.agents.web_research import get_web_research_agent
from app.agents.internal_research import get_internal_research_agent
from app.agents.senior_research import get_senior_research_agent
from app.agents.data_analysis import get_data_analysis_agent
from app.agents.coding_assistant import get_coding_assistant_agent
from app.agents.team_manager import get_team_manager_agent
from app.agents.document_extraction_agent import get_document_extraction_agent
from app.core.config import settings
from app.core.metrics import timing_decorator

# Configure logging
logger = logging.getLogger(__name__)

# Singleton workflow instance
_workflow_instance = None

def supervisor_router(state: WorkflowState) -> str:
    """
    Determine the next step after the supervisor agent.
    
    Args:
        state: The current workflow state
        
    Returns:
        String indicating the next node
    """
    # Check if there's an error or if the workflow is completed
    if state.error:
        logger.warning(f"Error detected in workflow: {state.error}")
        return END
        
    if state.completed:
        logger.info("Workflow is marked as completed, ending")
        return END

    # Check if agentic toggle is on
    if not state.context.get("agentic_toggle", False):
        logger.info("Agentic toggle is off, ending workflow")
        return END
    
    # Get the supervisor's plan
    supervisor_output = state.reports.get("supervisor", {})
    task_type = supervisor_output.get("task_type", "unknown")
    
    logger.info(f"Supervisor determined task type: {task_type}")
    
    # Route based on task complexity
    if task_type == "document_extraction":
        logger.info("Simple document extraction task, routing to document extraction agent")
        return "document_extraction"
    elif task_type == "complex_research":
        logger.info("Complex task requiring research, routing to internal and web research")
        # The research agents will be called in parallel
        return "research"
    else:
        # Default handling for undefined task types
        logger.warning(f"Unknown task type: {task_type}, defaulting to document extraction")
        return "document_extraction"

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
    
    senior_research_output = state.reports.get("senior_research", {})
    research_status = senior_research_output.get("research_status", "incomplete")
    research_loops = state.context.get("research_loops", 0)
    
    # Check if we need more research (up to 3 loops)
    if research_status == "needs_more_research" and research_loops < 3:
        # Increment the research loop counter
        state.context["research_loops"] = research_loops + 1
        logger.info(f"Research loop {research_loops + 1}/3: Senior Research requesting more information")
        return "research"
    elif research_status == "needs_analysis":
        logger.info("Research complete, analysis needed")
        return "team_manager"
    else:
        # Either research is complete or we've reached the maximum number of loops
        logger.info(f"Research complete or max loops reached (loops: {research_loops}/3)")
        return "team_manager"

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
    needs_analysis = team_manager_output.get("needs_analysis", False)
    needs_coding = team_manager_output.get("needs_coding", False)
    
    if needs_analysis:
        logger.info("Team Manager requesting data analysis")
        return "data_analysis" 
    elif needs_coding:
        logger.info("Team Manager requesting coding assistance")
        return "coding_assistant"
    else:
        logger.info("Team Manager finished processing, ending workflow")
        return END

async def run_research_agents(state: WorkflowState) -> WorkflowState:
    """
    Run the internal and web research agents in parallel.
    
    Args:
        state: The current workflow state
        
    Returns:
        Updated workflow state with research results
    """
    logger.info("Running internal and web research agents in parallel")
    
    # Get agent instances
    internal_research = get_internal_research_agent()
    web_research = get_web_research_agent()
    
    # Create a new state for tracking
    new_state = WorkflowState(**state.dict())
    new_state.current_step = "research"
    
    try:
        # Run both agents concurrently
        internal_result = await internal_research.ainvoke(new_state)
        web_result = await web_research.ainvoke(new_state)
        
        # Update the state with both results
        state.update_with_agent_output("internal_research", internal_result)
        state.update_with_agent_output("web_research", web_result)
        
        logger.info("Successfully completed parallel research")
        
        # Move to senior research for verification
        state.current_step = "senior_research"
        return state
        
    except Exception as e:
        logger.exception("Error running research agents in parallel")
        state.mark_error(f"Error in research phase: {str(e)}")
        return state

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
    document_extraction = get_document_extraction_agent()
    internal_research = get_internal_research_agent()
    web_research = get_web_research_agent()
    senior_research = get_senior_research_agent()
    data_analysis = get_data_analysis_agent()
    coding_assistant = get_coding_assistant_agent()
    team_manager = get_team_manager_agent()
    
    # Create a new graph
    workflow = StateGraph(WorkflowState)
    
    # Add agent nodes to the graph
    workflow.add_node("supervisor", supervisor)
    workflow.add_node("document_extraction", document_extraction)
    workflow.add_node("internal_research", internal_research) 
    workflow.add_node("web_research", web_research)
    workflow.add_node("senior_research", senior_research)
    workflow.add_node("data_analysis", data_analysis)
    workflow.add_node("coding_assistant", coding_assistant)
    workflow.add_node("team_manager", team_manager)
    
    # Add special node for parallel research execution
    workflow.add_node("research", run_research_agents)
    
    # Set up the workflow edges
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
    
    # From specialized agents back to team manager
    workflow.add_edge("data_analysis", "team_manager")
    workflow.add_edge("coding_assistant", "team_manager")
    
    # Configure LangSmith tracing if enabled
    config = {"configurable": {"thread_id": "default"}}
    if settings.LANGCHAIN_TRACING_V2:
        logger.info("LangSmith tracing enabled for workflow")
        config["callbacks"] = [RunTree(run_type="chain")]
    
    # Compile the workflow
    compiled_workflow = workflow.compile(**config if settings.LANGCHAIN_TRACING_V2 else {})
    
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