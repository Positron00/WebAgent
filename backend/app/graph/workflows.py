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

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from langchain.callbacks.tracers import LangChainTracer

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

# Singleton router instance
_research_router_instance = None

def research_router(state: WorkflowState) -> str:
    """
    Determine the next step from the state after a research-related step.
    This is needed to support security tests.
    
    Args:
        state: The current workflow state
        
    Returns:
        The name of the next node in the workflow
    """
    # Check for research plan
    if state.context.get("research_plan", {}).get("requires_web_search", False):
        return "web_research"
    elif state.context.get("research_plan", {}).get("requires_internal_search", False):
        return "internal_research"
    else:
        # Default to web research if not specified
        return "web_research"

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
    logger.debug(f"Supervisor full output: {json.dumps(supervisor_output, default=str)}")
    
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
    
    logger.info(f"Senior Research status: {research_status}, loop count: {research_loops}/3")
    logger.debug(f"Senior Research full output: {json.dumps(senior_research_output, default=str)}")
    
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
    
    logger.info(f"Team Manager assessment - needs analysis: {needs_analysis}, needs coding: {needs_coding}")
    logger.debug(f"Team Manager full output: {json.dumps(team_manager_output, default=str)}")
    
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
    start_time = time.time()
    
    # Get agent instances
    internal_research = get_internal_research_agent()
    web_research = get_web_research_agent()
    
    # Create a new state for tracking
    new_state = WorkflowState(**state.dict())
    new_state.current_step = "research"
    
    try:
        # Run both agents concurrently
        logger.debug("Starting internal research agent")
        internal_result = await internal_research.ainvoke(new_state)
        logger.debug("Starting web research agent")
        web_result = await web_research.ainvoke(new_state)
        
        # Log response times
        logger.info(f"Research agents execution completed in {time.time() - start_time:.2f} seconds")
        
        # Update the state with both results
        state.update_with_agent_output("internal_research", internal_result)
        state.update_with_agent_output("web_research", web_result)
        
        # Log the size of results
        internal_docs = len(internal_result.get("documents", []) if internal_result else [])
        web_docs = len(web_result.get("documents", []) if web_result else [])
        logger.info(f"Research complete - internal documents: {internal_docs}, web documents: {web_docs}")
        
        # Move to senior research for verification
        state.current_step = "senior_research"
        return state
        
    except Exception as e:
        logger.exception("Error running research agents in parallel")
        state.mark_error(f"Error in research phase: {str(e)}")
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
    internal_research = get_internal_research_agent()
    web_research = get_web_research_agent()
    senior_research = get_senior_research_agent()
    data_analysis = get_data_analysis_agent()
    coding_assistant = get_coding_assistant_agent()
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
    workflow.add_node("internal_research", internal_research)
    workflow.add_node("web_research", web_research)
    workflow.add_node("senior_research", senior_research)
    workflow.add_node("data_analysis", data_analysis)
    workflow.add_node("coding_assistant", coding_assistant)
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
    workflow.add_edge("coding_assistant", "team_manager")
    
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

def get_research_router():
    """
    Get the research router function.
    
    This function determines which research agent(s) to call based on the research plan.
    
    Returns:
        A callable function that takes a WorkflowState and returns the next node name.
    """
    global _research_router_instance
    
    # If we don't have an instance yet, create one
    if _research_router_instance is None:
        def research_router(state: WorkflowState) -> str:
            """
            Route to appropriate research agent(s) based on the research plan.
            
            Args:
                state: The current workflow state
                
            Returns:
                String indicating the next node
            """
            # Check if there's an error
            if state.error:
                logger.warning(f"Error detected in workflow: {state.error}")
                return END
                
            # Get the research plan from the supervisor
            research_plan = state.context.get("research_plan", {})
            requires_web_search = research_plan.get("requires_web_search", True)
            requires_internal_knowledge = research_plan.get("requires_internal_knowledge", True)
            
            logger.info(f"Research plan - web search: {requires_web_search}, internal knowledge: {requires_internal_knowledge}")
            
            # Route based on the research plan
            if requires_web_search and requires_internal_knowledge:
                # Need both research types, use the parallel research node
                logger.info("Both web and internal research required, executing in parallel")
                return "research"
            elif requires_web_search:
                # Only web research needed
                logger.info("Only web research required")
                return "web_research"
            elif requires_internal_knowledge:
                # Only internal research needed
                logger.info("Only internal research required")
                return "internal_research"
            else:
                # No research needed, move to senior research for synthesis
                logger.info("No research required, proceeding to senior research for synthesis")
                return "senior_research"
        
        # Store the instance
        _research_router_instance = research_router
    
    return _research_router_instance 