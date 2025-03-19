#!/usr/bin/env python
"""
LangGraph Multiagent Workflow Demo
==================================

This demo showcases the capabilities of the WebAgent's LangGraph multiagent workflow,
demonstrating how multiple agents collaborate to solve complex queries.

The workflow includes:
- Supervisor Agent: Plans and orchestrates the research process
- Document Extraction Agent: Extracts information from documents (simple path)
- Research Agents: Gather information from web and internal sources (complex path)
- Senior Research Agent: Verifies research against requirements (up to 3 loops)
- Data Analysis Agent: Analyzes research findings (when needed)
- Coding Assistant Agent: Generates code for visualizations (when needed)
- Team Manager Agent: Synthesizes results into a final comprehensive report

Usage:
    python -m backend.demos.multiagent_workflow_demo --query "Your research query here" [--agentic True/False]

Requirements:
    - Configured environment variables (check .env.example)
    - LangSmith API key for tracing (optional but recommended)
"""

import sys
import os
import time
import argparse
import asyncio
import logging
import warnings
from datetime import datetime
from pathlib import Path
import json
from typing import Dict, Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Suppress specific warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning"

# Define rate limiting parameters for Together AI API
TOGETHER_RATE_LIMIT_BUFFER = 12.0  # Buffer time in seconds (higher than the 10 req/min limit)
_last_together_api_call = 0

def enforce_together_api_rate_limit():
    """
    Enforce rate limiting for Together AI API calls.
    This ensures we don't exceed the rate limit of about 6 queries per minute
    for the free tier of Llama 3.3 70B Instruct Turbo model.
    """
    global _last_together_api_call
    current_time = time.time()
    elapsed = current_time - _last_together_api_call
    
    if _last_together_api_call > 0 and elapsed < TOGETHER_RATE_LIMIT_BUFFER:
        sleep_time = TOGETHER_RATE_LIMIT_BUFFER - elapsed
        logger.info(f"Rate limiting: Sleeping for {sleep_time:.2f} seconds to avoid Together AI rate limit")
        time.sleep(sleep_time)
    
    _last_together_api_call = time.time()

# Setup path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Configure detailed logging for demo
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"demos/logs/workflow_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger("multiagent_workflow_demo")

# Import workflow components
from backend.app.graph.workflows import get_agent_workflow
from backend.app.models.task import WorkflowState
from backend.app.core.config import settings
from backend.app.core.metrics import timing_decorator
from backend.app.core.setup_langsmith import setup_langsmith, get_langsmith_client

# Set up LangSmith tracing if available
if settings.LANGSMITH_API_KEY:
    import os
    
    # Configure LangSmith for this demo (using the standard setup utility)
    setup_langsmith()
    
    # Set a demo-specific project name
    os.environ["LANGCHAIN_PROJECT"] = f"webagent-workflow-demo-{datetime.now().strftime('%Y%m%d')}"
    
    # Get the client to verify it's working
    client = get_langsmith_client()
    if client:
        logger.info(f"LangSmith tracing enabled with project: {os.environ['LANGCHAIN_PROJECT']}")
    else:
        logger.warning("LangSmith client initialization failed. Check your API key.")
else:
    logger.warning("LangSmith API key not found. Tracing will not be available.")

# Define demo sample queries
SAMPLE_QUERIES = {
    "document_extraction": "Summarize the key points from the attached financial report, focusing on revenue growth and profit margins.",
    "web_research": "What are the latest advancements in large language model fine-tuning techniques in 2025?",
    "complex_research": "Compare renewable energy adoption rates in Europe versus North America over the past 5 years, focusing on solar and wind power.",
    "data_analysis": "Analyze the correlation between global temperature increases and extreme weather events over the past decade.",
    "coding": "Create a visualization showing the relationship between CO2 emissions and GDP growth for the top 20 economies."
}

def print_banner(title: str):
    """Print a formatted banner for demo sections."""
    width = 80
    padding = (width - len(title) - 2) // 2
    banner = "#" * width
    logger.info(banner)
    logger.info("#" + " " * padding + title + " " * (width - padding - len(title) - 2) + "#")
    logger.info(banner)

def print_separator():
    """Print a separator line."""
    logger.info("=" * 80)

def format_json(data: Dict[str, Any]) -> str:
    """Format JSON data for pretty display."""
    return json.dumps(data, indent=2, ensure_ascii=False)

def format_agent_output(agent: str, output: Dict[str, Any]) -> None:
    """Format and display agent output with highlighting."""
    print_separator()
    logger.info(f"🤖 {agent.upper()} AGENT OUTPUT")
    print_separator()
    
    # Handle different output formats based on agent
    if agent == "supervisor":
        if "task_type" in output:
            logger.info(f"Task Type: {output.get('task_type', 'Not specified')}")
        if "requirements" in output:
            logger.info("Requirements:")
            for i, req in enumerate(output.get("requirements", []), 1):
                logger.info(f"  {i}. {req}")
        if "research_plan" in output:
            research_plan = output.get("research_plan", {})
            logger.info("Research Plan:")
            logger.info(f"  Web Research Needed: {research_plan.get('needs_web_research', False)}")
            logger.info(f"  Internal Research Needed: {research_plan.get('needs_internal_research', False)}")
            if "questions" in research_plan:
                logger.info("  Research Questions:")
                for i, q in enumerate(research_plan.get("questions", []), 1):
                    logger.info(f"    {i}. {q}")
    elif agent == "senior_research":
        logger.info(f"Research Status: {output.get('research_status', 'Not specified')}")
        logger.info(f"Current Iteration: {output.get('current_iteration', 1)}")
        
        if "requirements_met" in output:
            logger.info("Requirements Met:")
            for req in output.get("requirements_met", []):
                logger.info(f"  • {req}")
        
        if "missing_information" in output and output.get("missing_information"):
            logger.info("Missing Information:")
            for info in output.get("missing_information", []):
                logger.info(f"  • {info}")
        
        if "research_questions" in output and output.get("research_questions"):
            logger.info("Additional Research Questions:")
            for q in output.get("research_questions", []):
                logger.info(f"  • {q}")
    elif agent == "team_manager":
        logger.info(f"Status: {output.get('status', 'Not specified')}")
        logger.info(f"Needs Data Analysis: {output.get('needs_analysis', False)}")
        logger.info(f"Needs Coding: {output.get('needs_coding', False)}")
        
        if "summary" in output:
            logger.info("Summary:")
            logger.info(f"  {output.get('summary', 'No summary provided')}")
        
        if "recommendations" in output:
            logger.info("Recommendations:")
            for rec in output.get("recommendations", []):
                logger.info(f"  • {rec}")
    elif agent in ["data_analysis", "coding_assistant"]:
        if "summary" in output:
            logger.info(f"Summary: {output.get('summary', 'No summary provided')}")
        
        if "status" in output:
            logger.info(f"Status: {output.get('status', 'Not specified')}")
    else:
        # Generic output display for other agents
        for key, value in output.items():
            if isinstance(value, (dict, list)) and len(str(value)) > 100:
                logger.info(f"{key}: [Complex data structure - see detailed logs]")
            else:
                logger.info(f"{key}: {value}")
    
    print_separator()

def watch_state_updates(state: WorkflowState, prev_state: Optional[WorkflowState] = None) -> WorkflowState:
    """Watch for state updates and display changes."""
    if prev_state is None:
        logger.info("Initial workflow state created")
        return state
    
    # Check for new agent outputs
    for agent_name, output in state.reports.items():
        if agent_name not in prev_state.reports:
            logger.info(f"New agent output received from: {agent_name}")
            format_agent_output(agent_name, output)
    
    # Check for state transitions
    if state.current_step != prev_state.current_step:
        logger.info(f"Workflow step transition: {prev_state.current_step} -> {state.current_step}")
    
    # Check for completion or errors
    if state.completed and not prev_state.completed:
        logger.info("Workflow marked as completed")
    
    if state.error and not prev_state.error:
        logger.error(f"Workflow error detected: {state.error}")
    
    return state.copy()

@timing_decorator
async def run_workflow_demo(query: str, agentic_toggle: bool = True) -> Dict[str, Any]:
    """
    Run the multiagent workflow demo with the given query.
    
    Args:
        query: The user query to process
        agentic_toggle: Whether to enable the agentic workflow (True) or use direct response (False)
        
    Returns:
        The final result dictionary
    """
    print_banner("LANGGRAPH MULTIAGENT WORKFLOW DEMO")
    logger.info(f"Query: {query}")
    logger.info(f"Agentic Workflow: {'Enabled' if agentic_toggle else 'Disabled'}")
    
    # Create a context for the workflow
    context = {
        "agentic_toggle": agentic_toggle,
        "previous_messages": [],
        "start_time": datetime.now().isoformat()
    }
    
    # Initialize the workflow state
    initial_state = WorkflowState(
        query=query,
        context=context,
        current_step="initializing"
    )
    
    # Get the workflow
    workflow = get_agent_workflow()
    logger.info("Agent workflow loaded successfully")
    
    try:
        # Set up state tracking
        prev_state = None
        prev_step = None
        
        # Create an event handler to track state changes
        async def state_tracker(state):
            nonlocal prev_state, prev_step
            
            # Track step changes to add pauses between different agent calls
            if state.current_step != prev_step and prev_step is not None:
                logger.info(f"Step change from {prev_step} to {state.current_step}, adding delay to manage API rate limits")
                await asyncio.sleep(2)  # Add a small delay between agent steps
            
            prev_step = state.current_step
            prev_state = watch_state_updates(state, prev_state)
            return state
        
        # Execute the workflow
        logger.info("Starting workflow execution...")
        final_state = await workflow.ainvoke(
            initial_state,
            {"callbacks": [state_tracker]}
        )
        
        logger.info("Workflow execution completed")
        
        # Log final state information
        if final_state.completed:
            logger.info("Workflow completed successfully")
            
            if final_state.final_report:
                print_banner("FINAL REPORT")
                logger.info(f"Title: {final_state.final_report.get('title', 'No title')}")
                logger.info(f"Type: {final_state.final_report.get('type', 'Not specified')}")
                
                content = final_state.final_report.get('content', '')
                if len(content) > 1000:
                    logger.info(f"Content: [First 1000 chars of {len(content)} total]")
                    logger.info(content[:1000] + "...")
                else:
                    logger.info("Content:")
                    logger.info(content)
                
                result = {
                    "status": "completed",
                    "title": final_state.final_report.get('title', 'No title'),
                    "content": final_state.final_report.get('content', ''),
                    "type": final_state.final_report.get('type', 'Not specified'),
                    "sources": final_state.final_report.get('sources', [])
                }
            else:
                logger.warning("Workflow completed but no final report was generated")
                result = {
                    "status": "incomplete",
                    "error": "No final report generated"
                }
        elif final_state.error:
            logger.error(f"Workflow failed with error: {final_state.error}")
            result = {
                "status": "error",
                "error": final_state.error
            }
        else:
            logger.warning("Workflow ended without completion or error")
            result = {
                "status": "incomplete",
                "message": "Workflow ended without completion"
            }
        
        # Print final execution statistics
        print_banner("WORKFLOW EXECUTION STATISTICS")
        logger.info(f"Total agents involved: {len(final_state.reports)}")
        logger.info(f"Research loops executed: {final_state.context.get('research_loops', 0)}")
        logger.info(f"Final step: {final_state.current_step}")
        
        return result
    
    except Exception as e:
        logger.exception(f"Error running workflow demo: {str(e)}")
        return {
            "status": "error",
            "error": str(e)
        }

def main():
    """Run the demo with command line arguments."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="LangGraph Multiagent Workflow Demo")
    parser.add_argument("--query", type=str, help="Research query to process")
    parser.add_argument("--agentic", type=lambda x: x.lower() == 'true', default=True, 
                       help="Enable agentic workflow (True/False)")
    parser.add_argument("--sample", type=str, choices=list(SAMPLE_QUERIES.keys()),
                       help="Use a sample query (document_extraction, web_research, complex_research, data_analysis, coding)")
    args = parser.parse_args()
    
    # Get the query from args or samples
    query = None
    if args.query:
        query = args.query
    elif args.sample:
        query = SAMPLE_QUERIES[args.sample]
        logger.info(f"Using sample query for {args.sample}")
    else:
        query = SAMPLE_QUERIES["complex_research"]
        logger.info("No query provided, using default complex research sample")
    
    # Run the workflow demo
    result = asyncio.run(run_workflow_demo(query, args.agentic))
    
    # Print completion message
    if result["status"] == "completed":
        logger.info("Demo completed successfully!")
    else:
        logger.warning(f"Demo ended with status: {result['status']}")
    
    logger.info("See the detailed logs for full agent interactions and workflow process")
    if settings.LANGSMITH_API_KEY and settings.LANGSMITH_TRACING_ENABLED:
        logger.info(f"LangSmith traces available in project: {os.environ.get('LANGCHAIN_PROJECT')}")
        logger.info(f"View them at: https://smith.langchain.com/projects/{os.environ.get('LANGCHAIN_PROJECT')}")

if __name__ == "__main__":
    main() 