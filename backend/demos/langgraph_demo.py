#!/usr/bin/env python
"""
LangGraph MultiAgent Demo

This script demonstrates the LangGraph multiagent workflow in action.
It configures a simple query execution pipeline and logs the entire process.

Run with: python -m backend.demos.langgraph_demo
"""
import os
import sys
import json
import time
import asyncio
import logging
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("langgraph_demo")

# Add project root to path
script_dir = Path(__file__).resolve().parent
backend_dir = script_dir.parent
project_root = backend_dir.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(backend_dir))

# Import required modules
from app.models.task import WorkflowState
from app.graph.workflows import build_agent_workflow
from app.core.config import settings

# Set environment variables for the demo
os.environ["WEBAGENT_ENV"] = "dev"
os.environ["LOG_LEVEL"] = "DEBUG"
os.environ["LANGCHAIN_TRACING_V2"] = "false"  # Disable LangSmith for the demo

async def run_workflow_demo():
    """Run a demonstration of the LangGraph workflow."""
    logger.info("Starting LangGraph MultiAgent Demo")
    print("\n===== WebAgent LangGraph MultiAgent Demo =====\n")
    
    # Create a sample query
    query = input("Enter a query (or press Enter for default): ").strip()
    if not query:
        query = "What are the latest developments in large language model applications for business?"
        print(f"Using default query: '{query}'")
    
    # Create initial workflow state
    state = WorkflowState(
        query=query,
        context={
            "agentic_toggle": True,  # Enable agentic mode
            "user_id": "demo_user",
            "session_id": f"demo_{int(time.time())}",
            "timestamp": datetime.now().isoformat()
        },
        current_step="planning"
    )
    
    print(f"\nQuery: {query}")
    print("Creating workflow...")
    
    # Build the agent workflow
    start_time = time.time()
    workflow = build_agent_workflow(verbose=True)
    print(f"Workflow built in {time.time() - start_time:.2f} seconds")
    
    # Execute the workflow
    print("\nExecuting workflow...\n")
    execution_start = time.time()
    
    try:
        # Invoke the workflow
        result = await workflow.ainvoke(state)
        
        # Calculate total execution time
        total_time = time.time() - execution_start
        
        print(f"\nWorkflow completed in {total_time:.2f} seconds")
        print(f"Final state: {result.current_step}")
        
        # Summary of agent reports
        print("\n===== Agent Reports =====")
        for agent_name, report in result.reports.items():
            print(f"\n{agent_name.upper()} REPORT:")
            if isinstance(report, dict):
                summary = {k: v for k, v in report.items() if k not in ['documents', 'content', 'full_response']}
                print(json.dumps(summary, indent=2, default=str))
            else:
                print(f"  {report}")
                
        # Print final report if available
        if result.final_report:
            print("\n===== FINAL REPORT =====")
            print(json.dumps(result.final_report, indent=2, default=str))
            
        # Print execution trace
        print("\n===== Execution History =====")
        for step in result.history:
            print(f"{step.get('timestamp')}: {step.get('agent')} executed")
        
        return result
        
    except Exception as e:
        logger.exception("Error executing workflow")
        print(f"\nError: {str(e)}")
        return None

if __name__ == "__main__":
    try:
        result = asyncio.run(run_workflow_demo())
        if result:
            print("\nDemo completed successfully")
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        logger.exception("Unexpected error")
        print(f"\nUnexpected error: {str(e)}") 