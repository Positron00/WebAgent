#!/usr/bin/env python
"""
WebAgent Example Script
======================

This script demonstrates how to use the WebAgent platform
for different types of requests and workflows.
"""
import os
import sys
import time
import json
import logging
from pathlib import Path
import argparse

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import WebAgent modules
from backend.path_setup import setup_path
setup_path()  # Ensure proper import paths

from backend.app.agents.supervisor import get_supervisor_agent
from backend.app.graph.workflows import get_agent_workflow
from backend.app.models.task import WorkflowState
from backend.app.core.logger import setup_logger
from backend.app.core.config import settings

# Set up logging
logger = setup_logger("webagent_example", log_level=logging.INFO)

def direct_agent_request(request_type, query=None, document_path=None):
    """
    Make a direct request to a specific agent.
    
    Args:
        request_type: Type of request (document_extraction, research, etc.)
        query: Query text for research requests
        document_path: Path to document for extraction
        
    Returns:
        Result from the agent
    """
    logger.info(f"Making a direct {request_type} request")
    
    # Initialize the supervisor agent
    supervisor = get_supervisor_agent()
    
    # Create the request
    request = {"request_type": request_type}
    
    if query:
        request["query"] = query
    
    if document_path:
        request["document_path"] = document_path
        
    # Process the request
    start_time = time.time()
    result = supervisor.process_request(request)
    elapsed_time = time.time() - start_time
    
    logger.info(f"Request completed in {elapsed_time:.2f} seconds")
    return result

def workflow_request(query):
    """
    Run a full agent workflow for a given query.
    
    Args:
        query: User query to process
        
    Returns:
        Final workflow state
    """
    logger.info(f"Running workflow for query: {query}")
    
    # Get the agent workflow
    workflow = get_agent_workflow()
    
    # Create initial state
    state = WorkflowState(query=query)
    
    # Execute the workflow
    start_time = time.time()
    final_state = workflow.invoke(state)
    elapsed_time = time.time() - start_time
    
    logger.info(f"Workflow completed in {elapsed_time:.2f} seconds")
    return final_state

def print_diagnostics():
    """Print diagnostic information about the system."""
    logger.info("=== WebAgent Diagnostics ===")
    
    # Get environment information
    logger.info(f"Python version: {sys.version}")
    logger.info(f"WebAgent version: {settings.VERSION}")
    logger.info(f"Environment: {settings.WEBAGENT_ENV}")
    
    # Check supervisor agent
    try:
        supervisor = get_supervisor_agent()
        status = supervisor.get_agent_status()
        logger.info(f"Supervisor status: {json.dumps(status, indent=2)}")
    except Exception as e:
        logger.error(f"Error getting supervisor status: {str(e)}")
    
    # Check workflow
    try:
        workflow = get_agent_workflow()
        nodes = workflow.nodes
        logger.info(f"Workflow nodes: {', '.join(nodes)}")
    except Exception as e:
        logger.error(f"Error getting workflow: {str(e)}")
    
    logger.info("=== End Diagnostics ===")

def main():
    """Main function to run example requests."""
    parser = argparse.ArgumentParser(description="WebAgent Example Script")
    
    parser.add_argument("--mode", choices=["direct", "workflow", "diagnostics"], 
                        default="diagnostics", help="Mode to run in")
    parser.add_argument("--type", choices=["document_extraction", "research", "team_management"],
                        help="Request type for direct mode")
    parser.add_argument("--query", help="Query for research requests")
    parser.add_argument("--document", help="Document path for extraction")
    
    args = parser.parse_args()
    
    # Run in specified mode
    if args.mode == "direct":
        if not args.type:
            logger.error("Request type is required for direct mode")
            return
        
        result = direct_agent_request(
            request_type=args.type,
            query=args.query,
            document_path=args.document
        )
        print(json.dumps(result, indent=2))
        
    elif args.mode == "workflow":
        if not args.query:
            logger.error("Query is required for workflow mode")
            return
        
        state = workflow_request(args.query)
        
        # Print final report
        if state.final_report:
            print("\n=== Final Report ===")
            print(json.dumps(state.final_report, indent=2))
        
        if state.error:
            print(f"\nError: {state.error}")
    
    elif args.mode == "diagnostics":
        print_diagnostics()

if __name__ == "__main__":
    main() 