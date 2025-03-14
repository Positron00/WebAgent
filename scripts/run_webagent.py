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
import traceback

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
from backend.app.utils.diagnostics import print_diagnostics_report, get_diagnostics

# Set up logging
logger = setup_logger("webagent_example", log_level=logging.INFO)

def direct_agent_request(request_type, query=None, document_path=None, output_format="json"):
    """
    Make a direct request to a specific agent.
    
    Args:
        request_type: Type of request (document_extraction, research, etc.)
        query: Query text for research requests
        document_path: Path to document for extraction
        output_format: Format for output (json, text, markdown)
        
    Returns:
        Result from the agent
    """
    logger.info(f"Making a direct {request_type} request")
    
    try:
        # Initialize the supervisor agent
        supervisor = get_supervisor_agent()
        
        # Create the request
        request = {"request_type": request_type}
        
        if query:
            request["query"] = query
        
        if document_path:
            # Validate document path
            doc_path = Path(document_path)
            if not doc_path.exists():
                raise FileNotFoundError(f"Document not found: {document_path}")
                
            request["document_path"] = str(doc_path.resolve())
            
        # Process the request
        start_time = time.time()
        result = supervisor.process_request(request)
        elapsed_time = time.time() - start_time
        
        logger.info(f"Request completed in {elapsed_time:.2f} seconds")
        
        # Format the output
        if output_format == "json":
            return result
        elif output_format == "text":
            return format_result_as_text(result)
        elif output_format == "markdown":
            return format_result_as_markdown(result)
        else:
            return result
            
    except Exception as e:
        logger.error(f"Error in direct agent request: {str(e)}")
        logger.debug(traceback.format_exc())
        return {"error": str(e), "traceback": traceback.format_exc()}

def workflow_request(query, max_iterations=3, timeout=300):
    """
    Run a full agent workflow for a given query.
    
    Args:
        query: User query to process
        max_iterations: Maximum number of research iterations
        timeout: Maximum time in seconds to wait for completion
        
    Returns:
        Final workflow state
    """
    logger.info(f"Running workflow for query: {query}")
    
    try:
        # Get the agent workflow
        workflow = get_agent_workflow()
        
        # Create initial state with configuration
        state = WorkflowState(
            query=query,
            context={
                "max_iterations": max_iterations,
                "timeout": timeout
            }
        )
        
        # Execute the workflow
        start_time = time.time()
        final_state = workflow.invoke(state)
        elapsed_time = time.time() - start_time
        
        logger.info(f"Workflow completed in {elapsed_time:.2f} seconds")
        
        # Check for errors
        if final_state.error:
            logger.error(f"Workflow error: {final_state.error}")
            
        return final_state
        
    except Exception as e:
        logger.error(f"Error in workflow request: {str(e)}")
        logger.debug(traceback.format_exc())
        state = WorkflowState(query=query)
        state.error = str(e)
        return state

def format_result_as_text(result):
    """Format result as plain text."""
    if "content" in result:
        return result["content"]
    elif "extracted_data" in result:
        text = "Extracted Data:\n\n"
        for key, value in result["extracted_data"].items():
            text += f"{key}: {value}\n"
        return text
    else:
        return str(result)

def format_result_as_markdown(result):
    """Format result as markdown."""
    if "content" in result:
        return result["content"]
    elif "extracted_data" in result:
        md = "# Extracted Data\n\n"
        for key, value in result["extracted_data"].items():
            md += f"## {key}\n{value}\n\n"
        return md
    else:
        return f"```json\n{json.dumps(result, indent=2)}\n```"

def run_diagnostics(check_network=False, check_llm=False):
    """
    Run system diagnostics.
    
    Args:
        check_network: Whether to check network connectivity
        check_llm: Whether to check LLM availability
    """
    logger.info("Running WebAgent diagnostics")
    
    try:
        # Use the diagnostics module
        diagnostics = get_diagnostics()
        
        # Selectively disable certain checks
        if not check_network:
            diagnostics.check_network_connectivity = lambda: {"status": "skipped"}
            
        if not check_llm:
            diagnostics.check_language_model_availability = lambda: {"status": "skipped"}
            
        # Print the diagnostics report
        print_diagnostics_report()
        
    except Exception as e:
        logger.error(f"Error running diagnostics: {str(e)}")
        logger.debug(traceback.format_exc())
        print(f"Error running diagnostics: {str(e)}")

def main():
    """Main function to run example requests."""
    parser = argparse.ArgumentParser(description="WebAgent Example Script")
    
    parser.add_argument("--mode", choices=["direct", "workflow", "diagnostics"], 
                        default="diagnostics", help="Mode to run in")
    parser.add_argument("--type", choices=["document_extraction", "research", "team_management", "data_analysis", "coding_assistant"],
                        help="Request type for direct mode")
    parser.add_argument("--query", help="Query for research requests")
    parser.add_argument("--document", help="Document path for extraction")
    parser.add_argument("--output", choices=["json", "text", "markdown"], default="json",
                        help="Output format")
    parser.add_argument("--max-iterations", type=int, default=3,
                        help="Maximum number of research iterations")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Maximum time in seconds to wait for completion")
    parser.add_argument("--check-network", action="store_true",
                        help="Check network connectivity in diagnostics mode")
    parser.add_argument("--check-llm", action="store_true",
                        help="Check LLM availability in diagnostics mode")
    
    args = parser.parse_args()
    
    # Run in specified mode
    if args.mode == "direct":
        if not args.type:
            logger.error("Request type is required for direct mode")
            print("Error: Request type is required for direct mode")
            print("Example: python scripts/run_webagent.py --mode direct --type research --query 'What is LangGraph?'")
            return
        
        result = direct_agent_request(
            request_type=args.type,
            query=args.query,
            document_path=args.document,
            output_format=args.output
        )
        
        if args.output == "json":
            print(json.dumps(result, indent=2))
        else:
            print(result)
        
    elif args.mode == "workflow":
        if not args.query:
            logger.error("Query is required for workflow mode")
            print("Error: Query is required for workflow mode")
            print("Example: python scripts/run_webagent.py --mode workflow --query 'What are the latest advancements in LLMs?'")
            return
        
        state = workflow_request(
            args.query,
            max_iterations=args.max_iterations,
            timeout=args.timeout
        )
        
        # Print final report
        if state.final_report:
            print("\n=== Final Report ===")
            if args.output == "json":
                print(json.dumps(state.final_report, indent=2))
            elif args.output == "text":
                print(format_result_as_text(state.final_report))
            elif args.output == "markdown":
                print(format_result_as_markdown(state.final_report))
        
        if state.error:
            print(f"\nError: {state.error}")
    
    elif args.mode == "diagnostics":
        run_diagnostics(
            check_network=args.check_network,
            check_llm=args.check_llm
        )

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        print(f"Unhandled error: {str(e)}")
        traceback.print_exc() 