#!/usr/bin/env python
"""
Integrated Coding Assistant and Review Demo
===========================================

This demo showcases the integrated workflow of code generation, review, and debugging
within the LangGraph framework using:
1. Coding Assistant Agent - Generates Python code based on user queries
2. Coding Review Agent - Reviews, executes, debugs and fixes the generated code

Usage:
    python -m backend.demos.integrated_coding_demo

Requirements:
    - Configured environment variables (check .env.example)
"""

import os
import sys
import asyncio
import logging
import traceback
from datetime import datetime
import tempfile
import json
from pprint import pprint
import subprocess
import re

# Setup path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("integrated_coding_demo")

# Import required modules
from backend.app.agents.coding_assistant import get_coding_assistant_agent
from backend.app.agents.coding_review import get_coding_review_agent
from backend.app.models.task import WorkflowState
from backend.app.core.config import settings
from backend.app.utils.metrics import timing_decorator
from backend.app.services.llm import confirm_llm_selection
from backend.demos.research_results_display import display_research_results
from langgraph.graph import StateGraph, END

# Sample code tasks to test the integrated workflow
CODING_TASKS = [
    {
        "title": "Simple Data Processing with Bug",
        "query": "Write a Python function to parse a CSV file containing user data (name, email, age) and return a list of users who are over 18 years old. The function should handle errors gracefully."
    },
    {
        "title": "Incomplete Exception Handling",
        "query": "Create a function that connects to an API, fetches data, and processes it. Include error handling for API timeouts."
    },
    {
        "title": "Performance Issue",
        "query": "Write a function that finds all prime numbers up to a given limit using the Sieve of Eratosthenes algorithm."
    },
    {
        "title": "Security Vulnerability",
        "query": "Create a function that takes a SQL query parameter from user input and executes it against a database."
    }
]

def build_coding_workflow():
    """
    Build a LangGraph workflow for code generation, review, and debugging.
    
    Returns:
        StateGraph: The compiled workflow graph
    """
    logger.info("Building coding workflow graph")
    
    # Get the agent instances
    coding_assistant = get_coding_assistant_agent()
    coding_review = get_coding_review_agent()
    
    # Create the graph
    workflow = StateGraph(WorkflowState)
    
    # Add nodes to the graph
    workflow.add_node("coding_assistant", coding_assistant)
    workflow.add_node("coding_review", coding_review)
    
    # Set up edges
    workflow.add_edge("__start__", "coding_assistant")
    workflow.add_edge("coding_assistant", "coding_review")
    workflow.add_edge("coding_review", END)
    
    # Compile the workflow
    compiled_workflow = workflow.compile()
    logger.info("Compiled coding workflow successfully")
    
    return compiled_workflow

@timing_decorator
async def run_integrated_coding_demo():
    """Run the integrated coding assistant and review demo."""
    logger.info("Starting Integrated Coding Assistant and Review Demo")
    
    # Initialize lists to store results for display
    all_results = []
    all_errors = []
    
    print("\n" + "=" * 80)
    print("INTEGRATED CODING ASSISTANT AND REVIEW DEMO")
    print("=" * 80)
    print("This demo shows the complete code generation, review, and debugging workflow")
    print("using a LangGraph-based architecture with OpenAI GPT-4o-mini")
    
    # Set model directly without interactive prompt
    provider = "openai" 
    model = "gpt-4o-mini"
    
    # Set environment variables for the selected provider and model
    os.environ["CONFIRM_LLM_SELECTION"] = "true"
    os.environ["CONFIRMED_LLM_PROVIDER"] = provider
    os.environ["CONFIRMED_LLM_MODEL"] = model
    
    # Build the LangGraph workflow
    print("\nBuilding LangGraph workflow...")
    workflow = build_coding_workflow()
    
    # Create output directory for saving generated code
    output_dir = os.path.join(current_dir, "demo_outputs")
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each coding task
    for idx, task in enumerate(CODING_TASKS):
        logger.info(f"\n\n=== TASK {idx+1}: {task['title']} ===\n")
        print(f"\n{'=' * 60}")
        print(f"TASK {idx+1}: {task['title']}")
        print(f"{'=' * 60}")
        print(f"Query: {task['query']}")
        
        try:
            # Create initial workflow state
            state = WorkflowState(
                query=task["query"],
                context={
                    "language": "python",
                    "code_style": "modern",
                    "include_comments": True,
                    "include_examples": True
                }
            )
            
            # Execute the workflow using LangGraph
            print("\nExecuting LangGraph workflow...")
            start_time = datetime.now()
            final_state = workflow.invoke(state)
            total_duration = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"Workflow completed in {total_duration:.2f} seconds")
            print(f"Workflow completed in {total_duration:.2f} seconds")
            
            # Extract the results from both agents
            if "coding_assistant" in final_state.reports and "coding_review" in final_state.reports:
                coding_report = final_state.reports["coding_assistant"]
                review_report = final_state.reports["coding_review"]
                
                # Get the original code
                original_code = coding_report.get("code", "")
                
                # Display the code generation results
                print("\n--- CODE GENERATION RESULTS ---")
                print("\nGenerated Code:")
                print("```python")
                print(original_code)
                print("```\n")
                
                if "explanation" in coding_report:
                    print("Code Explanation:")
                    print(coding_report["explanation"])
                    print()
                
                # Display the code review results
                print("\n--- CODE REVIEW RESULTS ---")
                
                if "review" in review_report:
                    print("\nReview:")
                    print(review_report["review"])
                    print()
                
                # Display execution results
                original_execution = review_report.get("original_execution", {})
                original_success = original_execution.get("success", False)
                original_result = original_execution.get("result", "")
                
                print(f"\nOriginal Code Execution: {'SUCCESS' if original_success else 'FAILED'}")
                print("```")
                print(original_result)
                print("```")
                
                # Display fixes if any
                fixed_code = review_report.get("fixed_code", "")
                if fixed_code:
                    print("\nFixed Code:")
                    print("```python")
                    print(fixed_code)
                    print("```\n")
                    
                    if "improvement_explanation" in review_report:
                        print("Improvement Explanation:")
                        print(review_report["improvement_explanation"])
                        print()
                    
                    # Display fixed code execution results
                    fixed_execution = review_report.get("fixed_execution", {})
                    fixed_success = fixed_execution.get("success", False)
                    fixed_result = fixed_execution.get("result", "")
                    
                    print(f"Fixed Code Execution: {'SUCCESS' if fixed_success else 'FAILED'}")
                    print("```")
                    print(fixed_result)
                    print("```")
                
                # Save the results to files
                filename_base = f"task_{idx+1}_{task['title'].lower().replace(' ', '_')}"
                
                # Save original code
                with open(os.path.join(output_dir, f"{filename_base}_original.py"), "w") as f:
                    f.write(f"# {task['title']}\n")
                    f.write(f"# Query: {task['query']}\n\n")
                    f.write(review_report.get("original_code", original_code))
                
                # Save fixed code if available
                if fixed_code:
                    with open(os.path.join(output_dir, f"{filename_base}_fixed.py"), "w") as f:
                        f.write(f"# {task['title']} (Fixed Version)\n")
                        f.write(f"# Query: {task['query']}\n\n")
                        f.write(fixed_code)
                
                # Store results for display
                all_results.append({
                    "task": task["title"],
                    "query": task["query"],
                    "workflow_steps": [
                        {
                            "name": "Code Generation",
                            "code": original_code,
                            "explanation": coding_report.get("explanation", "")
                        },
                        {
                            "name": "Code Review",
                            "review": review_report.get("review", ""),
                            "execution_analysis": review_report.get("execution_analysis", ""),
                            "original_execution": original_execution,
                            "fixed_code": fixed_code,
                            "improvement_explanation": review_report.get("improvement_explanation", ""),
                            "fixed_execution": fixed_execution if fixed_code else None
                        }
                    ]
                })
                all_errors.append(None)
            else:
                error_msg = "Workflow did not complete successfully"
                logger.error(error_msg)
                print(f"Error: {error_msg}")
                all_results.append(None)
                all_errors.append(error_msg)
                
        except Exception as e:
            error_msg = f"Error running workflow: {str(e)}"
            logger.error(error_msg)
            print(f"Error: {error_msg}")
            traceback.print_exc()
            all_results.append(None)
            all_errors.append(error_msg)
    
    # Display summary of all results
    logger.info("\nDisplaying summary of all results...")
    try:
        display_research_results(
            queries=[task["query"] for task in CODING_TASKS],
            results=all_results,
            errors=all_errors,
            title="Integrated Coding Assistant and Review Results"
        )
    except Exception as e:
        logger.error(f"Error displaying results: {e}")
    
    logger.info("Integrated Coding Assistant and Review Demo completed")
    print("\nDemo completed. Generated code files saved to:", output_dir)

if __name__ == "__main__":
    # Run the demo
    asyncio.run(run_integrated_coding_demo()) 