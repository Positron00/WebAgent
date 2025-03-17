#!/usr/bin/env python
"""
Coding Assistant Agent Demo
==========================

This demo demonstrates the capabilities of the Coding Assistant Agent,
which specializes in generating and explaining code for various tasks.

Usage:
    python -m backend.demos.coding_assistant_demo

Requirements:
    - Configured environment variables (check .env.example)
"""

import os
import sys
import asyncio
import logging
from datetime import datetime
import tempfile
import json
from pprint import pprint

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
logger = logging.getLogger("coding_assistant_demo")

# Import agents and services
from backend.app.agents.coding_assistant import get_coding_assistant_agent
from backend.app.models.task import WorkflowState
from backend.app.core.config import settings
from backend.app.utils.metrics import timing_decorator
from backend.app.services.llm import confirm_llm_selection

# Sample code tasks to test
CODING_TASKS = [
    {
        "title": "Simple Data Processing Function",
        "query": "Write a Python function to parse a CSV file containing user data (name, email, age) and return a list of users who are over 18 years old."
    },
    {
        "title": "Vector Database Integration",
        "query": "Create a function that connects to ChromaDB, embeds a document using OpenAI embeddings, and stores it in the vector database. Include error handling and proper documentation."
    },
    {
        "title": "Web API Endpoint",
        "query": "Write a FastAPI endpoint that accepts file uploads, validates them for size and type (PDF only), and returns a JSON response with the file's metadata."
    },
    {
        "title": "Async Task Processing",
        "query": "Implement an asynchronous task processing system using Python's asyncio that can handle multiple document processing tasks in parallel with proper error handling and status updates."
    }
]

@timing_decorator
async def run_coding_assistant_demo():
    """Run the coding assistant agent demo."""
    logger.info("Starting Coding Assistant Agent Demo")
    
    # Ask user to confirm LLM provider and model selection
    print("\n" + "=" * 80)
    print("CODING ASSISTANT AGENT DEMO")
    print("=" * 80)
    print("This demo requires an LLM for generating code and explanations.")
    print("The default LLM provider is TogetherAI with the Llama-3.3-70B model.")
    
    # Confirm LLM provider and model
    provider, model = confirm_llm_selection()
    
    # Get coding assistant agent
    print("\nInitializing Coding Assistant Agent...")
    coding_agent = get_coding_assistant_agent()
    
    # Run through each coding task
    for idx, task in enumerate(CODING_TASKS):
        logger.info(f"\n\n=== TASK {idx+1}: {task['title']} ===\n")
        
        # Create a workflow state for this task
        state = WorkflowState(
            query=task["query"],
            context={
                "language": "python",
                "code_style": "modern",
                "include_comments": True,
                "include_examples": True
            }
        )
        
        # Run coding assistant agent
        logger.info(f"Generating code for: {task['title']}")
        print(f"\nRunning code generation with {provider} ({model}) as the LLM provider...")
        print(f"Task: {task['title']}")
        try:
            start_time = datetime.now()
            result_state = await coding_agent.run(state)
            duration = (datetime.now() - start_time).total_seconds()
            
            # Print results
            logger.info(f"Code Generation Results (completed in {duration:.2f} seconds):")
            if result_state.output:
                if isinstance(result_state.output, dict):
                    # Extract code and explanation if available
                    code = result_state.output.get("code", "No code provided")
                    explanation = result_state.output.get("explanation", "No explanation provided")
                    
                    logger.info("Generated Code:")
                    print("\n```python")
                    print(code)
                    print("```\n")
                    
                    logger.info("Explanation:")
                    print(explanation)
                    
                    # Save the generated code to a file
                    output_dir = os.path.join(current_dir, "demo_outputs")
                    os.makedirs(output_dir, exist_ok=True)
                    
                    filename = f"task_{idx+1}_{task['title'].lower().replace(' ', '_')}.py"
                    filepath = os.path.join(output_dir, filename)
                    
                    with open(filepath, "w") as f:
                        f.write("# " + task["title"] + "\n")
                        f.write("# " + task["query"] + "\n\n")
                        f.write(code)
                    
                    logger.info(f"Code saved to: {filepath}")
                else:
                    pprint(result_state.output)
            else:
                logger.warning("No code generation results returned.")
                
        except Exception as e:
            logger.error(f"Error running coding assistant agent: {e}")
    
    # Test code generation with specific requirements
    logger.info("\n\n=== TESTING CODE GENERATION WITH SPECIFIC REQUIREMENTS ===\n")
    
    specific_task = {
        "query": "Create a Python class for batch processing documents with vector embedding. It should handle multiple file types, implement proper error handling, and log processing metrics.",
        "context": {
            "language": "python",
            "target_python_version": "3.12",
            "code_style": "pep8",
            "required_libraries": ["langchain", "chromadb", "openai"],
            "include_type_hints": True,
            "include_unit_tests": True
        }
    }
    
    logger.info(f"Generating code with specific requirements:")
    try:
        # Create a workflow state with these parameters
        state = WorkflowState(
            query=specific_task["query"],
            context=specific_task["context"]
        )
        
        start_time = datetime.now()
        result_state = await coding_agent.run(state)
        duration = (datetime.now() - start_time).total_seconds()
        
        # Print results
        logger.info(f"Code Generation Results (completed in {duration:.2f} seconds):")
        if result_state.output:
            if isinstance(result_state.output, dict):
                # Extract code
                code = result_state.output.get("code", "No code provided")
                
                logger.info("Generated Code:")
                print("\n```python")
                print(code)
                print("```\n")
                
                # Check if the code includes the requested features
                type_hints_included = "def " in code and ":" in code and "->" in code
                unit_tests_included = "unittest" in code or "pytest" in code or "test_" in code
                
                logger.info(f"Code includes type hints: {type_hints_included}")
                logger.info(f"Code includes unit tests: {unit_tests_included}")
                
                # Save the generated code to a file
                output_dir = os.path.join(current_dir, "demo_outputs")
                os.makedirs(output_dir, exist_ok=True)
                
                filepath = os.path.join(output_dir, "batch_processor_with_requirements.py")
                
                with open(filepath, "w") as f:
                    f.write("# Batch Document Processor with Specific Requirements\n")
                    f.write("# " + specific_task["query"] + "\n\n")
                    f.write(code)
                
                logger.info(f"Code saved to: {filepath}")
            else:
                pprint(result_state.output)
        else:
            logger.warning("No code generation results returned.")
            
    except Exception as e:
        logger.error(f"Error running coding assistant agent: {e}")
    
    logger.info("Coding Assistant Agent Demo completed.")

if __name__ == "__main__":
    # Run the demo
    asyncio.run(run_coding_assistant_demo()) 