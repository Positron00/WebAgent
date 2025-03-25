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
from backend.demos.research_results_display import display_research_results

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
    
    # Initialize lists to store results for GUI display
    all_queries = []
    all_results = []
    all_errors = []
    
    # Ask user to confirm LLM provider and model selection
    print("\n" + "=" * 80)
    print("CODING ASSISTANT AGENT DEMO")
    print("=" * 80)
    print("This demo requires an LLM for generating code and explanations.")
    print("Using OpenAI GPT-4o mini with code interpreter")
    
    # Set model directly without interactive prompt
    provider = "openai" 
    model = "gpt-4o-mini"
    
    # Set environment variables for the selected provider and model
    os.environ["CONFIRM_LLM_SELECTION"] = "true"
    os.environ["CONFIRMED_LLM_PROVIDER"] = provider
    os.environ["CONFIRMED_LLM_MODEL"] = model
    
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
            result_state = await coding_agent(state)
            duration = (datetime.now() - start_time).total_seconds()
        except Exception as e:
            logger.error(f"Error running coding assistant agent: {e}")
            # Store error for display
            all_queries.append(f"{task['title']}: {task['query']}")
            all_results.append(None)
            all_errors.append(str(e))
            continue
        
        try:
            # Store results for display
            all_queries.append(f"{task['title']}: {task['query']}")
            all_results.append(result_state.output)
            all_errors.append(None)
            
            # Print results
            logger.info(f"Code Generation Results (completed in {duration:.2f} seconds):")
            
            # Debug - Print available keys
            print(f"\nWorkflowState has these attributes: {dir(result_state)}")
            print(f"Reports keys: {result_state.reports.keys()}")
            if "coding_assistant" in result_state.reports:
                coding_report = result_state.reports["coding_assistant"]
                print(f"Found coding_assistant report: {type(coding_report)}")
                if isinstance(coding_report, dict):
                    print(f"Report keys: {coding_report.keys()}")
                    if "code" in coding_report:
                        print(f"Found code with length: {len(str(coding_report['code']))}")
            
            # Extract code from the reports
            code = None
            explanation = None
            
            if "coding_assistant" in result_state.reports:
                coding_report = result_state.reports["coding_assistant"]
                if isinstance(coding_report, dict):
                    code = coding_report.get("code", None)
                    explanation = coding_report.get("explanation", None)
                    
                    # Update the results list with the actual code for display
                    all_results[-1] = {
                        "code": code,
                        "explanation": explanation
                    }
            
            # Clean up and format the code before displaying
            if code:
                # Print raw code for debugging
                print(f"\nRaw code data received ({len(code)} chars):")
                print(f"Code starts with: {code[:50]}")
                print(f"Code ends with: {code[-50:] if len(code) > 50 else code}")
                
                # Extract code blocks from the result
                import re
                code_blocks = re.findall(r'```(?:python)?\s*(.*?)```', code, re.DOTALL)
                if code_blocks:
                    # Use the longest code block
                    code = max(code_blocks, key=len).strip()
                    print(f"Extracted code block with length: {len(code)}")
                elif code.startswith("```python") and not code.endswith("```"):
                    # Handle case where closing backticks are missing
                    code = code.replace("```python", "", 1).strip()
                    print(f"Extracted code block from truncated block: {len(code)}")
                elif "def " in code or "import " in code or "class " in code:
                    # There's already code without the backticks
                    print(f"Using code that already contains Python syntax")
                else:
                    # Try to extract if the result contains JSON-like structure
                    code_match = re.search(r'"code"\s*:\s*"(.+?)(?:"|$)', code, re.DOTALL)
                    if code_match:
                        extracted = code_match.group(1).replace("\\n", "\n").replace('\\"', '"')
                        if len(extracted) > 50:  # Only use if reasonably long
                            code = extracted
                            print(f"Extracted code from JSON structure: {len(code)}")
            
            # If code is still not valid Python or empty, try to extract from raw output
            if not code or not any(marker in code for marker in ["def ", "import ", "class "]):
                print("Code still missing or invalid, checking result_state.output...")
                if result_state.output and isinstance(result_state.output, str):
                    raw_output = result_state.output
                    # Look for any Python code in the raw output
                    code_blocks = re.findall(r'```(?:python)?\s*(.*?)```', raw_output, re.DOTALL)
                    if code_blocks:
                        code = max(code_blocks, key=len).strip()
                        print(f"Found code in raw output: {len(code)} chars")
                    elif any(marker in raw_output for marker in ["def ", "import ", "class "]):
                        # Extract lines that look like Python code
                        code_lines = []
                        for line in raw_output.split('\n'):
                            if any(marker in line for marker in ["def ", "import ", "class ", "    ", "return ", "if "]):
                                code_lines.append(line)
                        
                        if code_lines:
                            code = '\n'.join(code_lines)
                            print(f"Extracted {len(code_lines)} lines of Python code from raw output")
            
            logger.info("Generated Code:")
            print("\n```python")
            print(code)
            print("```\n")
            
            if explanation:
                logger.info("Explanation:")
                print(explanation)
            
            # Save the generated code to a file
            # Use the absolute path from the script location
            output_dir = os.path.join(current_dir, "demo_outputs")
            os.makedirs(output_dir, exist_ok=True)
            
            filename = f"task_{idx+1}_{task['title'].lower().replace(' ', '_')}.py"
            filepath = os.path.join(output_dir, filename)
            
            try:
                with open(filepath, "w") as f:
                    f.write("# " + task["title"] + "\n")
                    f.write("# " + task["query"] + "\n\n")
                    
                    # Clean up code before saving to file
                    if isinstance(code, str):
                        # Check if code still has backticks
                        code_to_save = code
                        
                        if code_to_save.startswith("```") and code_to_save.endswith("```"):
                            code_to_save = code_to_save[3:-3].strip()
                            
                            # Check for language identifier
                            if code_to_save.startswith("python"):
                                code_to_save = code_to_save[6:].strip()
                        
                        # Fix common escaped characters
                        code_to_save = code_to_save.replace("\\n", "\n").replace('\\"', '"').replace("\\t", "\t")
                        
                        # Handle JSON format with "code" key
                        if '"code":' in code_to_save:
                            import re
                            code_match = re.search(r'"code":\s*"(.*?)(?:"|$)', code_to_save, re.DOTALL)
                            if code_match:
                                code_to_save = code_match.group(1)
                                # Further clean up any escaped characters
                                code_to_save = code_to_save.replace("\\n", "\n").replace('\\"', '"')
                        
                        f.write(code_to_save)
                    else:
                        f.write(str(code))
                
                logger.info(f"Code saved to: {filepath}")
                print(f"Code saved to: {filepath}")
            except Exception as save_error:
                logger.error(f"Error saving code to file: {save_error}")
                print(f"Error saving code to file: {save_error}")
            
            # Check if no code was found
            if code is None or not code:
                logger.warning("No code found in output")
                print("No code found in output")
        except Exception as e:
            logger.error(f"Error running coding assistant agent: {e}")
            # Store error for display
            all_queries.append(f"{task['title']}: {task['query']}")
            all_results.append(None)
            all_errors.append(str(e))
    
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
        result_state = await coding_agent(state)
        duration = (datetime.now() - start_time).total_seconds()
    except Exception as e:
        logger.error(f"Error running coding assistant agent: {e}")
        # Store error for display
        all_queries.append(f"{specific_task['query']}: {specific_task['query']}")
        all_results.append(None)
        all_errors.append(str(e))
        
    try:
        # Store results for display
        all_queries.append(f"{specific_task['query']}: {specific_task['query']}")
        all_results.append(result_state.output)
        all_errors.append(None)
        
        # Print results
        logger.info(f"Code Generation Results (completed in {duration:.2f} seconds):")
        
        # Debug - Print available keys
        print(f"\nWorkflowState has these attributes: {dir(result_state)}")
        print(f"Reports keys: {result_state.reports.keys()}")
        if "coding_assistant" in result_state.reports:
            coding_report = result_state.reports["coding_assistant"]
            print(f"Found coding_assistant report: {type(coding_report)}")
            if isinstance(coding_report, dict):
                print(f"Report keys: {coding_report.keys()}")
                if "code" in coding_report:
                    print(f"Found code with length: {len(str(coding_report['code']))}")
        
        # Extract code from the reports
        code = None
        explanation = None
        
        if "coding_assistant" in result_state.reports:
            coding_report = result_state.reports["coding_assistant"]
            if isinstance(coding_report, dict):
                code = coding_report.get("code", None)
                explanation = coding_report.get("explanation", None)
                
                # Update the results list with the actual code for display
                all_results[-1] = {
                    "code": code,
                    "explanation": explanation
                }
        
        # Clean up and format the code before displaying
        if code:
            # Print raw code for debugging
            print(f"\nRaw code data received ({len(code)} chars):")
            print(f"Code starts with: {code[:50]}")
            print(f"Code ends with: {code[-50:] if len(code) > 50 else code}")
            
            # Extract code blocks from the result
            import re
            code_blocks = re.findall(r'```(?:python)?\s*(.*?)```', code, re.DOTALL)
            if code_blocks:
                # Use the longest code block
                code = max(code_blocks, key=len).strip()
                print(f"Extracted code block with length: {len(code)}")
            elif code.startswith("```python") and not code.endswith("```"):
                # Handle case where closing backticks are missing
                code = code.replace("```python", "", 1).strip()
                print(f"Extracted code block from truncated block: {len(code)}")
            elif "def " in code or "import " in code or "class " in code:
                # There's already code without the backticks
                print(f"Using code that already contains Python syntax")
            else:
                # Try to extract if the result contains JSON-like structure
                code_match = re.search(r'"code"\s*:\s*"(.+?)(?:"|$)', code, re.DOTALL)
                if code_match:
                    extracted = code_match.group(1).replace("\\n", "\n").replace('\\"', '"')
                    if len(extracted) > 50:  # Only use if reasonably long
                        code = extracted
                        print(f"Extracted code from JSON structure: {len(code)}")
            
            # If code is still not valid Python or empty, try to extract from raw output
            if not code or not any(marker in code for marker in ["def ", "import ", "class "]):
                print("Code still missing or invalid, checking result_state.output...")
                if result_state.output and isinstance(result_state.output, str):
                    raw_output = result_state.output
                    # Look for any Python code in the raw output
                    code_blocks = re.findall(r'```(?:python)?\s*(.*?)```', raw_output, re.DOTALL)
                    if code_blocks:
                        code = max(code_blocks, key=len).strip()
                        print(f"Found code in raw output: {len(code)} chars")
                    elif any(marker in raw_output for marker in ["def ", "import ", "class "]):
                        # Extract lines that look like Python code
                        code_lines = []
                        for line in raw_output.split('\n'):
                            if any(marker in line for marker in ["def ", "import ", "class ", "    ", "return ", "if "]):
                                code_lines.append(line)
                        
                        if code_lines:
                            code = '\n'.join(code_lines)
                            print(f"Extracted {len(code_lines)} lines of Python code from raw output")
            
            logger.info("Generated Code:")
            print("\n```python")
            print(code)
            print("```\n")
            
            if explanation:
                logger.info("Explanation:")
                print(explanation)
            
            # Save the generated code to a file
            # Use the absolute path from the script location
            output_dir = os.path.join(current_dir, "demo_outputs")
            os.makedirs(output_dir, exist_ok=True)
            
            filepath = os.path.join(output_dir, "batch_processor_with_requirements.py")
            
            try:
                with open(filepath, "w") as f:
                    f.write("# Batch Document Processor with Specific Requirements\n")
                    f.write("# " + specific_task["query"] + "\n\n")
                    
                    # Clean up code before saving to file
                    if isinstance(code, str):
                        # Check if code still has backticks
                        code_to_save = code
                        
                        if code_to_save.startswith("```") and code_to_save.endswith("```"):
                            code_to_save = code_to_save[3:-3].strip()
                            
                            # Check for language identifier
                            if code_to_save.startswith("python"):
                                code_to_save = code_to_save[6:].strip()
                        
                        # Fix common escaped characters
                        code_to_save = code_to_save.replace("\\n", "\n").replace('\\"', '"').replace("\\t", "\t")
                        
                        # Handle JSON format with "code" key
                        if '"code":' in code_to_save:
                            import re
                            code_match = re.search(r'"code":\s*"(.*?)(?:"|$)', code_to_save, re.DOTALL)
                            if code_match:
                                code_to_save = code_match.group(1)
                                # Further clean up any escaped characters
                                code_to_save = code_to_save.replace("\\n", "\n").replace('\\"', '"')
                        
                        f.write(code_to_save)
                    else:
                        f.write(str(code))
                
                logger.info(f"Code saved to: {filepath}")
                print(f"Code saved to: {filepath}")
            except Exception as save_error:
                logger.error(f"Error saving code to file: {save_error}")
                print(f"Error saving code to file: {save_error}")
            
            # Check if no code was found
            if code is None or not code:
                logger.warning("No code found in output")
                print("No code found in output")
    except Exception as e:
        logger.error(f"Error running coding assistant agent: {e}")
        # Store error for display
        all_queries.append(f"{specific_task['query']}: {specific_task['query']}")
        all_results.append(None)
        all_errors.append(str(e))
    
    # After all tasks are complete, display results in GUI
    logger.info("\nDisplaying results in GUI...")
    try:
        display_research_results(
            queries=all_queries,
            results=all_results,
            errors=all_errors,
            title="Coding Assistant Results"
        )
    except Exception as e:
        logger.error(f"Error displaying results in GUI: {e}")
    
    logger.info("Coding Assistant Agent Demo completed.")

if __name__ == "__main__":
    # Run the demo
    asyncio.run(run_coding_assistant_demo()) 
