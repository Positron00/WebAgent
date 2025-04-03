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
    
    # Define a condition to check if fixed code needs further review
    def needs_further_review(state: WorkflowState) -> str:
        """
        Check if the code still needs further review after fixing.
        
        Rules:
        1. If it's already been through 2 review cycles, stop (prevent infinite loops)
        2. If the fixed code execution failed, go back to coding_review
        3. Otherwise, end the workflow
        """
        # Check the review count
        review_count = state.context.get("review_count", 0)
        
        # Check if we already hit the maximum review cycles (2)
        if review_count >= 2:
            logger.info(f"Max review count reached ({review_count}). Ending workflow.")
            return "end"
            
        # Check if there's a coding_review report with fixed execution
        if "coding_review" in state.reports:
            review_report = state.reports["coding_review"]
            fixed_execution = review_report.get("fixed_execution", {})
            
            # Check if fixed code execution failed
            if fixed_execution and fixed_execution.get("success") is False:
                # Increment the review count
                state.context["review_count"] = review_count + 1
                logger.info(f"Fixed code execution failed. Starting review cycle {review_count + 1}.")
                
                # Update the state to use the fixed code as the new input code
                if "fixed_code" in review_report:
                    review_report["original_code"] = review_report["fixed_code"]
                    review_report["fixed_code"] = ""
                    review_report["fixed_execution"] = {}
                
                return "coding_review"
        
        # Default: end the workflow
        return "end"
    
    # Set up edges
    workflow.add_edge("__start__", "coding_assistant")
    workflow.add_edge("coding_assistant", "coding_review")
    workflow.add_conditional_edges(
        "coding_review",
        needs_further_review,
        {
            "coding_review": "coding_review",
            "end": END
        }
    )
    
    # Compile the workflow - use compile for async support
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
            final_state = await workflow.ainvoke(state)
            total_duration = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"Workflow completed in {total_duration:.2f} seconds")
            print(f"Workflow completed in {total_duration:.2f} seconds")
            
            # For debugging: log the structure of the final state
            logger.debug(f"Final state structure: {type(final_state)}")
            logger.debug(f"Final state keys: {dir(final_state)}")
            logger.debug(f"Final state content: {final_state}")
            
            # Print the structure of final_state to understand how to access it
            print("\nFinal state structure:")
            print(f"Type: {type(final_state)}")
            
            # Try to print more details about the final_state
            print("Keys in final_state dict:", list(final_state.keys()) if hasattr(final_state, 'keys') else "No keys method")
            print("Values method available:", hasattr(final_state, 'values'))
            if hasattr(final_state, 'values'):
                value_result = final_state.values()
                print(f"Values result type: {type(value_result)}")
                if hasattr(value_result, 'keys'):
                    print(f"Values keys: {list(value_result.keys())}")
            
            # Let's inspect keys/values directly
            for key in dir(final_state):
                if not key.startswith('_'):
                    try:
                        value = getattr(final_state, key, None)
                        if value is not None:
                            print(f"- {key}: {type(value)}")
                            
                            # Check if it's a dict-like object
                            if hasattr(value, 'keys'):
                                try:
                                    keys = list(value.keys())
                                    print(f"  - Keys: {keys}")
                                except:
                                    pass
                    except:
                        pass
            
            # Process the task output - extract actual code from the workflow results
            task_name = task["title"].lower().replace(' ', '_')
            filename_base = f"task_{idx+1}_{task_name}"
            
            # Prepare output directory and files
            os.makedirs(output_dir, exist_ok=True)
            
            # Extract actual generated code and review from final_state
            original_code = ""
            fixed_code = ""
            original_execution = {"success": False, "result": "No execution results available"}
            fixed_execution = {"success": False, "result": "No fixed execution results available"}
            review_iterations = []
            
            # Access the contents correctly based on the AddableValuesDict structure
            try:
                # Check if we can access the data directly through keys
                coding_assistant_data = final_state.get('coding_assistant', None)
                coding_review_data = final_state.get('coding_review', None)
                
                # Get initial code from the coding_assistant dictionary
                if coding_assistant_data:
                    if isinstance(coding_assistant_data, dict) and "code" in coding_assistant_data:
                        original_code = coding_assistant_data["code"]
                        explanation = coding_assistant_data.get("explanation", "")
                        logger.info(f"Found original code with length: {len(original_code)}")
                
                # Get review results from the coding_review dictionary
                if coding_review_data:
                    if isinstance(coding_review_data, dict):
                        fixed_code = coding_review_data.get("fixed_code", "")
                        review = coding_review_data.get("review", "")
                        improvement_explanation = coding_review_data.get("improvement_explanation", "")
                        
                        # Get execution results
                        if "original_execution" in coding_review_data:
                            original_execution = coding_review_data["original_execution"]
                        
                        if "fixed_execution" in coding_review_data:
                            fixed_execution = coding_review_data["fixed_execution"]
                        
                        # Add first review iteration
                        review_iterations.append({
                            "review": review,
                            "original_code": original_code,
                            "fixed_code": fixed_code,
                            "original_execution": original_execution,
                            "fixed_execution": fixed_execution,
                            "improvement_explanation": improvement_explanation
                        })
                
                # Print debug information about what we found
                print(f"\nFound data from workflow:")
                print(f"- Original code found: {len(original_code) > 0}")
                print(f"- Fixed code found: {len(fixed_code) > 0}")
                print(f"- Review found: {len(review) > 0 if 'review' in locals() else False}")
                
                # Debug print first 100 chars of code if found
                if len(original_code) > 0:
                    print(f"\nPreview of original code: {original_code[:100]}...")
                if len(fixed_code) > 0:
                    print(f"\nPreview of fixed code: {fixed_code[:100]}...")
                    
            except Exception as e:
                logger.error(f"Error extracting data from workflow result: {str(e)}")
                print(f"Error extracting data: {str(e)}")
            
            # If no code was found in the reports, log an error
            if not original_code:
                logger.warning(f"No code found in reports for task: {task['title']}")
                print(f"\nWARNING: No code found in reports. Using fallback examples.")
                
                # Fall back to example code for demo purposes if needed
                example_files = {
                    "Simple Data Processing with Bug": {
                        "original": "import csv\n\ndef parse_user_data(file_path):\n    adult_users = []\n    \n    try:\n        with open(file_path, 'r', newline='') as csvfile:\n            reader = csv.DictReader(csvfile)\n            for row in reader:\n                try:\n                    age = int(row.get('age', 0))\n                    if age > 18:\n                        adult_users.append(row)\n                except (ValueError, TypeError):\n                    pass\n    except Exception as e:\n        print(f\"Error reading CSV file: {e}\")\n        \n    return adult_users\n\nif __name__ == \"__main__\":\n    users = parse_user_data(\"users.csv\")\n    print(f\"Found {len(users)} adult users\")\n    for user in users:\n        print(f\"Name: {user['name']}, Email: {user['email']}, Age: {user['age']}\")",
                        "fixed": "import csv\n\ndef parse_user_data(file_path):\n    adult_users = []\n    \n    try:\n        with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:\n            reader = csv.DictReader(csvfile)\n            for row in reader:\n                try:\n                    age = int(row.get('age', 0))\n                    if age >= 18:  # Changed from > 18 to >= 18\n                        adult_users.append(row)\n                except (ValueError, TypeError) as e:\n                    print(f\"Invalid age for user {row.get('name', 'unknown')}: {e}\")\n    except FileNotFoundError:\n        print(f\"File not found: {file_path}\")\n    except Exception as e:\n        print(f\"Error reading CSV file: {e}\")\n        \n    return adult_users\n\nif __name__ == \"__main__\":\n    # Create a sample CSV file for testing\n    sample_data = [\n        {\"name\": \"Alice\", \"email\": \"alice@example.com\", \"age\": \"25\"},\n        {\"name\": \"Bob\", \"email\": \"bob@example.com\", \"age\": \"17\"},\n        {\"name\": \"Charlie\", \"email\": \"charlie@example.com\", \"age\": \"30\"},\n        {\"name\": \"David\", \"email\": \"david@example.com\", \"age\": \"invalid\"},\n    ]\n    \n    sample_file = \"sample_users.csv\"\n    try:\n        with open(sample_file, 'w', newline='', encoding='utf-8') as csvfile:\n            fieldnames = [\"name\", \"email\", \"age\"]\n            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)\n            writer.writeheader()\n            writer.writerows(sample_data)\n        \n        users = parse_user_data(sample_file)\n        print(f\"Found {len(users)} adult users\")\n        for user in users:\n            print(f\"Name: {user['name']}, Email: {user['email']}, Age: {user['age']}\")\n    finally:\n        import os\n        if os.path.exists(sample_file):\n            os.remove(sample_file)"
                    },
                    "Incomplete Exception Handling": {
                        "original": "import requests\nimport json\n\ndef fetch_data_from_api(url, timeout=5):\n    response = requests.get(url, timeout=timeout)\n    response.raise_for_status()\n    return response.json()\n\ndef process_data(url):\n    try:\n        data = fetch_data_from_api(url)\n        return data\n    except Exception as e:\n        print(f\"Error: {e}\")\n        return None",
                        "fixed": "import requests\nimport json\nimport time\n\ndef fetch_data_from_api(url, timeout=5, max_retries=3):\n    for attempt in range(max_retries):\n        try:\n            response = requests.get(url, timeout=timeout)\n            response.raise_for_status()\n            return response.json()\n        except requests.exceptions.Timeout:\n            print(f\"Timeout error on attempt {attempt+1}/{max_retries}\")\n            if attempt < max_retries - 1:\n                # Exponential backoff before retry\n                wait_time = 2 ** attempt\n                print(f\"Retrying in {wait_time} seconds...\")\n                time.sleep(wait_time)\n            else:\n                raise TimeoutError(f\"API request to {url} timed out after {max_retries} attempts\")\n        except requests.exceptions.RequestException as e:\n            raise ConnectionError(f\"API request failed: {str(e)}\")\n\ndef process_data(url):\n    try:\n        data = fetch_data_from_api(url)\n        # Process the data\n        result = {\n            \"status\": \"success\",\n            \"record_count\": len(data) if isinstance(data, list) else 1,\n            \"data\": data\n        }\n        return result\n    except TimeoutError as e:\n        print(f\"Timeout error: {e}\")\n        return {\"status\": \"error\", \"error\": \"timeout\", \"message\": str(e)}\n    except ConnectionError as e:\n        print(f\"Connection error: {e}\")\n        return {\"status\": \"error\", \"error\": \"connection\", \"message\": str(e)}\n    except Exception as e:\n        print(f\"Unexpected error: {e}\")\n        return {\"status\": \"error\", \"error\": \"unknown\", \"message\": str(e)}"
                    },
                    "Performance Issue": {
                        "original": "def find_primes(limit):\n    primes = []\n    for num in range(2, limit + 1):\n        is_prime = True\n        for i in range(2, num):\n            if num % i == 0:\n                is_prime = False\n                break\n        if is_prime:\n            primes.append(num)\n    return primes",
                        "fixed": "def find_primes(limit):\n    \"\"\"\n    Find all prime numbers up to a given limit using the Sieve of Eratosthenes algorithm.\n    \n    Args:\n        limit (int): Upper bound of the range to find primes in\n        \n    Returns:\n        list: A list of prime numbers from 2 to limit\n    \"\"\"\n    # Initialize the sieve array with True values\n    sieve = [True] * (limit + 1)\n    \n    # 0 and 1 are not prime\n    if limit >= 0:\n        sieve[0] = False\n    if limit >= 1:\n        sieve[1] = False\n    \n    # Apply the sieve algorithm\n    for i in range(2, int(limit**0.5) + 1):\n        if sieve[i]:\n            # Mark all multiples of i as non-prime\n            for j in range(i*i, limit+1, i):\n                sieve[j] = False\n    \n    # Collect the prime numbers\n    primes = [i for i in range(limit + 1) if sieve[i]]\n    return primes\n\n# Example usage\nif __name__ == \"__main__\":\n    # Test with different limits\n    limits = [10, 100, 1000]\n    for limit in limits:\n        primes = find_primes(limit)\n        print(f\"Primes up to {limit}: {len(primes)}\")\n        if limit <= 100:  # Only print full list for smaller limits\n            print(primes)"
                    },
                    "Security Vulnerability": {
                        "original": "import sqlite3\n\ndef execute_query(user_query):\n    conn = sqlite3.connect('database.db')\n    cursor = conn.cursor()\n    cursor.execute(user_query)\n    result = cursor.fetchall()\n    conn.close()\n    return result",
                        "fixed": "import sqlite3\nimport re\n\ndef execute_query(user_query, params=None):\n    \"\"\"\n    Safely execute a SQL query against a database using parameterized queries.\n    \n    Args:\n        user_query (str): The SQL query to execute\n        params (tuple, dict, optional): Parameters to use with the query\n        \n    Returns:\n        list: The query results\n        \n    Raises:\n        ValueError: If the query contains unsafe operations or is not properly parameterized\n    \"\"\"\n    # Validate the query for basic security checks\n    if not is_safe_query(user_query):\n        raise ValueError(\"Potentially unsafe query detected\")\n    \n    # Ensure params is provided for queries that need parameters\n    if needs_parameters(user_query) and params is None:\n        raise ValueError(\"Parameters required for this query but not provided\")\n    \n    try:\n        conn = sqlite3.connect('database.db')\n        cursor = conn.cursor()\n        \n        # Execute with or without parameters as needed\n        if params is not None:\n            cursor.execute(user_query, params)\n        else:\n            cursor.execute(user_query)\n            \n        result = cursor.fetchall()\n        conn.close()\n        return result\n    except sqlite3.Error as e:\n        raise DatabaseError(f\"Database error: {str(e)}\")\n    finally:\n        if 'conn' in locals() and conn:\n            conn.close()\n\ndef is_safe_query(query):\n    \"\"\"\n    Check if a query appears safe to execute.\n    \n    Args:\n        query (str): The SQL query to validate\n        \n    Returns:\n        bool: True if the query appears safe, False otherwise\n    \"\"\"\n    # Check for SQL injection patterns\n    dangerous_patterns = [\n        r'--', # SQL comment\n        r';\\s*\\w+', # Multiple statements\n        r'UNION\\s+ALL', # UNION attacks\n        r'DROP\\s+TABLE', # Table dropping\n        r'DROP\\s+DATABASE', # Database dropping\n    ]\n    \n    lower_query = query.lower()\n    \n    # Block dangerous operations\n    if any(re.search(pattern, query, re.IGNORECASE) for pattern in dangerous_patterns):\n        return False\n        \n    # Only allow SELECT statements for this example\n    if not lower_query.strip().startswith('select'):\n        return False\n        \n    return True\n\ndef needs_parameters(query):\n    \"\"\"\n    Determine if a query should be using parameters based on the presence of WHERE clauses.\n    \n    Args:\n        query (str): The SQL query to check\n        \n    Returns:\n        bool: True if the query should use parameters, False otherwise\n    \"\"\"\n    # Check if the query has a WHERE clause but no parameterization\n    if re.search(r'WHERE\\s+\\w+\\s*=\\s*[\\'\\\"\\d]', query, re.IGNORECASE):\n        return True\n    return False\n\n# Custom exception for database errors\nclass DatabaseError(Exception):\n    pass\n\n# Example usage\nif __name__ == \"__main__\":\n    try:\n        # Safe query without parameters\n        safe_query = \"SELECT * FROM users LIMIT 10\"\n        print(\"Safe query example:\")\n        print(f\"Query: {safe_query}\")\n        try:\n            print(execute_query(safe_query))\n        except Exception as e:\n            print(f\"Error: {e}\")\n            \n        # Safe query with parameters\n        param_query = \"SELECT * FROM users WHERE age > ?\"\n        params = (18,)\n        print(\"\\nParameterized query example:\")\n        print(f\"Query: {param_query}, Params: {params}\")\n        try:\n            print(execute_query(param_query, params))\n        except Exception as e:\n            print(f\"Error: {e}\")\n            \n        # Unsafe query example\n        unsafe_query = \"SELECT * FROM users WHERE username = 'admin' OR 1=1; DROP TABLE users;\"\n        print(\"\\nUnsafe query example:\")\n        print(f\"Query: {unsafe_query}\")\n        try:\n            print(execute_query(unsafe_query))\n        except ValueError as e:\n            print(f\"Correctly rejected unsafe query: {e}\")\n    except Exception as e:\n        print(f\"Unexpected error: {e}\")"
                    }
                }
                
                # Get the appropriate fallback code for this task
                task_examples = example_files.get(task["title"], {"original": "# No example for this task", "fixed": "# No fixed example for this task"})
                original_code = task_examples["original"]
                fixed_code = task_examples["fixed"]
                
                # Add warning message
                original_code = f"# WARNING: Using fallback example, actual code generation failed\n\n{original_code}"
                fixed_code = f"# WARNING: Using fallback example, actual code review failed\n\n{fixed_code}"
                
                # Simulate execution results if using fallback
                if len(review_iterations) == 0:
                    review_iterations.append({
                        "review": "FALLBACK EXAMPLE: Actual review failed",
                        "original_code": original_code,
                        "fixed_code": fixed_code,
                        "original_execution": {"success": False, "result": "No actual execution (fallback example)"},
                        "fixed_execution": {"success": True, "result": "No actual execution (fallback example)"}
                    })
            
            # Save the code to files
            with open(os.path.join(output_dir, f"{filename_base}_original.py"), "w") as f:
                f.write(f"# {task['title']}\n")
                f.write(f"# Query: {task['query']}\n\n")
                f.write(original_code)
            
            with open(os.path.join(output_dir, f"{filename_base}_fixed.py"), "w") as f:
                f.write(f"# {task['title']} (Fixed Version)\n")
                f.write(f"# Query: {task['query']}\n\n") 
                f.write(fixed_code)
                
            # Display the results
            print("\n--- CODE GENERATION RESULTS ---")
            print("\nGenerated Code:")
            print("```python")
            print(original_code[:500] + ("..." if len(original_code) > 500 else ""))
            print("```\n")
            
            # Display the code review results
            print("\n--- CODE REVIEW RESULTS ---")
            
            # First review cycle
            if review_iterations:
                print("\nInitial Review:")
                print(review_iterations[0]["review"])
                print()
                
                # Display execution results
                print(f"\nOriginal Code Execution: {'SUCCESS' if original_execution.get('success', False) else 'FAILED'}")
                print("```")
                print(original_execution.get("result", "No result available"))
                print("```")
                
                # Display fixes
                print("\nFixed Code:")
                print("```python") 
                print(fixed_code[:500] + ("..." if len(fixed_code) > 500 else ""))
                print("```\n")
                
                # Get improvement explanation from the first review cycle
                improvement_explanation = review_iterations[0].get("improvement_explanation", "Fixed implementation addresses issues in the original code")
                print("Improvement Explanation:")
                print(improvement_explanation)
                print()
                
                # Display fixed code execution results
                fixed_execution_success = fixed_execution.get("success", False) if fixed_execution else False
                fixed_execution_result = fixed_execution.get("result", "No result available") if fixed_execution else "No result available"
                
                print(f"Fixed Code Execution: {'SUCCESS' if fixed_execution_success else 'FAILED'}")
                print("```")
                print(fixed_execution_result)
                print("```")
                
                # Display additional review cycles if any
                if len(review_iterations) > 1:
                    for i, review_iter in enumerate(review_iterations[1:], 2):
                        print(f"\n--- REVIEW CYCLE {i} ---")
                        print(f"\nReview #{i}:")
                        print(review_iter["review"])
                        print()
                        
                        # Second cycle execution results
                        print(f"\nIteration {i} Execution: {'SUCCESS' if review_iter['original_execution'].get('success', False) else 'FAILED'}")
                        print("```")
                        print(review_iter["original_execution"].get("result", "No result available"))
                        print("```")
                        
                        # Display final optimized execution
                        print(f"\nFinal Optimized Execution: {'SUCCESS' if review_iter['fixed_execution'].get('success', False) else 'FAILED'}")
                        print("```")
                        print(review_iter["fixed_execution"].get("result", "No result available"))
                        print("```")
            
            # Store results for display
            all_results.append({
                "task": task["title"],
                "query": task["query"],
                "code": original_code,
                "code_explanation": "Generated code based on your query",
                "review": review_iterations[0]["review"] if review_iterations else "No review available",
                "execution_analysis": "Analysis of execution results",
                "original_execution": original_execution,
                "fixed_code": fixed_code,
                "improvement_explanation": review_iterations[0].get("improvement_explanation", "") if review_iterations else "",
                "fixed_execution": fixed_execution,
                "review_iterations": review_iterations,
                "workflow_steps": [
                    {
                        "name": "Code Generation",
                        "code": original_code,
                        "explanation": "Generated code based on your query"
                    },
                    {
                        "name": "Code Review",
                        "review": review_iterations[0]["review"] if review_iterations else "No review available",
                        "execution_analysis": "Analysis of execution results",
                        "original_execution": original_execution,
                        "fixed_code": fixed_code,
                        "improvement_explanation": review_iterations[0].get("improvement_explanation", "") if review_iterations else "",
                        "fixed_execution": fixed_execution
                    }
                ]
            })
            all_errors.append(None)
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
        # Convert all_results to a format compatible with result display
        display_results = []
        for result in all_results:
            if result is None:
                display_results.append(None)
                continue
                
            # Create a format that matches what format_result expects for coding output
            execution_result_text = f"Original Execution: {'SUCCESS' if result['original_execution']['success'] else 'FAILED'}\n{result['original_execution']['result']}\n\nFixed Execution: {'SUCCESS' if result['fixed_execution']['success'] else 'FAILED'}\n{result['fixed_execution']['result']}"
            
            # If there were multiple review iterations, include them in the results
            if 'review_iterations' in result and len(result['review_iterations']) > 1:
                execution_result_text += "\n\n--- ADDITIONAL REVIEW CYCLES ---\n"
                for i, review_iter in enumerate(result['review_iterations'][1:], 2):
                    execution_result_text += f"\nReview Cycle {i}:\n"
                    execution_result_text += f"Execution: {'SUCCESS' if review_iter['original_execution']['success'] else 'FAILED'}\n"
                    execution_result_text += f"{review_iter['original_execution']['result']}\n\n"
                    execution_result_text += f"Final Optimized Execution: {'SUCCESS' if review_iter['fixed_execution']['success'] else 'FAILED'}\n"
                    execution_result_text += f"{review_iter['fixed_execution']['result']}"
            
            display_result = {
                "code": result["code"],
                "explanation": result["code_explanation"],
                "execution_result": execution_result_text,
                "review": f"Review:\n{result['review']}\n\nImprovement Explanation:\n{result['improvement_explanation']}"
            }
            display_results.append(display_result)
            
        display_research_results(
            queries=[task["query"] for task in CODING_TASKS],
            results=display_results,
            errors=all_errors,
            title="Integrated Coding Assistant and Review Results"
        )
    except Exception as e:
        logger.error(f"Error displaying results: {e}")
        traceback.print_exc()
    
    logger.info("Integrated Coding Assistant and Review Demo completed")
    print("\nDemo completed. Generated code files saved to:", output_dir)

if __name__ == "__main__":
    # Run the demo
    asyncio.run(run_integrated_coding_demo()) 