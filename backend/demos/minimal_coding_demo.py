#!/usr/bin/env python
"""
Minimal Coding Assistant Demo
=============================

This is a simplified version of the coding assistant demo that bypasses
most of the infrastructure to directly use the LLM API.
"""

import os
import sys
import asyncio
import logging
from datetime import datetime
import json
from pprint import pprint
import requests

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
logger = logging.getLogger("minimal_coding_demo")

# Import settings to get API key
from backend.app.core.config import settings

from langchain_openai import ChatOpenAI
from langchain_together import Together
from langchain.schema.messages import HumanMessage

# Sample code task to test
SAMPLE_TASK = "Write a Python function to parse a CSV file containing user data (name, email, age) and return a list of users who are over 18 years old."

def run_demo():
    """Run a minimal demo that directly uses the LLM API."""
    logger.info("Starting Minimal Coding Assistant Demo")
    
    print("\n" + "=" * 80)
    print("MINIMAL CODING ASSISTANT DEMO")
    print("=" * 80)
    print("This demo uses Together AI with the Llama-3.3-70B model")
    print("=" * 80)
    
    print("\nInitializing LLM...")
    
    # Initialize Together AI client directly
    try:
        # Get API key from settings
        together_api_key = settings.TOGETHER_API_KEY
        
        if not together_api_key:
            raise ValueError("TOGETHER_API_KEY is not set in config settings")
            
        # Format a simple prompt
        prompt = f"""You are a coding assistant expert in Python.

Task: {SAMPLE_TASK}

Write clear, efficient, and well-commented Python code to complete this task.
Provide the code implementation without any explanation before or after the code."""
        
        # Create API request
        url = "https://api.together.xyz/v1/completions"
        headers = {
            "Authorization": f"Bearer {together_api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            "prompt": prompt,
            "max_tokens": 2000,
            "temperature": 0.7,
        }
        
        print("Generating code...")
        start_time = datetime.now()
        
        # Make the API request
        response = requests.post(url, headers=headers, json=data)
        response_data = response.json()
        
        if "choices" in response_data and len(response_data["choices"]) > 0:
            result = response_data["choices"][0]["text"]
        else:
            raise ValueError(f"Unexpected API response: {response_data}")
            
        duration = (datetime.now() - start_time).total_seconds()
        
        # Print the result
        print(f"\nGenerated Code (in {duration:.2f} seconds):")
        print("\n```python")
        print(result)
        print("```\n")
        
        # Save the output
        output_dir = os.path.join(current_dir, "demo_outputs")
        os.makedirs(output_dir, exist_ok=True)
        
        filepath = os.path.join(output_dir, "minimal_demo_output.py")
        
        with open(filepath, "w") as f:
            f.write("# Sample Code Task\n")
            f.write(f"# {SAMPLE_TASK}\n\n")
            f.write(result)
        
        logger.info(f"Code saved to: {filepath}")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
    
    logger.info("Minimal Coding Assistant Demo completed.")

if __name__ == "__main__":
    # Run the demo
    run_demo() 