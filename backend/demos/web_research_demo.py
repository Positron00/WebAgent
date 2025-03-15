#!/usr/bin/env python
"""
Web Research Agent Demo
======================

This demo demonstrates the capabilities of the Web Research Agent,
which specializes in searching the web for information.

Usage:
    python -m backend.demos.web_research_demo

Requirements:
    - Configured environment variables (check .env.example)
    - API keys for web search providers (Tavily or alternative)
"""

import os
import sys
import asyncio
import logging
from datetime import datetime
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
logger = logging.getLogger("web_research_demo")

# Import agents and services
from backend.app.agents.web_research import get_web_research_agent
from backend.app.models.task import WorkflowState
from backend.app.core.config import settings
from backend.app.utils.metrics import timing_decorator

@timing_decorator
async def run_web_research_demo():
    """Run the web research agent demo with various queries."""
    logger.info("Starting Web Research Agent Demo")
    
    # Get web research agent
    web_agent = get_web_research_agent()
    
    # Define research queries to test
    research_queries = [
        "What are the latest developments in generative AI in 2025?",
        "How have graph databases evolved in the past year?",
        "What are the best practices for document processing in RAG systems?",
    ]
    
    for idx, query in enumerate(research_queries):
        logger.info(f"\n\n=== QUERY {idx+1}: {query} ===\n")
        
        # Create a workflow state for this query
        state = WorkflowState(
            query=query,
            context={
                "research_priority": "accuracy",
                "max_sources": 3,
                "search_depth": "standard"
            }
        )
        
        # Run web research agent
        logger.info(f"Running web research for: {query}")
        try:
            start_time = datetime.now()
            result_state = await web_agent.run(state)
            duration = (datetime.now() - start_time).total_seconds()
            
            # Print results
            logger.info(f"Web Research Results (completed in {duration:.2f} seconds):")
            if result_state.output:
                # Format the research output
                if isinstance(result_state.output, dict):
                    logger.info("Summary:")
                    print(result_state.output.get("summary", "No summary provided"))
                    
                    logger.info("\nSources:")
                    for idx, source in enumerate(result_state.output.get("sources", [])):
                        print(f"{idx+1}. {source.get('title', 'Untitled')} - {source.get('url', 'No URL')}")
                        
                    logger.info("\nFactual Information:")
                    for idx, fact in enumerate(result_state.output.get("facts", [])):
                        print(f"{idx+1}. {fact}")
                else:
                    pprint(result_state.output)
            else:
                logger.warning("No research results returned.")
                
        except Exception as e:
            logger.error(f"Error running web research agent: {e}")
    
    # Test research with specific parameters
    logger.info("\n\n=== TESTING RESEARCH WITH PARAMETERS ===\n")
    
    parameter_tests = [
        {
            "query": "Latest advancements in vector databases for RAG systems",
            "params": {
                "research_priority": "speed",
                "max_sources": 2,
                "search_depth": "quick"
            }
        },
        {
            "query": "Best practices for fine-tuning large language models in 2025",
            "params": {
                "research_priority": "comprehensiveness",
                "max_sources": 5,
                "search_depth": "deep"
            }
        }
    ]
    
    for idx, test in enumerate(parameter_tests):
        logger.info(f"\n=== PARAMETER TEST {idx+1}: {test['query']} ===\n")
        
        # Create a workflow state with these parameters
        state = WorkflowState(
            query=test["query"],
            context=test["params"]
        )
        
        # Run web research agent
        logger.info(f"Running web research with parameters: {test['params']}")
        try:
            start_time = datetime.now()
            result_state = await web_agent.run(state)
            duration = (datetime.now() - start_time).total_seconds()
            
            # Print results
            logger.info(f"Web Research Results (completed in {duration:.2f} seconds):")
            if result_state.output:
                # Check if the response includes source count info
                source_count = len(result_state.output.get("sources", []))
                logger.info(f"Retrieved {source_count} sources")
                
                print("\nSummary:")
                print(result_state.output.get("summary", "No summary provided"))
            else:
                logger.warning("No research results returned.")
                
        except Exception as e:
            logger.error(f"Error running web research agent: {e}")
    
    logger.info("Web Research Agent Demo completed.")

if __name__ == "__main__":
    # Run the demo
    asyncio.run(run_web_research_demo()) 