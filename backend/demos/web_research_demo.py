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
from backend.app.services.llm import confirm_llm_selection
from backend.demos.research_results_display import display_research_results

@timing_decorator
async def run_web_research_demo():
    """Run the web research agent demo with various queries."""
    logger.info("Starting Web Research Agent Demo")
    
    # Ask user to confirm LLM provider and model selection
    print("\n" + "=" * 80)
    print("WEB RESEARCH AGENT DEMO")
    print("=" * 80)
    print("This demo requires an LLM for processing web search results.")
    print("The default LLM provider is TogetherAI with the Llama-3.3-70B model.")
    
    # Confirm LLM provider and model
    provider, model = confirm_llm_selection()
    
    # Get web research agent
    print("\nInitializing Web Research Agent...")
    web_agent = get_web_research_agent()
    
    # Define research queries to test
    print("\n" + "=" * 80)
    print("RESEARCH QUERIES")
    print("=" * 80)
    print("1. What are the latest developments in generative AI in 2025?")
    print("2. How have graph databases evolved in the past year?")
    print("3. What are the best practices for document processing in RAG systems?")
    print("4. Custom query")
    
    query_choice = input("\nSelect a query (1-4, default=1): ").strip()
    
    if query_choice == "2":
        query = "How have graph databases evolved in the past year?"
    elif query_choice == "3":
        query = "What are the best practices for document processing in RAG systems?"
    elif query_choice == "4":
        query = input("\nEnter your custom research query: ").strip()
    else:
        query = "What are the latest developments in generative AI in 2025?"
    
    print(f"\nSelected research query: {query}")
    
    # Ask about research parameters
    print("\n" + "=" * 80)
    print("RESEARCH PARAMETERS")
    print("=" * 80)
    print("1. Standard (balanced speed and depth)")
    print("2. Comprehensive (deeper research, more sources)")
    print("3. Quick (faster, fewer sources)")
    
    param_choice = input("\nSelect research approach (1-3, default=1): ").strip()
    
    research_priority = "balanced"
    max_sources = 3
    search_depth = "standard"
    
    if param_choice == "2":
        research_priority = "accuracy"
        max_sources = 5
        search_depth = "deep"
        print("\nSelected comprehensive research approach")
    elif param_choice == "3":
        research_priority = "speed"
        max_sources = 2
        search_depth = "quick"
        print("\nSelected quick research approach")
    else:
        print("\nSelected standard research approach")
    
    # Create a workflow state for this query
    state = WorkflowState(
        query=query,
        context={
            "research_priority": research_priority,
            "max_sources": max_sources,
            "search_depth": search_depth
        }
    )
    
    # Run web research agent
    logger.info(f"Running web research for: {query}")
    print(f"\nRunning web research with {provider} ({model}) as the LLM provider...")
    print(f"Research priority: {research_priority}, Max sources: {max_sources}")
    
    try:
        start_time = datetime.now()
        result_state = await web_agent.run(state)
        duration = (datetime.now() - start_time).total_seconds()
        
        # Print results
        print("\n" + "=" * 80)
        print(f"WEB RESEARCH RESULTS (completed in {duration:.2f} seconds)")
        print("=" * 80)
        
        if result_state.output:
            # Format the research output
            if isinstance(result_state.output, dict):
                print("\nSUMMARY:")
                print("=" * 80)
                print(result_state.output.get("summary", "No summary provided"))
                
                print("\nSOURCES:")
                print("=" * 80)
                for idx, source in enumerate(result_state.output.get("sources", [])):
                    print(f"{idx+1}. {source.get('title', 'Untitled')} - {source.get('url', 'No URL')}")
                    
                print("\nFACTUAL INFORMATION:")
                print("=" * 80)
                for idx, fact in enumerate(result_state.output.get("facts", [])):
                    print(f"{idx+1}. {fact}")
            else:
                pprint(result_state.output)
        else:
            logger.warning("No research results returned.")
            print("No research results were returned.")
            
        # Display results in GUI
        logger.info("\nDisplaying results in GUI...")
        try:
            display_research_results(
                queries=[query],
                results=[result_state.output],
                errors=[None],
                title="Web Research Results"
            )
        except Exception as e:
            logger.error(f"Error displaying results in GUI: {e}")
    except Exception as e:
        logger.error(f"Error running web research agent: {e}")
        print(f"\nERROR: {e}")
        
        # Display error in GUI
        try:
            display_research_results(
                queries=[query],
                results=[None],
                errors=[str(e)],
                title="Web Research Results"
            )
        except Exception as display_error:
            logger.error(f"Error displaying results in GUI: {display_error}")
    
    print("\nWeb Research Agent Demo completed.")

if __name__ == "__main__":
    # Run the demo
    asyncio.run(run_web_research_demo()) 