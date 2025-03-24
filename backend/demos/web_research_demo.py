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

Environment Variables:
---------------------
This demo uses the following environment variables:
- TAVILY_API_KEY: API key for Tavily web search
- LLM_PROVIDER: Which LLM provider to use (together, openai, groq, etc.)
- TOGETHER_MODEL: Model name for Together AI (if using Together provider)

All these variables are documented in .env.example and .env.demos.example
"""

import os
import sys
import asyncio
import logging
import traceback
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

# Create file handler to save logs
log_dir = os.path.join(current_dir, "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"web_research_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(file_handler)
logger.setLevel(logging.DEBUG)  # Enable debug logging

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
    logger.debug(f"Script path: {os.path.abspath(__file__)}")
    logger.debug(f"Current directory: {os.getcwd()}")
    logger.debug(f"Environment: {settings.WEBAGENT_ENV}")
    
    # Log available API keys (without revealing the actual keys)
    logger.debug(f"Tavily API key configured: {bool(settings.TAVILY_API_KEY)}")
    logger.debug(f"Together API key configured: {bool(settings.TOGETHER_API_KEY)}")
    logger.debug(f"OpenAI API key configured: {bool(settings.OPENAI_API_KEY)}")
    logger.debug(f"Anthropic API key configured: {bool(settings.ANTHROPIC_API_KEY)}")
    
    # Ask user to confirm LLM provider and model selection
    logger.info("Prompting user for LLM provider and model selection")
    print("\n" + "=" * 80)
    print("WEB RESEARCH AGENT DEMO")
    print("=" * 80)
    print("This demo requires an LLM for processing web search results.")
    print("The default LLM provider is TogetherAI with the Llama-3.3-70B model.")
    
    try:
        # Confirm LLM provider and model
        logger.debug("Calling confirm_llm_selection() to get provider and model")
        provider, model = confirm_llm_selection(required_provider="together")
        logger.info(f"User selected LLM provider: {provider}")
        logger.info(f"User selected model: {model}")
    except Exception as e:
        logger.error(f"Error during LLM selection: {str(e)}")
        logger.error(traceback.format_exc())
        print(f"Error during LLM selection: {e}")
        return
    
    # Get web research agent
    print("\nInitializing Web Research Agent...")
    logger.info("Initializing Web Research Agent")
    
    try:
        web_agent = get_web_research_agent()
        logger.debug("Web Research Agent successfully initialized")
    except Exception as e:
        logger.error(f"Failed to initialize Web Research Agent: {str(e)}")
        logger.error(traceback.format_exc())
        print(f"ERROR: Failed to initialize Web Research Agent: {e}")
        
        # Check if the error is related to the TogetherAI API key
        if "TOGETHER_API_KEY" in str(e):
            print("\nThe TogetherAI API key is missing or invalid.")
            print("Please ensure the TOGETHER_API_KEY is properly set in your .env.local file")
            print(f"Looking for .env.local at: {os.path.join(os.getcwd(), '.env.local')}")
            
            # Suggest solution
            print("\nPossible solutions:")
            print("1. Check that .env.local exists in the project root")
            print("2. Ensure TOGETHER_API_KEY is properly set in the .env.local file")
            print("3. Restart the application after making changes")
            
        return
    
    # Define research queries to test
    logger.info("Prompting user to select a research query")
    print("\n" + "=" * 80)
    print("RESEARCH QUERIES")
    print("=" * 80)
    print("1. What are the latest developments in generative AI in 2025?")
    print("2. How have graph databases evolved in the past year?")
    print("3. What are the best practices for document processing in RAG systems?")
    print("4. Custom query")
    
    query_choice = input("\nSelect a query (1-4, default=1): ").strip()
    logger.debug(f"User selected query choice: {query_choice}")
    
    if query_choice == "2":
        query = "How have graph databases evolved in the past year?"
        logger.info("Selected query about graph databases")
    elif query_choice == "3":
        query = "What are the best practices for document processing in RAG systems?"
        logger.info("Selected query about RAG systems")
    elif query_choice == "4":
        query = input("\nEnter your custom research query: ").strip()
        logger.info(f"User entered custom query: {query}")
    else:
        query = "What are the latest developments in generative AI in 2025?"
        logger.info("Selected default query about generative AI")
    
    print(f"\nSelected research query: {query}")
    logger.info(f"Final query: {query}")
    
    # Ask about research parameters
    logger.info("Prompting user for research parameters")
    print("\n" + "=" * 80)
    print("RESEARCH PARAMETERS")
    print("=" * 80)
    print("1. Standard (balanced speed and depth)")
    print("2. Comprehensive (deeper research, more sources)")
    print("3. Quick (faster, fewer sources)")
    
    param_choice = input("\nSelect research approach (1-3, default=1): ").strip()
    logger.debug(f"User selected parameter choice: {param_choice}")
    
    research_priority = "balanced"
    max_sources = 3
    search_depth = "standard"
    
    if param_choice == "2":
        research_priority = "accuracy"
        max_sources = 5
        search_depth = "deep"
        print("\nSelected comprehensive research approach")
        logger.info("Selected comprehensive research approach")
    elif param_choice == "3":
        research_priority = "speed"
        max_sources = 2
        search_depth = "quick"
        print("\nSelected quick research approach")
        logger.info("Selected quick research approach")
    else:
        print("\nSelected standard research approach")
        logger.info("Selected standard research approach")
    
    logger.debug(f"Research parameters: priority={research_priority}, max_sources={max_sources}, depth={search_depth}")
    
    # Create a workflow state for this query
    logger.debug("Creating workflow state")
    state = WorkflowState(
        query=query,
        context={
            "query": query,
            "research_priority": research_priority,
            "max_sources": max_sources,
            "search_depth": search_depth,
            "research_plan": {
                "plan": f"Research about {query} with {research_priority} priority.",
                "requires_web_search": True,
                "search_depth": search_depth
            }
        }
    )
    logger.debug(f"Workflow state created: {state}")
    
    # Run web research agent
    logger.info(f"Running web research for: {query}")
    print(f"\nRunning web research with {provider} ({model}) as the LLM provider...")
    print(f"Research priority: {research_priority}, Max sources: {max_sources}")
    
    try:
        logger.debug("Starting web_agent function")
        start_time = datetime.now()
        
        # Record environment variables that might affect search
        logger.debug(f"Current TAVILY_API_KEY status: {'Set' if settings.TAVILY_API_KEY else 'Not set'}")
        logger.debug(f"Current LLM_PROVIDER: {settings.LLM_DEFAULT_PROVIDER}")
        logger.debug(f"Current TOGETHER_MODEL: {settings.TOGETHER_MODEL or 'Not set'}")
        
        # Call the web_agent function directly, not as a method
        result_state = await web_agent(state)
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"Web research completed in {duration:.2f} seconds")
        
        # Get the research results from the context
        research_results = result_state.context.get("web_research_results", {})
        
        # Ensure we have a valid result structure
        if not research_results or not isinstance(research_results, dict):
            logger.warning("Invalid research results format")
            research_results = {
                "analysis": "No analysis available",
                "sources": []
            }
        
        # If analysis is missing but sources exist, add a placeholder
        if "sources" in research_results and research_results["sources"] and "analysis" not in research_results:
            research_results["analysis"] = "No analysis was generated, but sources were found. Please review the sources directly."
        
        # Log detailed information about the result
        if research_results:
            logger.debug(f"Output contains {len(research_results.get('sources', []))} sources")
            
            # Log source domains for analysis
            source_domains = []
            for source in research_results.get('sources', []):
                url = source.get('url', '')
                if url:
                    try:
                        from urllib.parse import urlparse
                        domain = urlparse(url).netloc
                        source_domains.append(domain)
                    except Exception as domain_error:
                        logger.warning(f"Could not parse domain from url {url}: {domain_error}")
            if source_domains:
                logger.debug(f"Source domains: {', '.join(source_domains)}")
        
        # Print results
        print("\n" + "=" * 80)
        print(f"WEB RESEARCH RESULTS (completed in {duration:.2f} seconds)")
        print("=" * 80)
        
        if research_results:
            # Format the research output
            print("\nRESULTS:")
            print("=" * 80)
            
            # Print analysis if available
            if "analysis" in research_results:
                print("\nANALYSIS:")
                print(research_results.get("analysis", "No analysis provided"))
            
            # Print sources if available
            print("\nSOURCES:")
            print("=" * 80)
            for idx, source in enumerate(research_results.get("sources", [])):
                print(f"{idx+1}. {source.get('title', 'Untitled')} - {source.get('url', 'No URL')}")
        else:
            logger.warning("No research results returned")
            print("No research results were returned.")
            
        # Display results in GUI
        logger.info("Displaying results in GUI")
        try:
            display_research_results(
                queries=[query],
                results=[research_results],
                errors=[None],
                title="Web Research Results"
            )
            logger.debug("Successfully displayed results in GUI")
        except Exception as e:
            logger.error(f"Error displaying results in GUI: {str(e)}")
            logger.error(traceback.format_exc())
            print(f"Error displaying results in GUI: {e}")
    except Exception as e:
        logger.error(f"Error running web research agent: {str(e)}")
        logger.error(traceback.format_exc())
        print(f"\nERROR: {e}")
        
        # Try to get more detailed error info
        error_type = type(e).__name__
        logger.error(f"Error type: {error_type}")
        
        # Check for specific error types
        if "Tavily" in str(e):
            logger.error("This appears to be a Tavily API error. Check your API key and network connection.")
        elif "TogetherAI" in str(e) or "Together AI" in str(e):
            logger.error("This appears to be a Together API error. Check your API key and selected model.")
        elif "OpenAI" in str(e):
            logger.error("This appears to be an OpenAI API error. Check your API key and selected model.")
        
        # Display error in GUI
        try:
            logger.info("Displaying error in GUI")
            display_research_results(
                queries=[query],
                results=[None],
                errors=[str(e)],
                title="Web Research Results"
            )
            logger.debug("Successfully displayed error in GUI")
        except Exception as display_error:
            logger.error(f"Error displaying results in GUI: {str(display_error)}")
            logger.error(traceback.format_exc())
    
    # Save a copy of the results for analysis
    try:
        output_dir = os.path.join(current_dir, "results")
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = os.path.join(output_dir, f"web_research_result_{timestamp}.json")
        
        logger.info(f"Saving research results to {result_file}")
        with open(result_file, "w") as f:
            if research_results:
                json.dump(research_results, f, indent=2)
            else:
                json.dump({"error": "No results available"}, f)
    except Exception as save_error:
        logger.error(f"Error saving results: {str(save_error)}")
    
    logger.info("Web Research Agent Demo completed")
    print("\nWeb Research Agent Demo completed.")
    print(f"Logs saved to: {log_file}")
    print(f"Check the logs directory for detailed execution information.")

if __name__ == "__main__":
    # Run the demo
    try:
        logger.debug("Starting asyncio.run(run_web_research_demo())")
        asyncio.run(run_web_research_demo())
    except KeyboardInterrupt:
        logger.warning("Demo interrupted by user")
        print("\nDemo interrupted by user")
    except Exception as e:
        logger.critical(f"Unhandled exception in main: {str(e)}")
        logger.critical(traceback.format_exc())
        print(f"Critical error: {e}")
        print("See logs for details") 