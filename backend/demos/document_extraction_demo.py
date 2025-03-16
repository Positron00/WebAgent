#!/usr/bin/env python
"""
Document Extraction Agent Demo
==============================

This demo demonstrates the capabilities of the Document Extraction Agent,
which specializes in extracting structured information from documents.

Usage:
    python -m backend.demos.document_extraction_demo

Requirements:
    - Sample documents in the demo_docs/ directory
    - Configured environment variables (check .env.example)
"""

import os
import sys
import asyncio
import logging
import tempfile
import warnings
from pathlib import Path
from datetime import datetime
import json
from pprint import pprint

# Suppress specific warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning"

# Setup path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Configure logging - only show INFO and higher level messages
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("document_extraction_demo")

# Suppress specific loggers that produce warning messages
logging.getLogger("numexpr").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("chromadb.telemetry").setLevel(logging.ERROR)
logging.getLogger("langchain").setLevel(logging.ERROR)

# Import agents and services
from backend.app.agents.document_extraction_agent import get_document_extraction_agent
from backend.app.services.document_loader import get_document_loader_service
from backend.app.services.document_ingestion_service import get_document_ingestion_service
from backend.app.models.task import WorkflowState
from backend.app.core.config import settings
from backend.app.utils.metrics import timing_decorator

# Import the research results display
from backend.demos.research_results_display import display_research_results

# Path to demo documents
DEMO_DOCS_DIR = os.path.join(current_dir, "demo_docs")
os.makedirs(DEMO_DOCS_DIR, exist_ok=True)

# Create a sample document if none exists
def get_sample_document():
    """Get a sample document path for testing."""
    # Get the document ingestion service
    doc_ingestion = get_document_ingestion_service()
    
    # Get the financial report sample document
    sample_path = doc_ingestion.get_sample_document_path("sample_financial_report.txt")
    
    logger.info(f"Using sample document: {sample_path}")
    return sample_path


@timing_decorator
async def run_document_extraction_demo():
    """Run the document extraction agent demo."""
    logger.info("Starting Document Extraction Agent Demo")
    
    # Initialize results list for the GUI
    results = []
    
    # Ask for user confirmation to connect to the TogetherAI LLM
    print("\n" + "=" * 80)
    print("This demo will connect to the TogetherAI LLM API using your configured API key.")
    try:
        model_info = settings.LLM_PROVIDER if hasattr(settings, 'LLM_PROVIDER') else 'together'
        model_name = settings.DEFAULT_MODEL if hasattr(settings, 'DEFAULT_MODEL') else 'meta-llama/Llama-3.3-70B-Instruct-Turbo-Free'
        print(f"The model that will be used is: {model_name} (Provider: {model_info})")
    except AttributeError:
        print("Using default TogetherAI LLM configuration")
    print("=" * 80)
    
    confirmation = input("\nDo you want to continue and connect to the LLM API? (Y/n): ")
    if confirmation.lower() not in ["y", "yes",""]:
        logger.info("User declined to connect to the LLM API. Demo aborted.")
        print("Demo aborted by user.")
        return
    
    logger.info("User confirmed to connect to the LLM API.")
    print("\nConnecting to TogetherAI LLM API...")
    
    # Create a sample document if needed
    sample_doc_path = get_sample_document()
    
    # Get document loader service
    doc_loader = get_document_loader_service()
    
    # Get document extraction agent
    extraction_agent = get_document_extraction_agent()
    
    # Load document
    logger.info(f"Loading document: {sample_doc_path}")
    try:
        artifact_id = await doc_loader.load_document(sample_doc_path)
        logger.info(f"Document loaded. Artifact ID: {artifact_id}")
    except Exception as e:
        logger.error(f"Error loading document: {e}")
        return
    
    # Add delay to avoid rate limits
    await asyncio.sleep(1.5)
    print("Waiting 1.5 seconds to avoid API rate limits...")
    
    # Create workflow state
    state = WorkflowState(
        query="Extract financial metrics and department budgets from the Q1 2025 financial report",
        inputs={"document_id": artifact_id}
    )
    
    # Run extraction agent
    logger.info("Running document extraction agent...")
    try:
        result_state = await extraction_agent.run(state)
        
        # Print results
        logger.info("Document Extraction Results:")
        if result_state.output:
            pprint(result_state.output)
            # Add to results for GUI
            results.append(("Financial Metrics", result_state.output))
        else:
            logger.warning("No extraction results returned.")
            
    except Exception as e:
        logger.error(f"Error running document extraction agent: {e}")
    
    # Add delay to avoid rate limits
    await asyncio.sleep(1.5)
    print("Waiting 1.5 seconds to avoid API rate limits...")
    
    # Try another extraction with different query
    logger.info("\nRunning another extraction with different query...")
    state.query = "Create a summary of market analysis and outlook from the report"
    
    try:
        result_state = await extraction_agent.run(state)
        
        # Print results
        logger.info("Market Analysis Extraction Results:")
        if result_state.output:
            pprint(result_state.output)
            # Add to results for GUI
            results.append(("Market Analysis", result_state.output))
        else:
            logger.warning("No extraction results returned.")
            
    except Exception as e:
        logger.error(f"Error running document extraction agent: {e}")
    
    logger.info("Document Extraction Agent Demo completed.")
    
    # Show results in GUI
    if results:
        # Prepare data for display
        queries = []
        extraction_results = []
        errors = []
        
        for title, result in results:
            queries.append(title)
            extraction_results.append(result)
            errors.append(None)
        
        try:
            # Display results using the generic display function
            display_research_results(
                queries=queries,
                results=extraction_results,
                errors=errors,
                title="Document Extraction Results"
            )
        except Exception as e:
            logger.error(f"Error displaying results in GUI: {e}")

if __name__ == "__main__":
    # Run the demo
    asyncio.run(run_document_extraction_demo()) 