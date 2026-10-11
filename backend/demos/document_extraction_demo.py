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
from dotenv import load_dotenv

# Suppress specific warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning"

# Setup path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # backend directory
root_dir = os.path.dirname(parent_dir)     # project root directory
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Load environment variables from root directory first
load_dotenv(os.path.join(root_dir, ".env"))
# Load from .env.local in root dir with higher priority
load_dotenv(os.path.join(root_dir, ".env.local"), override=True)

# Change working directory to backend to ensure mlruns is created there
os.chdir(parent_dir)

# Configure logging - only show INFO and higher level messages
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("document_extraction_demo")
logger.info(f"Changed working directory to: {os.getcwd()}")

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
from backend.app.services.llm import confirm_llm_selection

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
    
    # Ask user to confirm LLM provider and model selection
    print("\n" + "=" * 80)
    print("DOCUMENT EXTRACTION AGENT DEMO")
    print("=" * 80)
    print("This demo requires an LLM for extracting information from documents.")
    print("The default LLM provider is TogetherAI with the Muse-Glimmer-30B model.")
    
    # Confirm LLM provider and model
    provider, model = confirm_llm_selection()
    
    # Create a sample document if needed
    sample_doc_path = get_sample_document()
    
    # Get document loader service
    doc_loader = get_document_loader_service()
    
    # Get document extraction agent
    print("\nInitializing Document Extraction Agent...")
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
    print(f"\nRunning extraction with {provider} ({model}) as the LLM provider...")
    try:
        result_state = await extraction_agent(state)
        
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
        result_state = await extraction_agent(state)
        
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