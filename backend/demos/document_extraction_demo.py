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
from pathlib import Path
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
logger = logging.getLogger("document_extraction_demo")

# Import agents and services
from backend.app.agents.document_extraction_agent import get_document_extraction_agent
from backend.app.services.document_loader import get_document_loader_service
from backend.app.models.task import WorkflowState
from backend.app.core.config import settings
from backend.app.utils.metrics import timing_decorator

# Path to demo documents
DEMO_DOCS_DIR = os.path.join(current_dir, "demo_docs")
os.makedirs(DEMO_DOCS_DIR, exist_ok=True)

# Create a sample document if none exists
def create_sample_document():
    """Create a sample document for testing if none exists."""
    sample_path = os.path.join(DEMO_DOCS_DIR, "sample_financial_report.txt")
    
    if not os.path.exists(sample_path):
        with open(sample_path, "w") as f:
            f.write("""
Financial Report - Q1 2025
==========================

Executive Summary
----------------
In Q1 2025, WebAgent Corp achieved significant growth with revenue reaching $12.5M, 
representing a 15% increase compared to Q1 2024. EBITDA improved by 22% to $3.2M, 
while net profit rose to $1.8M.

Financial Highlights
-------------------
- Revenue: $12.5M (+15% YoY)
- EBITDA: $3.2M (+22% YoY)
- Net Profit: $1.8M (+18% YoY)
- Operating Margin: 25.6% (up from 23.1%)
- Cash Reserves: $8.4M (as of March 31, 2025)

Department Breakdown
-------------------
1. Product Development: $4.2M investment, focused on AI agent capabilities
2. Sales & Marketing: $2.8M, with 65% digital marketing allocation
3. Operations: $1.9M, improved efficiency by 12%
4. Research: $2.1M, primarily for new vector database integration
5. Administration: $1.5M

Market Analysis
--------------
The AI agent market grew by approximately 28% in Q1 2025, with WebAgent gaining 2.3% 
market share. Key competitors include AgentWorks (18.2% share), AI Solutions (15.6%), 
and TechAgents (12.1%).

Outlook for Q2 2025
------------------
We project continued growth in Q2 with revenue estimates between $13.8M-$14.5M and 
EBITDA of $3.5M-$3.8M. Key initiatives include launching the new WebAgent Enterprise 
platform and expanding to three new international markets.
            """)
        logger.info(f"Created sample document: {sample_path}")
    return sample_path

@timing_decorator
async def run_document_extraction_demo():
    """Run the document extraction agent demo."""
    logger.info("Starting Document Extraction Agent Demo")
    
    # Create a sample document if needed
    sample_doc_path = create_sample_document()
    
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
        else:
            logger.warning("No extraction results returned.")
            
    except Exception as e:
        logger.error(f"Error running document extraction agent: {e}")
    
    # Try another extraction with different query
    logger.info("\nRunning another extraction with different query...")
    state.query = "Create a summary of market analysis and outlook from the report"
    
    try:
        result_state = await extraction_agent.run(state)
        
        # Print results
        logger.info("Market Analysis Extraction Results:")
        if result_state.output:
            pprint(result_state.output)
        else:
            logger.warning("No extraction results returned.")
            
    except Exception as e:
        logger.error(f"Error running document extraction agent: {e}")
    
    logger.info("Document Extraction Agent Demo completed.")

if __name__ == "__main__":
    # Run the demo
    asyncio.run(run_document_extraction_demo()) 