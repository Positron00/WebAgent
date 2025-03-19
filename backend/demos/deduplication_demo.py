#!/usr/bin/env python
"""
Document Deduplication Demo

This script demonstrates the document deduplication capabilities of the WebAgent.
It shows how the system identifies and removes duplicate information from research results.

Run with: python -m backend.demos.deduplication_demo
"""

import os
import logging
import asyncio
import sys
import time

# Add the parent directory to the path to import modules correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.app.models.task import WorkflowState
from backend.app.agents.internal_research import get_internal_research_agent

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set environment variables for testing
os.environ["RETRIEVAL_MODE"] = "vector"
os.environ["LLM_PROVIDER"] = "none"

async def test_deduplication():
    """Test the deduplication functionality."""
    logger.info("Testing deduplication functionality")
    
    # Create a research agent
    agent = get_internal_research_agent()
    
    # Create a test workflow state
    state = WorkflowState(
        query="ethics AI",
        context={
            "research_priority": "accuracy", 
            "knowledge_source": "internal",
            "include_sources": True,
            "retrieval_mode": "vector"
        }
    )
    
    # Run the agent
    logger.info("Running the agent...")
    start_time = time.time()
    result_state = await agent.run(state)
    end_time = time.time()
    
    # Check the results
    agent_output = result_state.agent_outputs.get("internal_research", {})
    documents = agent_output.get("documents", [])
    
    logger.info(f"Number of documents after potential deduplication: {len(documents)}")
    logger.info(f"Metrics: {agent_output.get('metrics', {})}")
    logger.info(f"Total execution time: {end_time - start_time:.2f} seconds")
    
    # Display top documents
    for i, doc in enumerate(documents[:3], 1):
        score = doc.get("score", 0)
        content = doc.get("page_content", "")[:100]
        logger.info(f"Document {i}: (Score: {score:.4f}) {content}...")
    
    return result_state

if __name__ == "__main__":
    logger.info("Starting deduplication test")
    asyncio.run(test_deduplication())
    logger.info("Test completed") 