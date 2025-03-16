#!/usr/bin/env python
"""
Test script for deduplication and conditional reranking in the Internal Research Agent.
"""

import os
import logging
import asyncio
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
    result_state = await agent.run(state)
    
    # Check the results
    agent_output = result_state.agent_outputs.get("internal_research", {})
    documents = agent_output.get("documents", [])
    
    logger.info(f"Number of documents after potential deduplication: {len(documents)}")
    logger.info(f"Metrics: {agent_output.get('metrics', {})}")
    
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