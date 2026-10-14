#!/usr/bin/env python
"""
Test script for deduplication and conditional reranking in the Internal Research Agent.
"""

import os
import sys
import logging
import asyncio
from pathlib import Path

# Dynamic path setup to support running from different locations
def setup_paths():
    """Set up the Python path to find modules regardless of where the script is run from."""
    script_dir = Path(os.path.abspath(os.path.dirname(__file__)))
    working_dir = Path(os.getcwd())
    
    # If we're in a 'tests' directory, go up one level to find backend
    if script_dir.name == 'tests':
        backend_dir = script_dir.parent
    else:
        backend_dir = working_dir
        
        # Try to find backend directory
        if working_dir.name == 'backend':
            backend_dir = working_dir
        elif 'backend' in working_dir.parts:
            # Navigate up until we find backend
            backend_dir = working_dir
            while backend_dir.name != 'backend' and backend_dir != backend_dir.parent:
                backend_dir = backend_dir.parent
        else:
            # We might be in the project root
            backend_dir = working_dir / 'backend'
            if not backend_dir.exists():
                # Last resort - try relative to script
                backend_dir = script_dir.parent
    
    # Add backend dir to path
    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))
        
    return backend_dir

# Set up paths
backend_dir = setup_paths()

# Import after path setup
try:
    from app.models.task import WorkflowState
    from app.agents.internal_research import get_internal_research_agent
except ImportError:
    # Second attempt with different import path
    sys.path.insert(0, str(backend_dir.parent))  # Add project root
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
    
    # Create a research agent function
    agent_function = get_internal_research_agent()
    
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
    result_state = await agent_function(state)
    
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

