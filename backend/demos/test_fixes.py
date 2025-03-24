#!/usr/bin/env python
"""
Test script for verifying fixes to the internal research demo.
"""
import os
import sys
import asyncio
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_fixes")

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import the components we need to test
from internal_research_demo import process_vector_only_query, SimpleRetriever, SimpleReranker

async def test_vector_only():
    """Test the vector-only retrieval with our fixes."""
    logger.info("Testing vector-only retrieval with SimpleRetriever.retrieve() method")
    
    # Use a sample document
    sample_docs = [os.path.join(current_dir, "demo_docs/ai_ethics_guidelines.txt")]
    if not os.path.exists(sample_docs[0]):
        # Try the backend data directory
        sample_docs = [os.path.join(current_dir, "../data/sample_docs/ai_ethics_guidelines.txt")]
        if not os.path.exists(sample_docs[0]):
            logger.error(f"Sample document not found: {sample_docs[0]}")
            return
    
    # Create the SimpleRetriever and SimpleReranker
    retriever = SimpleRetriever(sample_docs)
    reranker = SimpleReranker()
    
    # Run the vector-only retrieval
    result = await process_vector_only_query("What are the guidelines for AI ethics?", retriever, reranker)
    
    # Check the result
    if result.get("documents"):
        logger.info(f"Success! Retrieved {len(result.get('documents', []))} documents")
        return True
    else:
        logger.error("No documents retrieved")
        return False

if __name__ == "__main__":
    # Run the test
    success = asyncio.run(test_vector_only())
    if success:
        print("Test completed successfully!")
    else:
        print("Test failed!")
        sys.exit(1) 