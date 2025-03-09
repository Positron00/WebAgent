#!/usr/bin/env python
"""
Test script to verify LangSmith integration.

This script tests the LangSmith configuration and tracing functionality.
"""
import os
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import after environment is loaded
from app.core.setup_langsmith import setup_langsmith, get_langsmith_client, get_langchain_tracer, create_langsmith_tags
from app.core.loadEnvYAML import get_langsmith_config

def test_langsmith_config():
    """Test loading LangSmith configuration."""
    try:
        config = get_langsmith_config()
        logger.info(f"LangSmith configuration loaded successfully:")
        logger.info(f"  Project name: {config.project_name}")
        logger.info(f"  Tracing enabled: {config.tracing_enabled}")
        logger.info(f"  Log level: {config.log_level}")
        return True
    except Exception as e:
        logger.error(f"Failed to load LangSmith configuration: {str(e)}")
        return False

def test_langsmith_client():
    """Test LangSmith client initialization."""
    try:
        client = get_langsmith_client()
        if client:
            logger.info("LangSmith client initialized successfully")
            return True
        else:
            logger.warning("LangSmith client not initialized (API key may be missing)")
            return False
    except Exception as e:
        logger.error(f"Error initializing LangSmith client: {str(e)}")
        return False

def test_langsmith_tracer():
    """Test LangSmith tracer initialization."""
    try:
        tracer = get_langchain_tracer()
        if tracer:
            logger.info("LangSmith tracer initialized successfully")
            return True
        else:
            logger.warning("LangSmith tracer not initialized (API key may be missing)")
            return False
    except Exception as e:
        logger.error(f"Error initializing LangSmith tracer: {str(e)}")
        return False

def test_langsmith_tags():
    """Test creating LangSmith tags."""
    try:
        tags = create_langsmith_tags(agent_name="test_agent")
        logger.info(f"LangSmith tags created successfully: {tags}")
        return True
    except Exception as e:
        logger.error(f"Error creating LangSmith tags: {str(e)}")
        return False

def main():
    """Run all LangSmith tests."""
    logger.info("Starting LangSmith integration tests")
    
    # Check if LANGSMITH_API_KEY is set
    api_key = os.getenv("LANGSMITH_API_KEY")
    if not api_key:
        logger.warning("LANGSMITH_API_KEY environment variable is not set")
    else:
        logger.info("LANGSMITH_API_KEY environment variable is set")
    
    # Run tests
    config_test = test_langsmith_config()
    client_test = test_langsmith_client()
    tracer_test = test_langsmith_tracer()
    tags_test = test_langsmith_tags()
    
    # Report results
    all_passed = all([config_test, client_test, tracer_test, tags_test])
    if all_passed:
        logger.info("All LangSmith integration tests passed!")
    else:
        logger.warning("Some LangSmith integration tests failed")
    
    return all_passed

if __name__ == "__main__":
    main() 