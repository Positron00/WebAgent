#!/usr/bin/env python
"""
Test script to verify LangSmith integration.

This script tests the LangSmith configuration and tracing functionality.

Can be run from:
- WebAgent root: python backend/tests/test_langsmith.py
- Backend directory: python tests/test_langsmith.py
- As module: python -m backend.tests.test_langsmith or python -m tests.test_langsmith
"""
import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

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

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import after environment is loaded and paths are set up
try:
    from app.core.setup_langsmith import setup_langsmith, get_langsmith_client, get_langchain_tracer, create_langsmith_tags
    from app.core.loadEnvYAML import get_langsmith_config
except ImportError:
    # Second attempt with different import path
    sys.path.insert(0, str(backend_dir.parent))  # Add project root
    from backend.app.core.setup_langsmith import setup_langsmith, get_langsmith_client, get_langchain_tracer, create_langsmith_tags
    from backend.app.core.loadEnvYAML import get_langsmith_config

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