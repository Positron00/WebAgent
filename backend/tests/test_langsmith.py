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
import pytest

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
        assert config is not None, "LangSmith configuration should not be None"
        assert config.project_name, "Project name should be set"
    except Exception as e:
        logger.error(f"Failed to load LangSmith configuration: {str(e)}")
        assert False, f"Failed to load LangSmith configuration: {str(e)}"

def test_langsmith_client():
    """Test LangSmith client initialization."""
    try:
        client = get_langsmith_client()
        if client:
            logger.info("LangSmith client initialized successfully")
            assert client is not None, "LangSmith client should not be None"
        else:
            logger.warning("LangSmith client not initialized (API key may be missing)")
            # Skip test if API key is missing - don't fail
            pytest.skip("LangSmith client not initialized (API key may be missing)")
    except Exception as e:
        logger.error(f"Error initializing LangSmith client: {str(e)}")
        assert False, f"Error initializing LangSmith client: {str(e)}"

def test_langsmith_tracer():
    """Test LangSmith tracer initialization."""
    try:
        tracer = get_langchain_tracer()
        if tracer:
            logger.info("LangSmith tracer initialized successfully")
            assert tracer is not None, "LangSmith tracer should not be None"
        else:
            logger.warning("LangSmith tracer not initialized (API key may be missing)")
            # Skip test if API key is missing - don't fail
            pytest.skip("LangSmith tracer not initialized (API key may be missing)")
    except Exception as e:
        logger.error(f"Error initializing LangSmith tracer: {str(e)}")
        assert False, f"Error initializing LangSmith tracer: {str(e)}"

def test_langsmith_tags():
    """Test creating LangSmith tags."""
    try:
        tags = create_langsmith_tags(agent_name="test_agent")
        logger.info(f"LangSmith tags created successfully: {tags}")
        assert tags is not None, "LangSmith tags should not be None"
        assert len(tags) > 0, "LangSmith tags should not be empty"
        assert "agent" in tags, "Agent tag should be included"
        assert tags["agent"] == "test_agent", "Agent tag should have correct value"
    except Exception as e:
        logger.error(f"Error creating LangSmith tags: {str(e)}")
        assert False, f"Error creating LangSmith tags: {str(e)}"

def main():
    """Run all LangSmith tests."""
    logger.info("Starting LangSmith integration tests")
    
    # Check if LANGSMITH_API_KEY is set in settings
    from app.core.config import settings
    api_key = settings.LANGSMITH_API_KEY
    if not api_key:
        logger.warning("LANGSMITH_API_KEY not configured in settings (.env.local)")
    else:
        logger.info("LANGSMITH_API_KEY environment variable is set")
    
    # Run tests
    tests = [
        ("LangSmith config", test_langsmith_config),
        ("LangSmith client", test_langsmith_client),
        ("LangSmith tracer", test_langsmith_tracer),
        ("LangSmith tags", test_langsmith_tags)
    ]
    
    success_count = 0
    total_count = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"Running test: {test_name}")
        try:
            test_func()
            logger.info(f"✓ {test_name} test passed")
            success_count += 1
        except pytest.skip.Exception as e:
            logger.warning(f"⚠ {test_name} test skipped: {str(e)}")
        except Exception as e:
            logger.error(f"✗ {test_name} test failed: {str(e)}")
    
    # Print summary
    logger.info(f"Test summary: {success_count}/{total_count} tests passed")
    
    return success_count == total_count

