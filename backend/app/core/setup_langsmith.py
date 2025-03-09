"""
LangSmith setup and configuration for the WebAgent backend.
Provides tracing, logging, and observability for LLM-based agents.
"""
import os
import logging
from typing import Dict, Any, Optional

from langsmith import Client
from langchain.callbacks.tracers import LangChainTracer
from langchain.callbacks.tracers.langchain import wait_for_all_tracers

from app.core.config import settings

logger = logging.getLogger(__name__)

# Global LangSmith client
_langsmith_client = None
# Global LangChain tracer
_langchain_tracer = None

def setup_langsmith():
    """
    Configure LangSmith tracing based on environment settings.
    Sets required environment variables for LangSmith.
    """
    global _langsmith_client, _langchain_tracer
    
    if not settings.LANGSMITH_API_KEY:
        logger.warning("LangSmith API key not set. Tracing will be disabled.")
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        return False
    
    try:
        # Set up environment variables for LangSmith
        os.environ["LANGCHAIN_TRACING_V2"] = str(settings.LANGSMITH_TRACING_ENABLED).lower()
        os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
        os.environ["LANGCHAIN_API_KEY"] = settings.LANGSMITH_API_KEY
        os.environ["LANGCHAIN_PROJECT"] = settings.LANGSMITH_PROJECT_NAME
        
        # Initialize client
        _langsmith_client = Client(
            api_key=settings.LANGSMITH_API_KEY,
            api_url="https://api.smith.langchain.com"
        )
        
        # Initialize tracer with project name
        _langchain_tracer = LangChainTracer(
            project_name=settings.LANGSMITH_PROJECT_NAME
        )
        
        logger.info(f"LangSmith tracing initialized for project: {settings.LANGSMITH_PROJECT_NAME}")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize LangSmith: {str(e)}")
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        return False

def get_langsmith_client() -> Optional[Client]:
    """
    Get the LangSmith client singleton.
    
    Returns:
        LangSmith client or None if not initialized
    """
    global _langsmith_client
    if _langsmith_client is None and settings.LANGSMITH_API_KEY:
        setup_langsmith()
    return _langsmith_client

def get_langchain_tracer() -> Optional[LangChainTracer]:
    """
    Get the LangChain tracer singleton.
    
    Returns:
        LangChain tracer or None if not initialized
    """
    global _langchain_tracer
    if _langchain_tracer is None and settings.LANGSMITH_API_KEY:
        setup_langsmith()
    return _langchain_tracer

def create_langsmith_tags(agent_name: str, additional_tags: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    """
    Create a standardized set of tags for LangSmith tracing.
    
    Args:
        agent_name: The name of the agent or component being traced
        additional_tags: Optional additional tags to include
        
    Returns:
        Dictionary of tags for LangSmith
    """
    # Get version from config if possible
    try:
        from app.core.config import settings
        version = settings.VERSION
    except ImportError:
        version = "unknown"
    
    # Get environment
    env = os.getenv("WEBAGENT_ENV", "dev")
    
    # Create base tags
    base_tags = {
        "agent": agent_name,
        "version": version,
        "environment": env
    }
    
    # Add additional tags if provided
    if additional_tags:
        base_tags.update(additional_tags)
    
    return base_tags

# Initialize LangSmith on module import
setup_langsmith() 