"""
LLM service for the WebAgent backend.
Provides access to different LLM providers including OpenAI and Together AI.
"""
from typing import Dict, Optional
import logging
import os

from langchain_openai import ChatOpenAI
from langchain_together import Together as TogetherLLM
from app.core.config import settings
from app.core.loadEnvYAML import get_llm_config

logger = logging.getLogger(__name__)

# Define model constants
TOGETHER_DEFAULT_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"

def get_openai_llm(model_name: Optional[str] = None):
    """
    Get a LangChain ChatOpenAI instance.
    
    Args:
        model_name: Optional model name to use (defaults to settings.DEFAULT_MODEL)
        
    Returns:
        A configured ChatOpenAI instance
    """
    model = model_name or settings.DEFAULT_MODEL
    llm_config = get_llm_config()
    
    try:
        llm = ChatOpenAI(
            model=model,
            temperature=llm_config.temperature,
            timeout=llm_config.timeout_seconds,
            max_tokens=llm_config.max_tokens,
            api_key=settings.OPENAI_API_KEY,
        )
        return llm
    except Exception as e:
        logger.error(f"Error creating OpenAI LLM: {str(e)}")
        raise

def get_together_llm(model_name: Optional[str] = None):
    """
    Get a LangChain TogetherLLM instance.
    
    Args:
        model_name: Optional model name to use (defaults to TOGETHER_DEFAULT_MODEL)
        
    Returns:
        A configured TogetherLLM instance
    """
    model = model_name or TOGETHER_DEFAULT_MODEL
    llm_config = get_llm_config()
    
    # Check for Together API key in environment variables
    together_api_key = os.getenv("TOGETHER_API_KEY")
    
    if not together_api_key:
        logger.error("TOGETHER_API_KEY environment variable not set")
        raise ValueError("TOGETHER_API_KEY not set in environment variables")
    
    try:
        llm = TogetherLLM(
            model=model,
            temperature=llm_config.temperature,
            max_tokens=llm_config.max_tokens,
            together_api_key=together_api_key,
        )
        return llm
    except Exception as e:
        logger.error(f"Error creating Together AI LLM: {str(e)}")
        raise

def get_llm(model_name: Optional[str] = None):
    """
    Get a LangChain LLM instance based on the configured provider.
    
    Args:
        model_name: Optional model name to use (defaults to provider-specific default)
        
    Returns:
        A configured LLM instance
    """
    llm_config = get_llm_config()
    provider = llm_config.provider.lower()
    
    logger.info(f"Creating LLM instance with provider: {provider}")
    
    try:
        if provider == "together":
            return get_together_llm(model_name)
        elif provider == "openai":
            return get_openai_llm(model_name)
        else:
            logger.error(f"Unsupported LLM provider: {provider}")
            raise ValueError(f"Unsupported LLM provider: {provider}")
    except ValueError as e:
        # If Together AI is not available but was requested, fall back to OpenAI
        if provider == "together" and "TOGETHER_API_KEY not set" in str(e):
            logger.warning("Together AI not available, falling back to OpenAI")
            return get_openai_llm(model_name)
        else:
            raise

async def get_llm_status():
    """
    Check the status of the LLM service.
    
    Returns:
        Dict with status information
    """
    llm_config = get_llm_config()
    provider = llm_config.provider.lower()
    
    if provider == "together":
        if not os.getenv("TOGETHER_API_KEY"):
            return {"status": "unavailable", "error": "Together API key not configured", "provider": "together"}
        api_key = os.getenv("TOGETHER_API_KEY")
    elif provider == "openai":
        if not settings.OPENAI_API_KEY:
            return {"status": "unavailable", "error": "OpenAI API key not configured", "provider": "openai"}
        api_key = settings.OPENAI_API_KEY
    else:
        return {"status": "unavailable", "error": f"Unsupported provider: {provider}"}
    
    try:
        # Try creating an LLM instance
        llm = get_llm()
        
        # For status check, just return the provider info without making an actual API call
        # This avoids unnecessary costs for simple status checks
        return {
            "status": "ok",
            "model": llm_config.default_model,
            "provider": provider
        }
    except Exception as e:
        logger.error(f"LLM health check failed: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "model": llm_config.default_model,
            "provider": provider
        } 