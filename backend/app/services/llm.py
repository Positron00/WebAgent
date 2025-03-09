"""
LLM service for the WebAgent backend.
Provides access to OpenAI models and utilities.
"""
from typing import Dict, Optional
import logging

from langchain_openai import ChatOpenAI
from app.core.config import settings

logger = logging.getLogger(__name__)

def get_llm(model_name: Optional[str] = None):
    """
    Get a LangChain ChatOpenAI instance.
    
    Args:
        model_name: Optional model name to use (defaults to settings.DEFAULT_MODEL)
        
    Returns:
        A configured ChatOpenAI instance
    """
    model = model_name or settings.DEFAULT_MODEL
    
    try:
        llm = ChatOpenAI(
            model=model,
            temperature=0.7,
            api_key=settings.OPENAI_API_KEY,
        )
        return llm
    except Exception as e:
        logger.error(f"Error creating LLM: {str(e)}")
        raise

async def get_llm_status():
    """
    Check the status of the LLM service.
    
    Returns:
        Dict with status information
    """
    if not settings.OPENAI_API_KEY:
        return {"status": "unavailable", "error": "API key not configured"}
    
    try:
        # Create LLM with minimal configuration
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            max_tokens=5,
            temperature=0,
            api_key=settings.OPENAI_API_KEY,
        )
        
        # Try a simple completion
        response = await llm.ainvoke("Say hello")
        
        return {
            "status": "ok",
            "model": settings.DEFAULT_MODEL,
            "provider": "openai"
        }
    except Exception as e:
        logger.error(f"LLM health check failed: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "model": settings.DEFAULT_MODEL,
            "provider": "openai"
        } 