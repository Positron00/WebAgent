"""
Health check endpoints for the WebAgent backend.
"""
from fastapi import APIRouter, Depends

from app.core.config import settings
from app.services.llm import get_llm_status
from app.services.vectordb import get_vectordb_status

router = APIRouter()

@router.get("/")
async def health_check():
    """
    Basic health check endpoint for the API.
    """
    return {
        "status": "ok",
        "version": settings.VERSION,
        "environment": "development" if settings.DEBUG_MODE else "production",
    }

@router.get("/detailed")
async def detailed_health():
    """
    Detailed health check with information about connected services.
    """
    llm_status = await get_llm_status()
    vectordb_status = await get_vectordb_status()
    
    return {
        "api": {
            "status": "ok",
            "version": settings.VERSION,
        },
        "llm_service": llm_status,
        "vector_db": vectordb_status,
    } 