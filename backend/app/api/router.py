"""
Main API router for the WebAgent backend.
Includes all endpoint routers from different modules.
"""
from fastapi import APIRouter, Depends

from app.api.endpoints import chat, tasks, health, frontend, documents
from app.core.security import require_api_key

api_router = APIRouter(dependencies=[Depends(require_api_key)])
health_router = APIRouter()
health_router.include_router(health.router, prefix="/health", tags=["health"])

# Include endpoint routers
api_router.include_router(chat.router, prefix="/chat", tags=["chat"])
api_router.include_router(tasks.router, prefix="/tasks", tags=["tasks"])
api_router.include_router(frontend.router, tags=["frontend"])
api_router.include_router(documents.router, prefix="/documents", tags=["documents"]) 