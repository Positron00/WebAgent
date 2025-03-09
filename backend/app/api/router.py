"""
Main API router for the WebAgent backend.
Includes all endpoint routers from different modules.
"""
from fastapi import APIRouter

from app.api.endpoints import chat, tasks, health, frontend

api_router = APIRouter()

# Include endpoint routers
api_router.include_router(chat.router, prefix="/chat", tags=["chat"])
api_router.include_router(tasks.router, prefix="/tasks", tags=["tasks"])
api_router.include_router(health.router, prefix="/health", tags=["health"])
api_router.include_router(frontend.router, prefix="/v1", tags=["frontend"]) 