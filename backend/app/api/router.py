"""
Main API router for the WebAgent backend.
Includes all endpoint routers from different modules.
"""
from fastapi import APIRouter, Depends

from app.api.endpoints import chat, tasks, health, frontend, documents
from app.core.security import require_api_key

api_router = APIRouter(dependencies=[Depends(require_api_key)])

# Only the basic health check is open; the detailed one shows service status and needs the key.
health_router = APIRouter()
health_router.add_api_route("/health/", health.health_check, methods=["GET"], tags=["health"])
api_router.add_api_route("/health/detailed", health.detailed_health, methods=["GET"], tags=["health"])

# Include endpoint routers
api_router.include_router(chat.router, prefix="/chat", tags=["chat"])
api_router.include_router(tasks.router, prefix="/tasks", tags=["tasks"])
api_router.include_router(frontend.router, tags=["frontend"])
api_router.include_router(documents.router, prefix="/documents", tags=["documents"]) 