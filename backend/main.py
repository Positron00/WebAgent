"""
WebAgent Backend Microservice
----------------------------
Main application entry point for the FastAPI server.
"""
import asyncio
import os
import sys
import uuid
import uvicorn
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime

# Import path setup to ensure imports work correctly
from pathlib import Path
# Add the current directory to the path if not already there
current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Now import application modules
from app.api.router import api_router, health_router
from app.api.endpoints.task import router as task_router
from app.api.endpoints.triage import router as triage_router
from starlette.concurrency import run_in_threadpool
from app.services import triage_collect
from app.services.triage_model import warm_up
from app.core.config import settings
from app.core.middleware import setup_middlewares, LimitSizeMiddleware, SecurityHeadersMiddleware
from app.core.logger import logger
from app.core.metrics import setup_metrics, PrometheusMiddleware, ERROR_COUNT
from app.core.startup_checks import check_production_settings, server_code_execution_enabled
from app.core.security import (
    setup_security,
    require_api_key,
    sanitize_error_message,
)
from fastapi import Depends

# Create FastAPI app with OpenAPI documentation
app = FastAPI(
    title=settings.PROJECT_NAME,
    description=settings.PROJECT_DESCRIPTION,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url=f"{settings.API_V1_STR}/docs",
    redoc_url=f"{settings.API_V1_STR}/redoc",
    openapi_tags=[
        {"name": "health", "description": "Health check endpoints"},
        {"name": "chat", "description": "Chat API endpoints"},
        {"name": "tasks", "description": "Task management endpoints"},
        {"name": "frontend", "description": "Frontend compatibility endpoints"},
    ]
)

# First add the metrics middleware to ensure all requests are tracked
app.add_middleware(PrometheusMiddleware)

# Set up middleware in correct order
# 1. Security headers should be first to ensure all responses have correct headers
app.add_middleware(SecurityHeadersMiddleware)

# 2. Request size limiting should happen early to prevent DoS
app.add_middleware(
    LimitSizeMiddleware, 
    max_content_length=settings.REQUEST_SIZE_LIMIT
)

# 3. Set up other middlewares with the app
setup_middlewares(app)

# Add CORS middleware last since it's not critical for security
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up metrics endpoints
setup_metrics(app)
setup_security(app)

# Model-written code must not run inside the API server unless explicitly enabled.
settings.ENABLE_CODE_EXECUTION = server_code_execution_enabled()


@app.on_event("startup")
async def validate_production_settings() -> None:
    # Also run before uvicorn starts in the Docker image (see app/core/startup_checks.py).
    check_production_settings(settings.ENABLE_CODE_EXECUTION)
    cfg = getattr(settings, "TRIAGE", None)
    if cfg and cfg.enabled:
        await run_in_threadpool(warm_up)
    # Overrides are collected even with triage disabled, so retention cleanup always runs.
    triage_collect.cleanup()
    app.state.triage_cleanup_task = asyncio.create_task(triage_collect.cleanup_daily())


@app.on_event("shutdown")
async def stop_triage_cleanup() -> None:
    task = getattr(app.state, "triage_cleanup_task", None)
    if task is not None:
        task.cancel()


# Include API router
app.include_router(health_router, prefix=settings.API_V1_STR)
app.include_router(api_router, prefix=settings.API_V1_STR)
app.include_router(
    task_router,
    prefix="/api/task",
    tags=["task"],
    dependencies=[Depends(require_api_key)],
)
app.include_router(
    triage_router,
    prefix="/api/triage",
    tags=["triage"],
    dependencies=[Depends(require_api_key)],
)

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler to log all unhandled exceptions."""
    # Generate a unique error ID or use existing one from request state
    error_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    
    # Get additional context for better debugging
    method = request.method
    url = str(request.url)
    client_host = request.client.host if request.client else "unknown"
    exception_type = type(exc).__name__
    exception_msg = str(exc)
    
    # Record error in metrics
    ERROR_COUNT.labels(type=exception_type, component="global_handler").inc()
    
    # Log the exception with context
    logger.error(
        f"Unhandled exception: {exception_type}: {exception_msg}",
        extra={
            "request_id": error_id,
            "method": method,
            "url": url,
            "client_host": client_host,
            "exception_type": exception_type,
            "exception_message": exception_msg,
        }
    )
    
    # Return a JSON response with appropriate error details
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal Server Error",
            "message": (
                sanitize_error_message(exception_msg)
                if settings.DEBUG_MODE
                else "An unexpected error occurred"
            ),
            "error_id": error_id,
            "timestamp": datetime.now().isoformat(),
        },
    )

# Health check endpoint
@app.get("/")
async def health_check():
    """Simple health check endpoint"""
    logger.info("Health check requested")
    return {
        "status": "ok", 
        "version": settings.VERSION,
        "environment": settings.WEBAGENT_ENV,
    }


# Liveness/readiness probe for Docker HEALTHCHECK and Kubernetes. No auth, no rate limit,
# and not logged, since probes call it every few seconds.
@app.get("/health", include_in_schema=False)
async def liveness_probe():
    return {"status": "ok", "version": settings.VERSION}

# Log startup information
logger.info(f"Starting {settings.PROJECT_NAME} v{settings.VERSION}")
logger.info(f"Environment: {settings.WEBAGENT_ENV}")
logger.info(f"Debug mode: {settings.DEBUG_MODE}")
logger.info(f"API base URL: {settings.API_V1_STR}")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG_MODE,
        reload_dirs=[str(current_dir)],
        log_level="info",
    ) 