"""
WebAgent Backend Microservice
----------------------------
Main application entry point for the FastAPI server.
"""
import os
import sys
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.router import api_router
from app.core.config import settings
from app.core.middleware import setup_middlewares, LimitSizeMiddleware, SecurityHeadersMiddleware
from app.core.logger import logger
from app.core.metrics import setup_metrics, PrometheusMiddleware

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

# Set up middleware
setup_middlewares(app)

# Set up metrics
setup_metrics(app)

# Add metrics middleware
app.add_middleware(PrometheusMiddleware)

# Set up CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(api_router, prefix=settings.API_V1_STR)

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler to log all unhandled exceptions."""
    # Generate a unique error ID
    error_id = request.state.request_id if hasattr(request.state, "request_id") else "unknown"
    
    # Log the exception
    logger.error(
        f"Unhandled exception: {str(exc)}",
        extra={
            "request_id": error_id,
            "method": request.method,
            "url": str(request.url),
            "client_host": request.client.host if request.client else "unknown",
            "exception_type": type(exc).__name__,
        }
    )
    
    # Return a JSON response
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": str(exc) if settings.DEBUG_MODE else "An unexpected error occurred",
            "error_id": error_id,
        },
    )

# Health check endpoint
@app.get("/")
async def health_check():
    """Simple health check endpoint"""
    logger.info("Health check requested")
    return {"status": "ok", "version": settings.VERSION}

# Log startup information
logger.info(f"Starting {settings.PROJECT_NAME} v{settings.VERSION}")
logger.info(f"Environment: {settings.WEBAGENT_ENV}")
logger.info(f"Debug mode: {settings.DEBUG_MODE}")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG_MODE,
        log_level="info",
    ) 