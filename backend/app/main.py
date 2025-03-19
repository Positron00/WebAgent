"""Main FastAPI application.

This module initializes the FastAPI application and includes all routes.
"""

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import docs, models
from app.api.routes.observability import router as observability_router
from app.core.logger import logger

# Create FastAPI app
app = FastAPI(
    title="AI Observability API",
    description="WebAgent Backend API - Advanced agentic capabilities for web research, data analysis, and code generation",
    version="2.9.3",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(docs.router)
app.include_router(models.router)
app.include_router(observability_router)

@app.on_event("startup")
async def startup_event():
    """Run startup tasks."""
    logger.info("Starting up AI Observability API")

@app.on_event("shutdown")
async def shutdown_event():
    """Run shutdown tasks."""
    logger.info("Shutting down AI Observability API")

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Welcome to AI Observability API",
        "docs_url": "/docs",
        "metrics_url": "/observability/metrics",
        "health_url": "/observability/health",
    } 