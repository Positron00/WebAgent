"""
Middleware components for the WebAgent backend.

This module provides various middleware including:
- Security headers middleware
- Request validation middleware
- Logging middleware
"""
import time
import uuid
from typing import Callable, Dict
from app.core.log_context import redact_url

from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from app.core.api_errors import error_body
from starlette.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.logger import logger

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware to add security headers to all responses."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; object-src 'none'"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        # Remove unnecessary headers
        if "Server" in response.headers:
            del response.headers["Server"]
        
        return response

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log all requests."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate a unique request ID
        request_id = str(uuid.uuid4())
        
        # Add request ID to the request state
        request.state.request_id = request_id
        
        # Log request details
        client_host = request.client.host if request.client else "unknown"
        method = request.method
        url = redact_url(str(request.url))   # confirm/unsubscribe tokens are credentials
        
        logger.info(
            f"Request: {method} {url} - Client: {client_host} - ID: {request_id}"
        )
        
        # Record start time
        start_time = time.time()
        
        # Process the request
        try:
            response = await call_next(request)
            
            # Calculate processing time
            process_time = time.time() - start_time
            
            # Log response details
            logger.info(
                f"Response: {response.status_code} - Time: {process_time:.4f}s - ID: {request_id}"
            )
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            
            return response
        except Exception as e:
            # Log the exception
            logger.error(
                f"Request failed: {str(e)} - ID: {request_id}"
            )
            raise

class LimitSizeMiddleware(BaseHTTPMiddleware):
    """Middleware to limit the size of incoming requests."""
    
    def __init__(self, app: FastAPI, max_content_length: int = 10 * 1024 * 1024):  # 10 MB default
        super().__init__(app)
        self.max_content_length = max_content_length
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        content_length = request.headers.get("content-length")
        
        if content_length and int(content_length) > self.max_content_length:
            return JSONResponse(
                status_code=413,
                content=error_body("too_large", "Request too large"),
            )
        
        return await call_next(request)

def setup_middlewares(app: FastAPI) -> None:
    """Set up all middleware for the application."""
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
        allow_methods=settings.CORS_ALLOW_METHODS,
        allow_headers=settings.CORS_ALLOW_HEADERS,
    )
    
    # Add security headers middleware
    app.add_middleware(SecurityHeadersMiddleware)
    
    # Add request logging middleware
    app.add_middleware(RequestLoggingMiddleware)
    
    # Add request size limiting middleware
    app.add_middleware(LimitSizeMiddleware, max_content_length=settings.REQUEST_SIZE_LIMIT)