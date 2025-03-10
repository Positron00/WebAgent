"""
Metrics configuration for the WebAgent backend.

This module uses the Prometheus client library to expose metrics
for monitoring the application's performance and health.
"""
import time
from functools import wraps
from typing import Callable, Dict, Optional

from fastapi import FastAPI, Request, Response
from prometheus_client import (
    REGISTRY,
    Counter,
    Gauge,
    Histogram,
    Summary,
    CollectorRegistry,
    generate_latest,
)
from starlette.middleware.base import BaseHTTPMiddleware

# Create metrics
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total number of HTTP requests",
    ["method", "endpoint", "status_code"]
)

REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10)
)

REQUESTS_IN_PROGRESS = Gauge(
    "http_requests_in_progress",
    "Number of HTTP requests in progress",
    ["method", "endpoint"]
)

TASK_COUNT = Counter(
    "webagent_tasks_total",
    "Total number of WebAgent tasks",
    ["status"]
)

TASK_DURATION = Histogram(
    "webagent_task_duration_seconds",
    "Duration of WebAgent tasks in seconds",
    ["status"],
    buckets=(1, 5, 10, 30, 60, 120, 300, 600, 1800, 3600)
)

LLM_REQUEST_COUNT = Counter(
    "llm_requests_total",
    "Total number of LLM requests",
    ["provider", "model", "status"]
)

LLM_REQUEST_LATENCY = Histogram(
    "llm_request_duration_seconds",
    "LLM request latency in seconds",
    ["provider", "model"],
    buckets=(0.1, 0.5, 1, 3, 5, 10, 30, 60, 120)
)

LLM_TOKEN_COUNT = Counter(
    "llm_tokens_total",
    "Total number of tokens processed by LLMs",
    ["provider", "model", "type"]  # type = prompt or completion
)

# Middleware for tracking HTTP requests
class PrometheusMiddleware(BaseHTTPMiddleware):
    """Middleware that collects Prometheus metrics for HTTP requests."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        method = request.method
        path = request.url.path
        
        # Skip metrics endpoint to avoid recursion
        if path == "/api/v1/metrics":
            return await call_next(request)
        
        # Track requests in progress
        REQUESTS_IN_PROGRESS.labels(method=method, endpoint=path).inc()
        
        # Track request latency
        start_time = time.time()
        
        try:
            response = await call_next(request)
            
            # Record request metrics
            status_code = str(response.status_code)
            REQUEST_COUNT.labels(
                method=method, endpoint=path, status_code=status_code
            ).inc()
            
            # Record request latency
            duration = time.time() - start_time
            REQUEST_LATENCY.labels(method=method, endpoint=path).observe(duration)
            
            return response
        except Exception as e:
            # Record error metrics
            REQUEST_COUNT.labels(
                method=method, endpoint=path, status_code="500"
            ).inc()
            raise
        finally:
            # Track requests in progress (decrement)
            REQUESTS_IN_PROGRESS.labels(method=method, endpoint=path).dec()

# Decorator for tracking LLM requests
def track_llm_request(provider: str, model: str):
    """Decorator for tracking LLM API calls."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Track LLM request count and latency
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                
                # Record successful request
                LLM_REQUEST_COUNT.labels(
                    provider=provider, model=model, status="success"
                ).inc()
                
                # Record token usage if available
                if hasattr(result, "usage") and result.usage:
                    LLM_TOKEN_COUNT.labels(
                        provider=provider, model=model, type="prompt"
                    ).inc(result.usage.get("prompt_tokens", 0))
                    
                    LLM_TOKEN_COUNT.labels(
                        provider=provider, model=model, type="completion"
                    ).inc(result.usage.get("completion_tokens", 0))
                
                return result
            except Exception as e:
                # Record failed request
                LLM_REQUEST_COUNT.labels(
                    provider=provider, model=model, status="error"
                ).inc()
                raise
            finally:
                # Record request latency
                duration = time.time() - start_time
                LLM_REQUEST_LATENCY.labels(
                    provider=provider, model=model
                ).observe(duration)
        return wrapper
    return decorator

# Task tracking functions
def track_task_start(task_id: str):
    """Track the start of a task."""
    TASK_COUNT.labels(status="started").inc()

def track_task_completion(task_id: str, duration: float, status: str = "completed"):
    """Track the completion of a task."""
    TASK_COUNT.labels(status=status).inc()
    TASK_DURATION.labels(status=status).observe(duration)

# Setup metrics endpoint for FastAPI
def setup_metrics(app: FastAPI):
    """Setup metrics endpoint for FastAPI application."""
    
    @app.get("/api/v1/metrics")
    async def metrics():
        """Prometheus metrics endpoint."""
        return Response(
            content=generate_latest(REGISTRY),
            media_type="text/plain"
        ) 