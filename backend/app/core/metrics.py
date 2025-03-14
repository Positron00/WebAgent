"""
Metrics configuration for the WebAgent backend.

This module uses the Prometheus client library to expose metrics
for monitoring the application's performance and health.
"""
import time
from functools import wraps
from typing import Callable, Dict, Optional, List, Set

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
from starlette.status import HTTP_500_INTERNAL_SERVER_ERROR

from app.core.logger import logger

# Define endpoints that should be excluded from metrics collection
METRICS_EXCLUDED_ENDPOINTS: Set[str] = {
    "/api/v1/metrics",
    "/health",
    "/api/v1/health",
    "/api/v1/health/detailed"
}

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
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30)
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

# New metrics for v2.4.1
CONFIG_VALIDATION_COUNT = Counter(
    "config_validation_total",
    "Total number of configuration validations",
    ["status", "component"]
)

API_KEY_VALIDATION_COUNT = Counter(
    "api_key_validation_total",
    "Total number of API key validations",
    ["status", "service"]
)

ERROR_COUNT = Counter(
    "error_total",
    "Total number of errors",
    ["type", "component"]
)

MEMORY_USAGE = Gauge(
    "memory_usage_bytes",
    "Memory usage in bytes",
    ["type"]
)

# Middleware for tracking HTTP requests
class PrometheusMiddleware(BaseHTTPMiddleware):
    """Middleware that collects Prometheus metrics for HTTP requests."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        method = request.method
        path = request.url.path
        
        # Clean up path for better metrics grouping
        path = self._normalize_path(path)
        
        # Skip excluded endpoints
        if path in METRICS_EXCLUDED_ENDPOINTS:
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
            status_code = str(HTTP_500_INTERNAL_SERVER_ERROR)
            REQUEST_COUNT.labels(
                method=method, endpoint=path, status_code=status_code
            ).inc()
            ERROR_COUNT.labels(
                type=type(e).__name__, component="http_middleware"
            ).inc()
            logger.error(f"Error in HTTP request: {str(e)}")
            raise
        finally:
            # Track requests in progress (decrement)
            REQUESTS_IN_PROGRESS.labels(method=method, endpoint=path).dec()
    
    def _normalize_path(self, path: str) -> str:
        """
        Normalize path to avoid high cardinality in metrics.
        
        For example:
        - /api/v1/tasks/123 becomes /api/v1/tasks/:id
        - /api/v1/users/john/profile becomes /api/v1/users/:id/profile
        """
        # List of regex patterns to normalize
        patterns = [
            # UUID pattern
            (r'/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', '/:id'),
            # Numeric IDs
            (r'/\d+', '/:id'),
            # Task IDs in specific format
            (r'/task_\d+', '/task_:id'),
        ]
        
        normalized_path = path
        for pattern, replacement in patterns:
            import re
            normalized_path = re.sub(pattern, replacement, normalized_path)
        
        return normalized_path

# Decorator for tracking LLM requests
def track_llm_request(provider: str, model: str):
    """Decorator for tracking LLM API calls."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Default values for tracking
            actual_provider = provider
            actual_model = model
            
            # Try to extract model from kwargs if available
            if 'model' in kwargs and kwargs['model']:
                actual_model = kwargs['model']
            
            # Track LLM request count and latency
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                
                # Record successful request
                LLM_REQUEST_COUNT.labels(
                    provider=actual_provider, model=actual_model, status="success"
                ).inc()
                
                # Record token usage if available
                try:
                    if hasattr(result, "usage") and result.usage:
                        # For OpenAI-like responses
                        LLM_TOKEN_COUNT.labels(
                            provider=actual_provider, model=actual_model, type="prompt"
                        ).inc(result.usage.get("prompt_tokens", 0))
                        
                        LLM_TOKEN_COUNT.labels(
                            provider=actual_provider, model=actual_model, type="completion"
                        ).inc(result.usage.get("completion_tokens", 0))
                    elif isinstance(result, dict) and "usage" in result:
                        # For raw API responses
                        LLM_TOKEN_COUNT.labels(
                            provider=actual_provider, model=actual_model, type="prompt"
                        ).inc(result["usage"].get("prompt_tokens", 0))
                        
                        LLM_TOKEN_COUNT.labels(
                            provider=actual_provider, model=actual_model, type="completion"
                        ).inc(result["usage"].get("completion_tokens", 0))
                except Exception as token_e:
                    # Don't fail the request if token counting fails
                    logger.warning(f"Error counting tokens: {str(token_e)}")
                
                return result
            except Exception as e:
                # Record failed request and error type
                LLM_REQUEST_COUNT.labels(
                    provider=actual_provider, model=actual_model, status="error"
                ).inc()
                
                ERROR_COUNT.labels(
                    type=type(e).__name__, component="llm_request"
                ).inc()
                
                logger.error(f"Error in LLM request to {actual_provider}/{actual_model}: {str(e)}")
                raise
            finally:
                # Record request latency
                duration = time.time() - start_time
                LLM_REQUEST_LATENCY.labels(
                    provider=actual_provider, model=actual_model
                ).observe(duration)
        return wrapper
    return decorator

# Task tracking functions
def track_task_start(task_id: str):
    """Track the start of a task."""
    TASK_COUNT.labels(status="started").inc()
    logger.debug(f"Started task: {task_id}")

def track_task_completion(task_id: str, duration: float, status: str = "completed"):
    """Track the completion of a task."""
    TASK_COUNT.labels(status=status).inc()
    TASK_DURATION.labels(status=status).observe(duration)
    logger.debug(f"Completed task: {task_id} with status: {status} in {duration:.2f}s")

# Configuration validation tracking
def track_config_validation(component: str, is_valid: bool):
    """Track configuration validation."""
    status = "valid" if is_valid else "invalid"
    CONFIG_VALIDATION_COUNT.labels(status=status, component=component).inc()
    
    if not is_valid:
        logger.warning(f"Invalid configuration detected for component: {component}")

# API key validation tracking
def track_api_key_validation(service: str, is_valid: bool):
    """Track API key validation."""
    status = "valid" if is_valid else "invalid"
    API_KEY_VALIDATION_COUNT.labels(status=status, service=service).inc()
    
    if not is_valid:
        logger.warning(f"Invalid API key detected for service: {service}")

# Memory usage tracking
def update_memory_usage():
    """Update memory usage metrics."""
    try:
        import psutil
        import os
        
        # Get process info
        process = psutil.Process(os.getpid())
        
        # Update memory metrics
        memory_info = process.memory_info()
        MEMORY_USAGE.labels(type="rss").set(memory_info.rss)  # Resident Set Size
        MEMORY_USAGE.labels(type="vms").set(memory_info.vms)  # Virtual Memory Size
        
        logger.debug(f"Updated memory usage metrics: RSS={memory_info.rss/1024/1024:.2f}MB, VMS={memory_info.vms/1024/1024:.2f}MB")
    except Exception as e:
        logger.warning(f"Failed to update memory usage metrics: {str(e)}")

# Setup metrics endpoint for FastAPI
def setup_metrics(app: FastAPI):
    """Setup metrics endpoint for FastAPI application."""
    
    @app.get("/api/v1/metrics")
    async def metrics():
        """Prometheus metrics endpoint."""
        # Update memory usage before serving metrics
        update_memory_usage()
        
        return Response(
            content=generate_latest(REGISTRY),
            media_type="text/plain"
        )

# Dictionary to store execution times for functions
_execution_times = {}

def timing_decorator(func):
    """
    Decorator to measure and record function execution time.
    
    Args:
        func: Function to measure
        
    Returns:
        Wrapped function that measures execution time
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Get function name
        func_name = func.__name__
        
        # Initialize stats if not already present
        if func_name not in _execution_times:
            _execution_times[func_name] = {
                "total": 0.0,
                "count": 0,
                "min": float("inf"),
                "max": 0.0,
                "average": 0.0
            }
        
        # Measure execution time
        start_time = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start_time
        
        # Update stats
        _execution_times[func_name]["total"] += duration
        _execution_times[func_name]["count"] += 1
        _execution_times[func_name]["min"] = min(_execution_times[func_name]["min"], duration)
        _execution_times[func_name]["max"] = max(_execution_times[func_name]["max"], duration)
        _execution_times[func_name]["average"] = _execution_times[func_name]["total"] / _execution_times[func_name]["count"]
        
        # Log execution time
        logger.debug(f"Function {func_name} executed in {duration:.4f} seconds")
        
        return result
    
    return wrapper

def get_metric_statistics():
    """
    Get statistics for all metrics.
    
    Returns:
        Dictionary with metric statistics
    """
    # Get execution times
    execution_times = {k: v for k, v in _execution_times.items()}
    
    # Get Prometheus metrics
    metrics = {}
    
    # Add HTTP request metrics
    try:
        metrics["http_requests"] = {
            "total": sum([c.value for c in REQUEST_COUNT._metrics.values()]),
            "in_progress": sum([g.value for g in REQUESTS_IN_PROGRESS._metrics.values()]),
            "latency": {
                "average": _get_histogram_average(REQUEST_LATENCY),
                "p50": _get_histogram_quantile(REQUEST_LATENCY, 0.5),
                "p95": _get_histogram_quantile(REQUEST_LATENCY, 0.95),
                "p99": _get_histogram_quantile(REQUEST_LATENCY, 0.99)
            }
        }
    except Exception as e:
        logger.warning(f"Error getting HTTP request metrics: {str(e)}")
        metrics["http_requests"] = {"error": str(e)}
    
    # Add LLM request metrics
    try:
        metrics["llm_requests"] = {
            "total": sum([c.value for c in LLM_REQUEST_COUNT._metrics.values()]),
            "latency": {
                "average": _get_histogram_average(LLM_REQUEST_LATENCY),
                "p50": _get_histogram_quantile(LLM_REQUEST_LATENCY, 0.5),
                "p95": _get_histogram_quantile(LLM_REQUEST_LATENCY, 0.95),
                "p99": _get_histogram_quantile(LLM_REQUEST_LATENCY, 0.99)
            },
            "tokens": {
                "prompt": sum([c.value for c, l in LLM_TOKEN_COUNT._metrics.items() if l["type"] == "prompt"]),
                "completion": sum([c.value for c, l in LLM_TOKEN_COUNT._metrics.items() if l["type"] == "completion"])
            }
        }
    except Exception as e:
        logger.warning(f"Error getting LLM request metrics: {str(e)}")
        metrics["llm_requests"] = {"error": str(e)}
    
    # Add task metrics
    try:
        metrics["tasks"] = {
            "total": sum([c.value for c in TASK_COUNT._metrics.values()]),
            "duration": {
                "average": _get_histogram_average(TASK_DURATION),
                "p50": _get_histogram_quantile(TASK_DURATION, 0.5),
                "p95": _get_histogram_quantile(TASK_DURATION, 0.95),
                "p99": _get_histogram_quantile(TASK_DURATION, 0.99)
            }
        }
    except Exception as e:
        logger.warning(f"Error getting task metrics: {str(e)}")
        metrics["tasks"] = {"error": str(e)}
    
    # Add error metrics
    try:
        metrics["errors"] = {
            "total": sum([c.value for c in ERROR_COUNT._metrics.values()]),
            "by_type": {l["type"]: c.value for c, l in ERROR_COUNT._metrics.items()}
        }
    except Exception as e:
        logger.warning(f"Error getting error metrics: {str(e)}")
        metrics["errors"] = {"error": str(e)}
    
    return {
        "execution_times": execution_times,
        "metrics": metrics
    }

def _get_histogram_average(histogram):
    """
    Calculate the average value from a Prometheus histogram.
    
    Args:
        histogram: Prometheus histogram
        
    Returns:
        Average value or None if no data
    """
    total_sum = 0
    total_count = 0
    
    for metric in histogram._metrics.values():
        total_sum += metric.sum
        total_count += metric.sample_count
    
    if total_count > 0:
        return total_sum / total_count
    else:
        return None

def _get_histogram_quantile(histogram, quantile):
    """
    Calculate a quantile from a Prometheus histogram.
    
    Args:
        histogram: Prometheus histogram
        quantile: Quantile to calculate (0.0 to 1.0)
        
    Returns:
        Quantile value or None if no data
    """
    # This is a simplified implementation
    # For accurate quantiles, use Prometheus's built-in quantile functions
    
    # Collect all observations
    observations = []
    
    for metric in histogram._metrics.values():
        # Extract bucket boundaries and counts
        buckets = sorted([(float(k.split('_')[-1]), v) for k, v in metric.buckets.items()])
        
        # Skip if no data
        if not buckets:
            continue
        
        # Add observations based on bucket counts
        for i, (upper_bound, count) in enumerate(buckets):
            if i == 0:
                # First bucket
                observations.extend([upper_bound] * count)
            else:
                # Subsequent buckets
                prev_count = buckets[i-1][1]
                observations.extend([upper_bound] * (count - prev_count))
    
    # Calculate quantile
    if observations:
        observations.sort()
        index = int(quantile * len(observations))
        return observations[index]
    else:
        return None

def log_memory_usage():
    """
    Log current memory usage.
    
    Returns:
        Dictionary with memory usage information
    """
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        # Update memory usage gauge
        MEMORY_USAGE.labels(type="rss").set(memory_info.rss)
        MEMORY_USAGE.labels(type="vms").set(memory_info.vms)
        
        memory_usage = {
            "rss": memory_info.rss / 1024 / 1024,  # MB
            "vms": memory_info.vms / 1024 / 1024,  # MB
            "percent": process.memory_percent()
        }
        
        logger.debug(f"Memory usage: {memory_usage['rss']:.2f} MB (RSS), {memory_usage['percent']:.2f}%")
        
        return memory_usage
    except ImportError:
        logger.warning("psutil not installed, cannot log memory usage")
        return {"error": "psutil not installed"}
    except Exception as e:
        logger.error(f"Error logging memory usage: {str(e)}")
        return {"error": str(e)} 