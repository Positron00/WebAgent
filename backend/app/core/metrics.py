"""
Metrics configuration for the WebAgent backend.

This module uses the Prometheus client library to expose metrics
for monitoring the application's performance and health.
"""
import time
import os
import inspect
from functools import wraps
from typing import Callable, Dict, Optional, List, Set

from fastapi import FastAPI, Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    Counter,
    Gauge,
    Histogram,
    Summary,
    CollectorRegistry,
    generate_latest,
    multiprocess,
)
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.status import HTTP_500_INTERNAL_SERVER_ERROR

from app.core.logger import logger

# Create our own registry instance to avoid conflicts
WEBAGENT_REGISTRY = CollectorRegistry()

# Define endpoints that should be excluded from metrics collection
METRICS_EXCLUDED_ENDPOINTS: Set[str] = {
    "/metrics",
    "/api/v1/metrics",
    "/health",
    "/api/v1/health",
    "/api/v1/health/detailed"
}

# Create metrics using our custom registry
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total number of HTTP requests",
    ["method", "endpoint", "status_code"],
    registry=WEBAGENT_REGISTRY
)

REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30),
    registry=WEBAGENT_REGISTRY
)

REQUESTS_IN_PROGRESS = Gauge(
    "http_requests_in_progress",
    "Number of HTTP requests in progress",
    ["method", "endpoint"],
    multiprocess_mode="livesum",  # summed over live workers when PROMETHEUS_MULTIPROC_DIR is set
    registry=WEBAGENT_REGISTRY
)

TASK_COUNT = Counter(
    "webagent_tasks_total",
    "Total number of WebAgent tasks",
    ["status"],
    registry=WEBAGENT_REGISTRY
)

TASK_DURATION = Histogram(
    "webagent_task_duration_seconds",
    "Duration of WebAgent tasks in seconds",
    ["status"],
    buckets=(1, 5, 10, 30, 60, 120, 300, 600, 1800, 3600),
    registry=WEBAGENT_REGISTRY
)

LLM_REQUEST_COUNT = Counter(
    "llm_requests_total",
    "Total number of LLM requests",
    ["provider", "model", "status"],
    registry=WEBAGENT_REGISTRY
)

LLM_REQUEST_LATENCY = Histogram(
    "llm_request_duration_seconds",
    "LLM request latency in seconds",
    ["provider", "model"],
    buckets=(0.1, 0.5, 1, 3, 5, 10, 30, 60, 120),
    registry=WEBAGENT_REGISTRY
)

LLM_TOKEN_COUNT = Counter(
    "llm_tokens_total",
    "Total number of tokens processed by LLMs",
    ["provider", "model", "type"],  # type = prompt or completion
    registry=WEBAGENT_REGISTRY
)

# New metrics for v2.4.1
CONFIG_VALIDATION_COUNT = Counter(
    "config_validation_total",
    "Total number of configuration validations",
    ["status", "component"],
    registry=WEBAGENT_REGISTRY
)

API_KEY_VALIDATION_COUNT = Counter(
    "api_key_validation_total",
    "Total number of API key validations",
    ["status", "service"],
    registry=WEBAGENT_REGISTRY
)

ERROR_COUNT = Counter(
    "error_total",
    "Total number of errors",
    ["type", "component"],
    registry=WEBAGENT_REGISTRY
)

MEMORY_USAGE = Gauge(
    "memory_usage_bytes",
    "Memory usage in bytes",
    ["type"],
    multiprocess_mode="all",  # one series per worker (`pid` label) with several workers
    registry=WEBAGENT_REGISTRY
)

TRIAGE_DECISIONS = Counter(
    "webagent_triage_decisions_total",
    "Triage routing decisions",
    ["route", "decided_by"],
    registry=WEBAGENT_REGISTRY,
)

TRIAGE_LATENCY = Histogram(
    "webagent_triage_latency_seconds",
    "Triage classify latency",
    buckets=(0.01, 0.025, 0.05, 0.075, 0.1, 0.2, 0.5, 1.0, 3.0),
    registry=WEBAGENT_REGISTRY,
)

TRIAGE_OVERRIDES = Counter(
    "webagent_triage_overrides_total",
    "Research-in-depth override clicks",
    registry=WEBAGENT_REGISTRY,
)

TRIAGE_UNCERTAIN = Counter(
    "webagent_triage_uncertain_total",
    "Model decisions near the triage threshold",
    registry=WEBAGENT_REGISTRY,
)

GUARD_DECISIONS = Counter(
    "webagent_guard_decisions_total",
    "Security guard decisions",
    ["checkpoint", "action", "outcome", "shadow"],
    registry=WEBAGENT_REGISTRY,
)

GUARD_LATENCY = Histogram(
    "webagent_guard_latency_seconds",
    "Security guard decision latency",
    ["checkpoint"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 3.0),
    registry=WEBAGENT_REGISTRY,
)

GUARD_JUDGE = Counter(
    "webagent_guard_judge_total",
    "LLM judge verdicts (checkpoint 1 user turns and checkpoint 2 tool calls)",
    ["verdict"],
    registry=WEBAGENT_REGISTRY,
)

GUARD_ESCALATIONS = Counter(
    "webagent_guard_escalations_total",
    "Security guard escalations",
    registry=WEBAGENT_REGISTRY,
)

GUARD_UNLABELLED = Counter(
    "webagent_guard_unlabelled_total",
    "Guarded LLM calls without check_hop",
    ["agent"],
    registry=WEBAGENT_REGISTRY,
)

GUARD_CONTEXT_SCORE = Histogram(
    "webagent_contextdefender_score",
    "ContextDefender axis scores",
    ["axis", "source"],
    buckets=tuple(float(x) for x in range(0, 101, 5)),
    registry=WEBAGENT_REGISTRY,
)


GUARD_SHADOW_POLICY = Counter(
    "webagent_guard_shadow_policy_total",
    "Shadow-only policies that matched (recorded, not applied)",
    ["policy_id", "action"],
    registry=WEBAGENT_REGISTRY,
)
GUARD_ALIGNMENT = Histogram(
    "webagent_guard_search_alignment",
    "Alignment (0-100) of search and retrieval queries with the task at checkpoint 2",
    ["tainted"],
    buckets=(10, 20, 30, 40, 50, 60, 70, 80, 90, 100),
    registry=WEBAGENT_REGISTRY,
)
# Whole-checkpoint time (scoring, decisions, judge), for SNFR-01: checkpoint 1 < 150 ms p95 per hop,
# checkpoint 2 < 20 ms p95 per tool call without the judge. GUARD_LATENCY times the Cedar decision only.
GUARD_HOP_SECONDS = Histogram(
    "webagent_guard_hop_seconds",
    "Checkpoint 1 time per hop (scoring and decisions)",
    ["agent"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.15, 0.25, 0.5, 1.0, 2.5, 5.0),
    registry=WEBAGENT_REGISTRY,
)
GUARD_TOOL_SECONDS = Histogram(
    "webagent_guard_tool_seconds",
    "Checkpoint 2 time per tool call (checks, alignment, decision, judge)",
    ["action"],
    buckets=(0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 3.0, 5.0),
    registry=WEBAGENT_REGISTRY,
)


def record_guard_decision(checkpoint: int, action: str, decision) -> None:
    """Record metrics for one guard decision (``decision``: :class:`Decision`)."""
    shadow_label = "true" if decision.shadow else "false"
    GUARD_DECISIONS.labels(
        checkpoint=str(checkpoint),
        action=action,
        outcome=decision.outcome,
        shadow=shadow_label,
    ).inc()
    GUARD_LATENCY.labels(checkpoint=str(checkpoint)).observe(decision.latency_ms / 1000.0)
    for policy_id in getattr(decision, "shadow_policy_ids", ()) or ():
        GUARD_SHADOW_POLICY.labels(policy_id=policy_id, action=action).inc()


def record_triage_decision(decision, threshold: Optional[float] = None) -> None:
    """Record metrics for one triage decision (decision: TriageDecision).

    `threshold` is the one the decision used; model decisions need it for the uncertain count.
    """
    TRIAGE_DECISIONS.labels(route=decision.route, decided_by=decision.decided_by).inc()
    # Mode and override decisions never run `classify`; their 0 ms would hide its real latency.
    if decision.decided_by not in ("mode", "override"):
        TRIAGE_LATENCY.observe(decision.latency_ms / 1000.0)
    if decision.decided_by == "override":
        TRIAGE_OVERRIDES.inc()
    if decision.decided_by == "model" and decision.p_agentic is not None and threshold is not None:
        from app.core.config import settings

        cfg = getattr(settings, "TRIAGE", None)
        band = cfg.uncertain_band if cfg else 0.15
        if abs(decision.p_agentic - threshold) < band:
            TRIAGE_UNCERTAIN.inc()

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
    
    # /metrics is the path Prometheus, docker-compose and the Kubernetes annotations scrape.
    @app.get("/metrics", include_in_schema=False)
    @app.get("/api/v1/metrics")
    async def metrics():
        """Prometheus metrics endpoint."""
        # Update memory usage before serving metrics
        update_memory_usage()

        return Response(content=metrics_payload(), media_type=CONTENT_TYPE_LATEST)


def mark_worker_dead() -> None:
    """Call at shutdown: live gauges (requests in progress) stop counting this worker once it exits."""
    if os.environ.get("PROMETHEUS_MULTIPROC_DIR"):
        multiprocess.mark_process_dead(os.getpid())


def metrics_payload() -> bytes:
    """All metrics. With several uvicorn workers (PROMETHEUS_MULTIPROC_DIR set, as in the image) every
    worker writes its values to that folder and this sums them; otherwise a scrape would see only the
    worker that happened to answer it."""
    if os.environ.get("PROMETHEUS_MULTIPROC_DIR"):
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        return generate_latest(registry)
    return generate_latest(WEBAGENT_REGISTRY)

# Dictionary to store execution times for functions
_execution_times = {}

def _record_execution_time(func_name: str, duration: float) -> None:
    if func_name not in _execution_times:
        _execution_times[func_name] = {
            "total": 0.0,
            "count": 0,
            "min": float("inf"),
            "max": 0.0,
            "average": 0.0,
        }
    _execution_times[func_name]["total"] += duration
    _execution_times[func_name]["count"] += 1
    _execution_times[func_name]["min"] = min(_execution_times[func_name]["min"], duration)
    _execution_times[func_name]["max"] = max(_execution_times[func_name]["max"], duration)
    _execution_times[func_name]["average"] = (
        _execution_times[func_name]["total"] / _execution_times[func_name]["count"]
    )
    logger.debug(f"Function {func_name} executed in {duration:.4f} seconds")


def timing_decorator(func):
    """
    Decorator to measure and record function execution time.
    Supports both sync and async callables (e.g. FastAPI route handlers).
    """
    if inspect.iscoroutinefunction(func):

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                return await func(*args, **kwargs)
            finally:
                _record_execution_time(func.__name__, time.perf_counter() - start_time)

        return async_wrapper

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            _record_execution_time(func.__name__, time.perf_counter() - start_time)

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
            "total": sum([getattr(c, '_value', 0) if not hasattr(c, '_type') else 0 for c in REQUEST_COUNT._metrics.values()]),
            "in_progress": sum([getattr(g, '_value', 0) if not hasattr(g, '_type') else 0 for g in REQUESTS_IN_PROGRESS._metrics.values()]),
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
            "total": sum([getattr(c, '_value', 0) if not hasattr(c, '_type') else 0 for c in LLM_REQUEST_COUNT._metrics.values()]),
            "latency": {
                "average": _get_histogram_average(LLM_REQUEST_LATENCY),
                "p50": _get_histogram_quantile(LLM_REQUEST_LATENCY, 0.5),
                "p95": _get_histogram_quantile(LLM_REQUEST_LATENCY, 0.95),
                "p99": _get_histogram_quantile(LLM_REQUEST_LATENCY, 0.99)
            },
            "tokens": {
                "prompt": sum([getattr(c, '_value', 0) if not hasattr(c, '_type') else 0 for c, l in LLM_TOKEN_COUNT._metrics.items() if l["type"] == "prompt"]),
                "completion": sum([getattr(c, '_value', 0) if not hasattr(c, '_type') else 0 for c, l in LLM_TOKEN_COUNT._metrics.items() if l["type"] == "completion"])
            }
        }
    except Exception as e:
        logger.warning(f"Error getting LLM request metrics: {str(e)}")
        metrics["llm_requests"] = {"error": str(e)}
    
    # Add task metrics
    try:
        metrics["tasks"] = {
            "total": sum([getattr(c, '_value', 0) if not hasattr(c, '_type') else 0 for c in TASK_COUNT._metrics.values()]),
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
            "total": sum([getattr(c, '_value', 0) if not hasattr(c, '_type') else 0 for c in ERROR_COUNT._metrics.values()]),
            "by_type": {l["type"]: getattr(c, '_value', 0) if not hasattr(c, '_type') else 0 for c, l in ERROR_COUNT._metrics.items()}
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
    
    try:
        for metric in histogram._metrics.values():
            if hasattr(metric, 'sum') and hasattr(metric, 'sample_count'):
                total_sum += metric.sum
                total_count += metric.sample_count
        
        if total_count > 0:
            return total_sum / total_count
        else:
            return None
    except Exception as e:
        logger.warning(f"Error calculating histogram average: {str(e)}")
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
    
    try:
        # Collect all observations
        observations = []
        
        for metric in histogram._metrics.values():
            # Check if metric has buckets attribute
            if not hasattr(metric, 'buckets'):
                continue
                
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
    except Exception as e:
        logger.warning(f"Error calculating histogram quantile: {str(e)}")
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