"""
Observability Module
===================

This module provides comprehensive observability tools for monitoring
the WebAgent application's performance, health, and accuracy.

Features:
- Metrics collection and reporting for ML model performance
- Data quality monitoring
- Drift detection for model inputs and outputs
- Latency and throughput tracking
- Resource utilization monitoring
- Custom Prometheus metrics
- Integration with MLflow for experiment tracking
"""

import time
import logging
import os
import json
import psutil
import threading
from typing import Dict, Any, List, Optional, Union, Callable
from functools import wraps
from datetime import datetime
import traceback

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from prometheus_client import Counter, Gauge, Histogram, Summary

from app.core.logger import logger
from app.core.metrics import WEBAGENT_REGISTRY
from app.core.config import settings

# ML Model metrics
MODEL_PREDICTION_COUNT = Counter(
    "ml_model_predictions_total",
    "Total number of model predictions",
    ["model_name", "task_type", "status"],
    registry=WEBAGENT_REGISTRY
)

MODEL_LATENCY = Histogram(
    "ml_model_latency_seconds",
    "Model prediction latency in seconds",
    ["model_name", "task_type"],
    buckets=(0.01, 0.05, 0.1, 0.5, 1, 2.5, 5, 10, 30, 60),
    registry=WEBAGENT_REGISTRY
)

# Knowledge extraction metrics
ENTITY_EXTRACTION_COUNT = Counter(
    "entity_extraction_total",
    "Total number of named entities extracted",
    ["entity_type"],
    registry=WEBAGENT_REGISTRY
)

RELATION_EXTRACTION_COUNT = Counter(
    "relation_extraction_total",
    "Total number of relations extracted",
    ["relation_type"],
    registry=WEBAGENT_REGISTRY
)

EXTRACTION_CONFIDENCE = Histogram(
    "extraction_confidence",
    "Confidence scores for extractions",
    ["extraction_type"],
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99),
    registry=WEBAGENT_REGISTRY
)

# Document processing metrics
DOCUMENT_PROCESSING_COUNT = Counter(
    "document_processing_total",
    "Total number of documents processed",
    ["status", "doc_type"],
    registry=WEBAGENT_REGISTRY
)

DOCUMENT_PROCESSING_LATENCY = Histogram(
    "document_processing_latency_seconds",
    "Document processing latency in seconds",
    ["doc_type"],
    buckets=(0.1, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300, 600),
    registry=WEBAGENT_REGISTRY
)

DOCUMENT_SIZE = Histogram(
    "document_size_bytes",
    "Document size in bytes",
    ["doc_type"],
    buckets=(10, 100, 1024, 10*1024, 100*1024, 1024*1024, 10*1024*1024),
    registry=WEBAGENT_REGISTRY
)

# Vector database metrics
VECTORDB_QUERY_COUNT = Counter(
    "vectordb_query_total",
    "Total number of vector database queries",
    ["status"],
    registry=WEBAGENT_REGISTRY
)

VECTORDB_QUERY_LATENCY = Histogram(
    "vectordb_query_latency_seconds",
    "Vector database query latency in seconds",
    [],
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5),
    registry=WEBAGENT_REGISTRY
)

# Data quality metrics
DATA_QUALITY_SCORE = Gauge(
    "data_quality_score",
    "Data quality score (0-100)",
    ["dataset", "metric_type"],
    registry=WEBAGENT_REGISTRY
)

DATA_DRIFT_SCORE = Gauge(
    "data_drift_score",
    "Data drift score (0-1, higher means more drift)",
    ["component"],
    registry=WEBAGENT_REGISTRY
)

# Citation tracking metrics
CITATION_COUNT = Counter(
    "citation_total",
    "Total number of citations/references tracked",
    ["status"],
    registry=WEBAGENT_REGISTRY
)

# Hardware utilization
GPU_MEMORY_USAGE = Gauge(
    "gpu_memory_usage_bytes",
    "GPU memory usage in bytes",
    ["device"],
    registry=WEBAGENT_REGISTRY
)

GPU_UTILIZATION = Gauge(
    "gpu_utilization_percent",
    "GPU utilization percentage",
    ["device"],
    registry=WEBAGENT_REGISTRY
)

# ================ Decorator functions ================

def track_model_prediction(model_name: str, task_type: str):
    """Decorator to track model prediction metrics."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                MODEL_PREDICTION_COUNT.labels(
                    model_name=model_name,
                    task_type=task_type,
                    status="success"
                ).inc()
                return result
            except Exception as e:
                MODEL_PREDICTION_COUNT.labels(
                    model_name=model_name,
                    task_type=task_type,
                    status="error"
                ).inc()
                raise e
            finally:
                MODEL_LATENCY.labels(
                    model_name=model_name,
                    task_type=task_type
                ).observe(time.time() - start_time)
        return wrapper
    return decorator

def track_document_processing(doc_type: str):
    """Decorator to track document processing metrics."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                DOCUMENT_PROCESSING_COUNT.labels(
                    status="success", 
                    doc_type=doc_type
                ).inc()
                return result
            except Exception as e:
                DOCUMENT_PROCESSING_COUNT.labels(
                    status="error", 
                    doc_type=doc_type
                ).inc()
                raise e
            finally:
                DOCUMENT_PROCESSING_LATENCY.labels(
                    doc_type=doc_type
                ).observe(time.time() - start_time)
        return wrapper
    return decorator

def track_vectordb_query():
    """Decorator to track vector database query metrics."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                VECTORDB_QUERY_COUNT.labels(status="success").inc()
                return result
            except Exception as e:
                VECTORDB_QUERY_COUNT.labels(status="error").inc()
                raise e
            finally:
                VECTORDB_QUERY_LATENCY.observe(time.time() - start_time)
        return wrapper
    return decorator

# ================ Utility functions ================

def record_entity_extraction(entity_type: str, count: int = 1):
    """Record extraction of named entities."""
    ENTITY_EXTRACTION_COUNT.labels(entity_type=entity_type).inc(count)

def record_relation_extraction(relation_type: str, count: int = 1):
    """Record extraction of relations."""
    RELATION_EXTRACTION_COUNT.labels(relation_type=relation_type).inc(count)

def record_extraction_confidence(extraction_type: str, confidence: float):
    """Record confidence score for extraction."""
    EXTRACTION_CONFIDENCE.labels(extraction_type=extraction_type).observe(confidence)

def record_document_size(doc_type: str, size_bytes: int):
    """Record document size."""
    DOCUMENT_SIZE.labels(doc_type=doc_type).observe(size_bytes)

def record_data_quality(dataset: str, metric_type: str, score: float):
    """Record data quality score."""
    DATA_QUALITY_SCORE.labels(dataset=dataset, metric_type=metric_type).set(score)

def record_data_drift(component: str, score: float):
    """Record data drift score."""
    DATA_DRIFT_SCORE.labels(component=component).set(score)

def record_citation(status: str = "success"):
    """Record citation/reference tracking."""
    CITATION_COUNT.labels(status=status).inc()

# ================ GPU monitoring ================

def start_gpu_monitoring(interval_seconds: int = 30):
    """
    Start GPU monitoring in a background thread.
    Requires pytorch and nvidia-ml-py3 to be installed.
    
    Args:
        interval_seconds: Interval between GPU metrics collection
    """
    try:
        import torch
        if not torch.cuda.is_available():
            logger.warning("CUDA not available for GPU monitoring")
            return None
        
        try:
            import pynvml
            pynvml.nvmlInit()
        except (ImportError, Exception) as e:
            logger.warning(f"Could not initialize NVML for GPU monitoring: {e}")
            return None
        
        def monitor_gpu():
            while True:
                try:
                    device_count = pynvml.nvmlDeviceGetCount()
                    
                    for i in range(device_count):
                        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        
                        # Set memory usage
                        GPU_MEMORY_USAGE.labels(device=f"gpu{i}").set(memory_info.used)
                        
                        # Set utilization
                        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        GPU_UTILIZATION.labels(device=f"gpu{i}").set(utilization.gpu)
                        
                except Exception as e:
                    logger.error(f"Error monitoring GPU: {e}")
                
                time.sleep(interval_seconds)
        
        thread = threading.Thread(target=monitor_gpu, daemon=True)
        thread.start()
        logger.info(f"GPU monitoring started with {interval_seconds}s interval")
        return thread
    
    except ImportError:
        logger.warning("PyTorch or NVML not available for GPU monitoring")
        return None

# ================ MLflow integration ================

def log_to_mlflow(metrics: Optional[Dict[str, float]] = None,
                 params: Optional[Dict[str, Any]] = None,
                 tags: Optional[Dict[str, str]] = None,
                 artifacts: Optional[Dict[str, str]] = None,
                 experiment_name: Optional[str] = None):
    """
    Log metrics, parameters, and artifacts to MLflow.
    
    Args:
        metrics: Dictionary of metrics to log
        params: Dictionary of parameters to log
        tags: Dictionary of tags to log
        artifacts: Dictionary of artifacts to log (path -> artifact_name)
        experiment_name: Name of MLflow experiment
    """
    if not MLFLOW_AVAILABLE:
        logger.warning("MLflow not available for logging")
        return
    
    try:
        # Use the tracking URI from settings
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        logger.debug(f"Using MLflow tracking URI: {settings.MLFLOW_TRACKING_URI}")
        
        # Set experiment if provided
        if experiment_name:
            mlflow.set_experiment(experiment_name)
        
        # Check if a run is already active
        if mlflow.active_run():
            # Use existing run
            if metrics:
                mlflow.log_metrics(metrics)
            
            if params:
                mlflow.log_params(params)
            
            if tags:
                for key, value in tags.items():
                    mlflow.set_tag(key, value)
            
            if artifacts:
                for path, name in artifacts.items():
                    mlflow.log_artifact(path, name)
        else:
            # Start a new run
            with mlflow.start_run():
                # Log metrics
                if metrics:
                    mlflow.log_metrics(metrics)
                
                # Log params
                if params:
                    mlflow.log_params(params)
                
                # Log tags
                if tags:
                    for key, value in tags.items():
                        mlflow.set_tag(key, value)
                
                # Log artifacts
                if artifacts:
                    for path, name in artifacts.items():
                        mlflow.log_artifact(path, name)
    except Exception as e:
        logger.warning(f"Failed to log to MLflow: {str(e)}", exc_info=True)

# ================ Summary metrics ================

def get_observability_stats() -> Dict[str, Any]:
    """
    Get current observability statistics.
    
    Returns:
        Dictionary with observability metrics
    """
    stats = {
        "ml_model": {
            "predictions": {
                "total": sum([c.value for c in MODEL_PREDICTION_COUNT._metrics.values()]),
                "success_rate": _calculate_success_rate(MODEL_PREDICTION_COUNT, "status")
            },
            "latency": {
                "avg": _get_histogram_average(MODEL_LATENCY)
            }
        },
        "knowledge_extraction": {
            "entities": {
                "total": sum([c.value for c in ENTITY_EXTRACTION_COUNT._metrics.values()]),
                "by_type": {l["entity_type"]: c.value for c, l in ENTITY_EXTRACTION_COUNT._metrics.items()}
            },
            "relations": {
                "total": sum([c.value for c in RELATION_EXTRACTION_COUNT._metrics.values()]),
                "by_type": {l["relation_type"]: c.value for c, l in RELATION_EXTRACTION_COUNT._metrics.items()}
            },
            "confidence": {
                "avg": _get_histogram_average(EXTRACTION_CONFIDENCE)
            }
        },
        "document_processing": {
            "total": sum([c.value for c in DOCUMENT_PROCESSING_COUNT._metrics.values()]),
            "success_rate": _calculate_success_rate(DOCUMENT_PROCESSING_COUNT, "status"),
            "latency": {
                "avg": _get_histogram_average(DOCUMENT_PROCESSING_LATENCY)
            },
            "size": {
                "avg": _get_histogram_average(DOCUMENT_SIZE)
            }
        },
        "vectordb": {
            "queries": {
                "total": sum([c.value for c in VECTORDB_QUERY_COUNT._metrics.values()]),
                "success_rate": _calculate_success_rate(VECTORDB_QUERY_COUNT, "status")
            },
            "latency": {
                "avg": _get_histogram_average(VECTORDB_QUERY_LATENCY)
            }
        },
        "citations": {
            "total": sum([c.value for c in CITATION_COUNT._metrics.values()]),
            "success_rate": _calculate_success_rate(CITATION_COUNT, "status")
        }
    }
    
    return stats

def _calculate_success_rate(counter, label_name):
    """Helper to calculate success rate from a counter with success/error labels."""
    total = 0
    success = 0
    
    for metric, labels in counter._metrics.items():
        value = getattr(metric, '_value', 0) if not hasattr(metric, '_type') else 0
        total += value
        if labels.get(label_name) == "success":
            success += value
    
    if total > 0:
        return round((success / total) * 100, 2)
    return 100.0

def _get_histogram_average(histogram):
    """Calculate average value from a Prometheus histogram."""
    total_sum = 0
    total_count = 0
    
    try:
        for metric in histogram._metrics.values():
            if hasattr(metric, 'sum') and hasattr(metric, 'sample_count'):
                total_sum += metric.sum
                total_count += metric.sample_count
        
        if total_count > 0:
            return round(total_sum / total_count, 4)
        return None
    except Exception as e:
        logger.warning(f"Error calculating histogram average: {str(e)}")
        return None 