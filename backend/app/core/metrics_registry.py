"""
Metrics Registry
===============

A centralized registry for tracking and reporting metrics related to
ML models, data pipelines, and system performance. This module provides
a unified interface for recording metrics that can be exposed to 
monitoring systems or logged to MLflow.
"""

import logging
import time
import threading
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import statistics

from app.core.logger import logger
from app.core.observability import log_to_mlflow

class ModelMetricsRegistry:
    """Registry for tracking ML model performance metrics."""
    
    def __init__(self):
        self.model_metrics = {}
        self.prediction_counts = {}
        self.latency_samples = {}
        self.confidence_scores = {}
        self.lock = threading.RLock()
        
    def register_model(self, model_id: str, model_type: str, task_type: str, description: str = None):
        """
        Register a model to track metrics for.
        
        Args:
            model_id: Unique identifier for the model
            model_type: Type of model (e.g., "llm", "embedding", "ner")
            task_type: Type of task the model performs
            description: Optional description of the model
        """
        with self.lock:
            if model_id not in self.model_metrics:
                self.model_metrics[model_id] = {
                    "model_id": model_id,
                    "model_type": model_type,
                    "task_type": task_type,
                    "description": description,
                    "created_at": datetime.now().isoformat(),
                    "predictions": 0,
                    "errors": 0,
                    "avg_latency": 0,
                    "avg_confidence": 0,
                    "last_updated": datetime.now().isoformat(),
                }
                self.prediction_counts[model_id] = {"success": 0, "error": 0}
                self.latency_samples[model_id] = []
                self.confidence_scores[model_id] = []
                logger.info(f"Registered model {model_id} of type {model_type} for task {task_type}")
    
    def record_prediction(self, model_id: str, latency: float = None, success: bool = True, 
                         confidence: float = None, metadata: Dict[str, Any] = None):
        """
        Record metrics for a model prediction.
        
        Args:
            model_id: Identifier for the model
            latency: Prediction latency in seconds
            success: Whether the prediction was successful
            confidence: Confidence score (0-1) if available
            metadata: Additional metadata about the prediction
        """
        with self.lock:
            if model_id not in self.model_metrics:
                logger.warning(f"Unregistered model {model_id}. Register before recording metrics.")
                return
            
            # Update prediction counts
            if success:
                self.prediction_counts[model_id]["success"] += 1
            else:
                self.prediction_counts[model_id]["error"] += 1
            
            # Update latency metrics
            if latency is not None:
                self.latency_samples[model_id].append(latency)
                # Keep only the last 1000 samples to avoid memory bloat
                if len(self.latency_samples[model_id]) > 1000:
                    self.latency_samples[model_id] = self.latency_samples[model_id][-1000:]
            
            # Update confidence scores
            if confidence is not None:
                self.confidence_scores[model_id].append(confidence)
                # Keep only the last 1000 samples
                if len(self.confidence_scores[model_id]) > 1000:
                    self.confidence_scores[model_id] = self.confidence_scores[model_id][-1000:]
            
            # Update the summary metrics
            total = self.prediction_counts[model_id]["success"] + self.prediction_counts[model_id]["error"]
            
            avg_latency = 0
            if self.latency_samples[model_id]:
                avg_latency = sum(self.latency_samples[model_id]) / len(self.latency_samples[model_id])
            
            avg_confidence = 0
            if self.confidence_scores[model_id]:
                avg_confidence = sum(self.confidence_scores[model_id]) / len(self.confidence_scores[model_id])
            
            # Update the metrics
            self.model_metrics[model_id].update({
                "predictions": total,
                "errors": self.prediction_counts[model_id]["error"],
                "success_rate": (self.prediction_counts[model_id]["success"] / max(1, total)) * 100,
                "avg_latency": avg_latency,
                "avg_confidence": avg_confidence,
                "last_updated": datetime.now().isoformat()
            })
            
            # Include metadata if provided
            if metadata:
                if "metadata" not in self.model_metrics[model_id]:
                    self.model_metrics[model_id]["metadata"] = []
                
                # Add timestamp to metadata
                metadata_with_timestamp = {
                    "timestamp": datetime.now().isoformat(),
                    **metadata
                }
                
                # Keep only the last 100 metadata entries
                self.model_metrics[model_id]["metadata"] = \
                    (self.model_metrics[model_id]["metadata"] + [metadata_with_timestamp])[-100:]
    
    def get_model_metrics(self, model_id: str) -> Dict[str, Any]:
        """
        Get metrics for a specific model.
        
        Args:
            model_id: Identifier for the model
            
        Returns:
            Dictionary with metrics for the model
        """
        with self.lock:
            if model_id not in self.model_metrics:
                logger.warning(f"No metrics for model {model_id}")
                return {}
            
            metrics = self.model_metrics[model_id].copy()
            
            # Calculate additional statistical metrics if we have enough data
            if len(self.latency_samples[model_id]) >= 5:
                latency_samples = self.latency_samples[model_id]
                metrics["latency"] = {
                    "avg": metrics["avg_latency"],
                    "min": min(latency_samples),
                    "max": max(latency_samples),
                    "median": statistics.median(latency_samples),
                    "p95": statistics.quantiles(latency_samples, n=20)[18] if len(latency_samples) >= 20 else None,
                    "p99": statistics.quantiles(latency_samples, n=100)[98] if len(latency_samples) >= 100 else None
                }
            
            if len(self.confidence_scores[model_id]) >= 5:
                confidence_samples = self.confidence_scores[model_id]
                metrics["confidence"] = {
                    "avg": metrics["avg_confidence"],
                    "min": min(confidence_samples),
                    "max": max(confidence_samples),
                    "median": statistics.median(confidence_samples),
                    "p95": statistics.quantiles(confidence_samples, n=20)[18] if len(confidence_samples) >= 20 else None,
                    "p99": statistics.quantiles(confidence_samples, n=100)[98] if len(confidence_samples) >= 100 else None
                }
            
            return metrics
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get metrics for all registered models.
        
        Returns:
            Dictionary with metrics for all models
        """
        with self.lock:
            return {model_id: self.get_model_metrics(model_id) for model_id in self.model_metrics}
    
    def log_to_mlflow_experiment(self, experiment_name: str = "model_metrics"):
        """
        Log metrics for all models to MLflow.
        
        Args:
            experiment_name: Name of the MLflow experiment
        """
        all_metrics = self.get_all_metrics()
        
        for model_id, metrics in all_metrics.items():
            mlflow_metrics = {
                f"{model_id}_predictions": metrics.get("predictions", 0),
                f"{model_id}_success_rate": metrics.get("success_rate", 0),
                f"{model_id}_avg_latency": metrics.get("avg_latency", 0),
                f"{model_id}_avg_confidence": metrics.get("avg_confidence", 0),
            }
            
            params = {
                "model_id": model_id,
                "model_type": metrics.get("model_type", "unknown"),
                "task_type": metrics.get("task_type", "unknown"),
            }
            
            log_to_mlflow(
                metrics=mlflow_metrics,
                params=params,
                experiment_name=experiment_name
            )

class PipelineMetricsRegistry:
    """Registry for tracking data pipeline performance metrics."""
    
    def __init__(self):
        self.pipeline_metrics = {}
        self.processing_counts = {}
        self.latency_samples = {}
        self.lock = threading.RLock()
    
    def register_pipeline(self, pipeline_id: str, pipeline_type: str, description: str = None):
        """
        Register a pipeline to track metrics for.
        
        Args:
            pipeline_id: Unique identifier for the pipeline
            pipeline_type: Type of pipeline
            description: Optional description of the pipeline
        """
        with self.lock:
            if pipeline_id not in self.pipeline_metrics:
                self.pipeline_metrics[pipeline_id] = {
                    "pipeline_id": pipeline_id,
                    "pipeline_type": pipeline_type,
                    "description": description,
                    "created_at": datetime.now().isoformat(),
                    "jobs_processed": 0,
                    "failures": 0,
                    "avg_latency": 0,
                    "last_updated": datetime.now().isoformat(),
                }
                self.processing_counts[pipeline_id] = {"success": 0, "error": 0}
                self.latency_samples[pipeline_id] = []
                logger.info(f"Registered pipeline {pipeline_id} of type {pipeline_type}")
    
    def record_job(self, pipeline_id: str, latency: float = None, success: bool = True, 
                  items_processed: int = 1, metadata: Dict[str, Any] = None):
        """
        Record metrics for a pipeline job.
        
        Args:
            pipeline_id: Identifier for the pipeline
            latency: Job processing latency in seconds
            success: Whether the job was successful
            items_processed: Number of items processed
            metadata: Additional metadata about the job
        """
        with self.lock:
            if pipeline_id not in self.pipeline_metrics:
                logger.warning(f"Unregistered pipeline {pipeline_id}. Register before recording metrics.")
                return
            
            # Update processing counts
            if success:
                self.processing_counts[pipeline_id]["success"] += items_processed
            else:
                self.processing_counts[pipeline_id]["error"] += items_processed
            
            # Update latency metrics
            if latency is not None:
                self.latency_samples[pipeline_id].append(latency)
                # Keep only the last 1000 samples
                if len(self.latency_samples[pipeline_id]) > 1000:
                    self.latency_samples[pipeline_id] = self.latency_samples[pipeline_id][-1000:]
            
            # Update the summary metrics
            total = self.processing_counts[pipeline_id]["success"] + self.processing_counts[pipeline_id]["error"]
            
            avg_latency = 0
            if self.latency_samples[pipeline_id]:
                avg_latency = sum(self.latency_samples[pipeline_id]) / len(self.latency_samples[pipeline_id])
            
            # Update the metrics
            self.pipeline_metrics[pipeline_id].update({
                "jobs_processed": total,
                "failures": self.processing_counts[pipeline_id]["error"],
                "success_rate": (self.processing_counts[pipeline_id]["success"] / max(1, total)) * 100,
                "avg_latency": avg_latency,
                "last_updated": datetime.now().isoformat()
            })
            
            # Include metadata if provided
            if metadata:
                if "metadata" not in self.pipeline_metrics[pipeline_id]:
                    self.pipeline_metrics[pipeline_id]["metadata"] = []
                
                # Add timestamp to metadata
                metadata_with_timestamp = {
                    "timestamp": datetime.now().isoformat(),
                    **metadata
                }
                
                # Keep only the last 100 metadata entries
                self.pipeline_metrics[pipeline_id]["metadata"] = \
                    (self.pipeline_metrics[pipeline_id]["metadata"] + [metadata_with_timestamp])[-100:]
    
    def get_pipeline_metrics(self, pipeline_id: str) -> Dict[str, Any]:
        """
        Get metrics for a specific pipeline.
        
        Args:
            pipeline_id: Identifier for the pipeline
            
        Returns:
            Dictionary with metrics for the pipeline
        """
        with self.lock:
            if pipeline_id not in self.pipeline_metrics:
                logger.warning(f"No metrics for pipeline {pipeline_id}")
                return {}
            
            metrics = self.pipeline_metrics[pipeline_id].copy()
            
            # Calculate additional statistical metrics if we have enough data
            if len(self.latency_samples[pipeline_id]) >= 5:
                latency_samples = self.latency_samples[pipeline_id]
                metrics["latency"] = {
                    "avg": metrics["avg_latency"],
                    "min": min(latency_samples),
                    "max": max(latency_samples),
                    "median": statistics.median(latency_samples),
                    "p95": statistics.quantiles(latency_samples, n=20)[18] if len(latency_samples) >= 20 else None,
                    "p99": statistics.quantiles(latency_samples, n=100)[98] if len(latency_samples) >= 100 else None
                }
            
            return metrics
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get metrics for all registered pipelines.
        
        Returns:
            Dictionary with metrics for all pipelines
        """
        with self.lock:
            return {pipeline_id: self.get_pipeline_metrics(pipeline_id) for pipeline_id in self.pipeline_metrics}
    
    def log_to_mlflow_experiment(self, experiment_name: str = "pipeline_metrics"):
        """
        Log metrics for all pipelines to MLflow.
        
        Args:
            experiment_name: Name of the MLflow experiment
        """
        all_metrics = self.get_all_metrics()
        
        for pipeline_id, metrics in all_metrics.items():
            mlflow_metrics = {
                f"{pipeline_id}_jobs_processed": metrics.get("jobs_processed", 0),
                f"{pipeline_id}_success_rate": metrics.get("success_rate", 0),
                f"{pipeline_id}_avg_latency": metrics.get("avg_latency", 0),
            }
            
            params = {
                "pipeline_id": pipeline_id,
                "pipeline_type": metrics.get("pipeline_type", "unknown"),
            }
            
            log_to_mlflow(
                metrics=mlflow_metrics,
                params=params,
                experiment_name=experiment_name
            )

class DataQualityRegistry:
    """Registry for tracking data quality metrics."""
    
    def __init__(self):
        self.dataset_metrics = {}
        self.quality_scores = {}
        self.lock = threading.RLock()
    
    def register_dataset(self, dataset_id: str, dataset_type: str, description: str = None):
        """
        Register a dataset to track quality metrics for.
        
        Args:
            dataset_id: Unique identifier for the dataset
            dataset_type: Type of dataset
            description: Optional description of the dataset
        """
        with self.lock:
            if dataset_id not in self.dataset_metrics:
                self.dataset_metrics[dataset_id] = {
                    "dataset_id": dataset_id,
                    "dataset_type": dataset_type,
                    "description": description,
                    "created_at": datetime.now().isoformat(),
                    "data_points": 0,
                    "quality_score": 0,
                    "last_updated": datetime.now().isoformat(),
                }
                self.quality_scores[dataset_id] = []
                logger.info(f"Registered dataset {dataset_id} of type {dataset_type}")
    
    def record_quality_metric(self, dataset_id: str, metric_name: str, score: float, 
                             data_points: int = 1, metadata: Dict[str, Any] = None):
        """
        Record data quality metrics.
        
        Args:
            dataset_id: Identifier for the dataset
            metric_name: Name of the quality metric
            score: Quality score (0-100)
            data_points: Number of data points evaluated
            metadata: Additional metadata about the quality metric
        """
        with self.lock:
            if dataset_id not in self.dataset_metrics:
                logger.warning(f"Unregistered dataset {dataset_id}. Register before recording metrics.")
                return
            
            # Update quality scores
            self.quality_scores[dataset_id].append(score)
            # Keep only the last 1000 samples
            if len(self.quality_scores[dataset_id]) > 1000:
                self.quality_scores[dataset_id] = self.quality_scores[dataset_id][-1000:]
            
            # Update the summary metrics
            avg_quality = sum(self.quality_scores[dataset_id]) / len(self.quality_scores[dataset_id]) if self.quality_scores[dataset_id] else 0
            
            # Update the metrics
            self.dataset_metrics[dataset_id].update({
                "data_points": self.dataset_metrics[dataset_id].get("data_points", 0) + data_points,
                "quality_score": avg_quality,
                "last_updated": datetime.now().isoformat()
            })
            
            # Store metric-specific data
            metric_key = f"metric_{metric_name}"
            if metric_key not in self.dataset_metrics[dataset_id]:
                self.dataset_metrics[dataset_id][metric_key] = {"scores": [], "avg": 0}
            
            self.dataset_metrics[dataset_id][metric_key]["scores"].append(score)
            # Keep only the last 100 scores
            if len(self.dataset_metrics[dataset_id][metric_key]["scores"]) > 100:
                self.dataset_metrics[dataset_id][metric_key]["scores"] = self.dataset_metrics[dataset_id][metric_key]["scores"][-100:]
            
            self.dataset_metrics[dataset_id][metric_key]["avg"] = sum(self.dataset_metrics[dataset_id][metric_key]["scores"]) / len(self.dataset_metrics[dataset_id][metric_key]["scores"])
            
            # Include metadata if provided
            if metadata:
                if "metadata" not in self.dataset_metrics[dataset_id]:
                    self.dataset_metrics[dataset_id]["metadata"] = []
                
                # Add timestamp and metric name to metadata
                metadata_with_timestamp = {
                    "timestamp": datetime.now().isoformat(),
                    "metric_name": metric_name,
                    "score": score,
                    **metadata
                }
                
                # Keep only the last 100 metadata entries
                self.dataset_metrics[dataset_id]["metadata"] = \
                    (self.dataset_metrics[dataset_id]["metadata"] + [metadata_with_timestamp])[-100:]
    
    def get_dataset_metrics(self, dataset_id: str) -> Dict[str, Any]:
        """
        Get metrics for a specific dataset.
        
        Args:
            dataset_id: Identifier for the dataset
            
        Returns:
            Dictionary with metrics for the dataset
        """
        with self.lock:
            if dataset_id not in self.dataset_metrics:
                logger.warning(f"No metrics for dataset {dataset_id}")
                return {}
            
            metrics = self.dataset_metrics[dataset_id].copy()
            
            # Calculate additional statistical metrics if we have enough data
            if len(self.quality_scores[dataset_id]) >= 5:
                quality_scores = self.quality_scores[dataset_id]
                metrics["quality"] = {
                    "avg": metrics["quality_score"],
                    "min": min(quality_scores),
                    "max": max(quality_scores),
                    "median": statistics.median(quality_scores),
                    "p95": statistics.quantiles(quality_scores, n=20)[18] if len(quality_scores) >= 20 else None,
                    "p99": statistics.quantiles(quality_scores, n=100)[98] if len(quality_scores) >= 100 else None
                }
            
            return metrics
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get metrics for all registered datasets.
        
        Returns:
            Dictionary with metrics for all datasets
        """
        with self.lock:
            return {dataset_id: self.get_dataset_metrics(dataset_id) for dataset_id in self.dataset_metrics}
    
    def log_to_mlflow_experiment(self, experiment_name: str = "data_quality_metrics"):
        """
        Log metrics for all datasets to MLflow.
        
        Args:
            experiment_name: Name of the MLflow experiment
        """
        all_metrics = self.get_all_metrics()
        
        for dataset_id, metrics in all_metrics.items():
            mlflow_metrics = {
                f"{dataset_id}_quality_score": metrics.get("quality_score", 0),
                f"{dataset_id}_data_points": metrics.get("data_points", 0),
            }
            
            # Add metric-specific scores
            for key, value in metrics.items():
                if key.startswith("metric_") and isinstance(value, dict) and "avg" in value:
                    metric_name = key[7:]  # Remove "metric_" prefix
                    mlflow_metrics[f"{dataset_id}_{metric_name}"] = value["avg"]
            
            params = {
                "dataset_id": dataset_id,
                "dataset_type": metrics.get("dataset_type", "unknown"),
            }
            
            log_to_mlflow(
                metrics=mlflow_metrics,
                params=params,
                experiment_name=experiment_name
            )

# Create global instances
_model_metrics_registry = None
_pipeline_metrics_registry = None
_data_quality_registry = None

def get_model_metrics_registry() -> ModelMetricsRegistry:
    """Get the global model metrics registry."""
    global _model_metrics_registry
    if _model_metrics_registry is None:
        _model_metrics_registry = ModelMetricsRegistry()
    return _model_metrics_registry

def get_pipeline_metrics_registry() -> PipelineMetricsRegistry:
    """Get the global pipeline metrics registry."""
    global _pipeline_metrics_registry
    if _pipeline_metrics_registry is None:
        _pipeline_metrics_registry = PipelineMetricsRegistry()
    return _pipeline_metrics_registry

def get_data_quality_registry() -> DataQualityRegistry:
    """Get the global data quality registry."""
    global _data_quality_registry
    if _data_quality_registry is None:
        _data_quality_registry = DataQualityRegistry()
    return _data_quality_registry

def get_all_metrics() -> Dict[str, Any]:
    """
    Get metrics from all registries.
    
    Returns:
        Dictionary with all metrics
    """
    return {
        "model_metrics": get_model_metrics_registry().get_all_metrics(),
        "pipeline_metrics": get_pipeline_metrics_registry().get_all_metrics(),
        "data_quality_metrics": get_data_quality_registry().get_all_metrics(),
        "timestamp": datetime.now().isoformat()
    }

def log_all_metrics_to_mlflow():
    """Log all metrics to MLflow."""
    get_model_metrics_registry().log_to_mlflow_experiment()
    get_pipeline_metrics_registry().log_to_mlflow_experiment()
    get_data_quality_registry().log_to_mlflow_experiment() 