#!/usr/bin/env python3
"""
ML Observability Demo
====================

This demo showcases the ML observability features of the WebAgent platform,
including metrics collection, drift detection, and monitoring.

The demo simulates model predictions and data processing, tracks performance 
metrics over time, and illustrates how drift detection works in real-time.
"""

import time
import random
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import threading
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import WebAgent components
from backend.app.core.metrics_registry import (
    get_model_metrics_registry,
    get_pipeline_metrics_registry,
    get_data_quality_registry,
    get_all_metrics
)
from backend.app.core.drift_detection import (
    create_numerical_drift_detector,
    create_categorical_drift_detector,
    create_embedding_drift_detector,
    create_performance_drift_detector,
    get_all_drift_stats,
    should_retrain_any_model
)
from backend.app.core.logger import logger
from backend.app.core.mlflow_integration import (
    get_mlflow_manager,
    log_model_metrics,
    log_drift_detection,
    log_visualization,
    log_metrics_export
)
from backend.app.core.grafana_integration import (
    GrafanaDashboardGenerator,
    generate_model_drift_dashboard,
    generate_model_performance_dashboard,
    upload_dashboard_to_grafana
)

# Configure matplotlib for non-interactive mode if needed
plt.switch_backend('agg')

class ObservabilityDemo:
    """Demonstrates the observability features of the WebAgent platform."""
    
    def __init__(self, output_dir: str = None):
        """
        Initialize the observability demo.
        
        Args:
            output_dir: Directory to save output files (default: demos/results)
        """
        self.output_dir = output_dir or os.path.join(os.path.dirname(__file__), "results")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Get metric registries
        self.model_registry = get_model_metrics_registry()
        self.pipeline_registry = get_pipeline_metrics_registry()
        self.data_quality_registry = get_data_quality_registry()
        
        # Initialize drift detectors
        self.latency_detector = create_numerical_drift_detector("demo_latency", window_size=100, threshold=0.05)
        self.accuracy_detector = create_numerical_drift_detector("demo_accuracy", window_size=100, threshold=0.05)
        self.category_detector = create_categorical_drift_detector("demo_category", window_size=100, threshold=0.05)
        self.performance_detector = create_performance_drift_detector("demo_performance", window_size=50, threshold=0.1)
        
        # Initialize MLflow
        self.mlflow_manager = get_mlflow_manager()
        if self.mlflow_manager.initialized:
            self.mlflow_run_id = self.mlflow_manager.start_run(experiment_name="MLObservabilityDemo")
            logger.info(f"Started MLflow run with ID: {self.mlflow_run_id}")
        else:
            self.mlflow_run_id = None
            logger.warning("MLflow not initialized, metrics will not be logged to MLflow")
        
        # Initialize Grafana dashboard generator
        self.grafana_generator = GrafanaDashboardGenerator()
        
        # Track metrics over time for plotting
        self.metrics_history = {
            "timestamps": [],
            "latency": [],
            "accuracy": [],
            "token_usage": [],
            "error_rate": [],
            "drift_scores": []
        }
        
        # Simulation parameters
        self.normal_latency_mean = 0.2  # 200ms
        self.normal_latency_std = 0.05
        self.normal_accuracy_mean = 0.95
        self.normal_accuracy_std = 0.02
        
        # Drift parameters (will be changed during drift simulation)
        self.drift_latency_mean = 0.2
        self.drift_latency_std = 0.05
        self.drift_accuracy_mean = 0.95
        self.drift_accuracy_std = 0.02
        
        # Register models, pipelines, and datasets
        self._register_components()
        
        logger.info("Initialized ML Observability Demo")
    
    def _register_components(self):
        """Register models, pipelines, and datasets for demonstration."""
        # Register models
        self.model_registry.register_model(
            model_id="gpt4-demo", 
            model_type="llm", 
            task_type="text-generation",
            description="Demo LLM model for testing observability"
        )
        
        self.model_registry.register_model(
            model_id="embedding-demo", 
            model_type="embedding", 
            task_type="text-embedding",
            description="Demo embedding model for testing observability"
        )
        
        # Register pipelines
        self.pipeline_registry.register_pipeline(
            pipeline_id="doc-processing-demo",
            pipeline_type="document-processing",
            description="Demo document processing pipeline"
        )
        
        self.pipeline_registry.register_pipeline(
            pipeline_id="data-extraction-demo",
            pipeline_type="data-extraction",
            description="Demo data extraction pipeline"
        )
        
        # Register datasets
        self.data_quality_registry.register_dataset(
            dataset_id="documents-demo",
            dataset_type="text",
            description="Demo document dataset"
        )
        
        self.data_quality_registry.register_dataset(
            dataset_id="embeddings-demo",
            dataset_type="vector",
            description="Demo embedding dataset"
        )
        
        logger.info("Registered demo components in metrics registries")
        
        # Log component registrations to MLflow if available
        if self.mlflow_run_id:
            self.mlflow_manager.log_parameters({
                "model_ids": "gpt4-demo,embedding-demo",
                "pipeline_ids": "doc-processing-demo,data-extraction-demo",
                "dataset_ids": "documents-demo,embeddings-demo",
                "demo_type": "ml_observability"
            })
            
            logger.info("Logged component registrations to MLflow")
    
    def _generate_normal_prediction(self) -> Tuple[float, float, int, int, bool]:
        """
        Generate normal prediction metrics.
        
        Returns:
            Tuple of (latency, accuracy, prompt_tokens, completion_tokens, success)
        """
        latency = max(0.01, np.random.normal(self.normal_latency_mean, self.normal_latency_std))
        accuracy = min(1.0, max(0.5, np.random.normal(self.normal_accuracy_mean, self.normal_accuracy_std)))
        prompt_tokens = np.random.randint(10, 100)
        completion_tokens = np.random.randint(50, 500)
        success = random.random() < 0.98  # 98% success rate
        
        return latency, accuracy, prompt_tokens, completion_tokens, success
    
    def _generate_drift_prediction(self) -> Tuple[float, float, int, int, bool]:
        """
        Generate prediction metrics with drift.
        
        Returns:
            Tuple of (latency, accuracy, prompt_tokens, completion_tokens, success)
        """
        latency = max(0.01, np.random.normal(self.drift_latency_mean, self.drift_latency_std))
        accuracy = min(1.0, max(0.5, np.random.normal(self.drift_accuracy_mean, self.drift_accuracy_std)))
        prompt_tokens = np.random.randint(10, 100)
        completion_tokens = np.random.randint(50, 500)
        success = random.random() < 0.9  # 90% success rate during drift
        
        return latency, accuracy, prompt_tokens, completion_tokens, success
    
    def _generate_embedding(self, dim: int = 384) -> np.ndarray:
        """
        Generate a random embedding vector.
        
        Args:
            dim: Dimension of the embedding vector
            
        Returns:
            Random embedding vector
        """
        return np.random.normal(0, 1, dim).astype(np.float32)
    
    def simulate_normal_operations(self, num_samples: int = 100, sleep_time: float = 0.05):
        """
        Simulate normal model operations.
        
        Args:
            num_samples: Number of prediction samples to generate
            sleep_time: Time to sleep between samples (in seconds)
        """
        logger.info(f"Simulating {num_samples} normal operations")
        
        # Collect reference samples for drift detection
        for i in range(50):  # Use first 50 samples as reference
            latency, accuracy, prompt_tokens, completion_tokens, success = self._generate_normal_prediction()
            
            # Add reference samples for drift detection
            self.latency_detector.add_reference_sample(latency)
            self.accuracy_detector.add_reference_sample(accuracy)
            self.category_detector.add_reference_sample("normal" if success else "error")
            
            # Add performance metrics
            self.performance_detector.add_performance_metrics(
                accuracy=accuracy, 
                latency=latency, 
                confidence=random.uniform(0.8, 0.95)
            )
            
            # Log to MLflow if available
            if self.mlflow_run_id:
                self.mlflow_manager.log_metrics({
                    "reference_latency": latency,
                    "reference_accuracy": accuracy,
                    "reference_success": 1.0 if success else 0.0
                })
            
            # Sleep to simulate real-time operation
            time.sleep(sleep_time)
        
        # Simulate normal predictions
        for i in range(num_samples):
            latency, accuracy, prompt_tokens, completion_tokens, success = self._generate_normal_prediction()
            
            # Record model prediction
            self.model_registry.record_prediction(
                model_id="gpt4-demo",
                latency=latency,
                success=success,
                confidence=accuracy,
                metadata={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens
                }
            )
            
            # Record pipeline job
            self.pipeline_registry.record_job(
                pipeline_id="doc-processing-demo",
                latency=latency * 1.5,  # Pipeline is slower than model
                success=success,
                items_processed=random.randint(1, 5)
            )
            
            # Record data quality
            self.data_quality_registry.record_quality_metric(
                dataset_id="documents-demo",
                metric_name="completeness",
                score=random.uniform(90, 99),
                data_points=random.randint(10, 50)
            )
            
            # Check for drift
            latency_score, latency_drift = self.latency_detector.add_current_sample(latency)
            accuracy_score, accuracy_drift = self.accuracy_detector.add_current_sample(accuracy)
            category_score, category_drift = self.category_detector.add_current_sample("normal" if success else "error")
            
            # Update performance metrics
            perf_score, perf_drift = self.performance_detector.add_performance_metrics(
                accuracy=accuracy, 
                latency=latency, 
                confidence=random.uniform(0.8, 0.95)
            )
            
            # Log to MLflow if available
            if self.mlflow_run_id:
                self.mlflow_manager.log_metrics({
                    "latency": latency,
                    "accuracy": accuracy,
                    "success": 1.0 if success else 0.0,
                    "tokens": prompt_tokens + completion_tokens,
                    "latency_drift_score": latency_score,
                    "accuracy_drift_score": accuracy_score,
                    "category_drift_score": category_score,
                    "performance_drift_score": perf_score
                })
                
                # Log drift detection events
                if any([latency_drift, accuracy_drift, category_drift, perf_drift]):
                    log_drift_detection(
                        detector_id="normal_operation",
                        scores={
                            "latency": latency_score,
                            "accuracy": accuracy_score,
                            "category": category_score,
                            "performance": perf_score
                        },
                        detected=True
                    )
            
            # Store metrics for plotting
            self.metrics_history["timestamps"].append(datetime.now())
            self.metrics_history["latency"].append(latency)
            self.metrics_history["accuracy"].append(accuracy)
            self.metrics_history["token_usage"].append(prompt_tokens + completion_tokens)
            self.metrics_history["error_rate"].append(0 if success else 1)
            self.metrics_history["drift_scores"].append(max(latency_score, accuracy_score, category_score, perf_score))
            
            if i % 10 == 0:
                logger.info(f"Processed {i} normal samples")
            
            # Sleep to simulate real-time operation
            time.sleep(sleep_time)
        
        logger.info("Completed normal operations simulation")
    
    def simulate_gradual_drift(self, num_samples: int = 100, sleep_time: float = 0.05):
        """
        Simulate gradual drift in model performance.
        
        Args:
            num_samples: Number of prediction samples to generate
            sleep_time: Time to sleep between samples (in seconds)
        """
        logger.info(f"Simulating gradual drift over {num_samples} samples")
        
        # Gradually increase latency and decrease accuracy
        latency_increment = 0.3 / num_samples  # Gradually increase to +300ms
        accuracy_decrement = 0.2 / num_samples  # Gradually decrease by 20%
        
        # Log drift simulation parameters to MLflow
        if self.mlflow_run_id:
            self.mlflow_manager.log_parameters({
                "drift_simulation_samples": num_samples,
                "latency_increment_total": 0.3,
                "accuracy_decrement_total": 0.2
            })
        
        for i in range(num_samples):
            # Update drift parameters
            drift_factor = i / num_samples
            self.drift_latency_mean = self.normal_latency_mean + (drift_factor * 0.3)
            self.drift_latency_std = self.normal_latency_std + (drift_factor * 0.1)
            self.drift_accuracy_mean = self.normal_accuracy_mean - (drift_factor * 0.2)
            self.drift_accuracy_std = self.normal_accuracy_std + (drift_factor * 0.05)
            
            # Generate drifted prediction
            latency, accuracy, prompt_tokens, completion_tokens, success = self._generate_drift_prediction()
            
            # Record model prediction
            self.model_registry.record_prediction(
                model_id="gpt4-demo",
                latency=latency,
                success=success,
                confidence=accuracy,
                metadata={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens
                }
            )
            
            # Record pipeline job
            self.pipeline_registry.record_job(
                pipeline_id="doc-processing-demo",
                latency=latency * 1.5,
                success=success,
                items_processed=random.randint(1, 5)
            )
            
            # Record data quality (quality decreases with drift)
            quality_score = 95 - (drift_factor * 15)  # Decrease from 95 to 80
            self.data_quality_registry.record_quality_metric(
                dataset_id="documents-demo",
                metric_name="completeness",
                score=quality_score,
                data_points=random.randint(10, 50)
            )
            
            # Check for drift
            latency_score, latency_drift = self.latency_detector.add_current_sample(latency)
            accuracy_score, accuracy_drift = self.accuracy_detector.add_current_sample(accuracy)
            category_score, category_drift = self.category_detector.add_current_sample("normal" if success else "error")
            
            # Update performance metrics
            confidence = max(0.6, 0.9 - (drift_factor * 0.3))  # Decrease from 0.9 to 0.6
            perf_score, perf_drift = self.performance_detector.add_performance_metrics(
                accuracy=accuracy, 
                latency=latency, 
                confidence=confidence
            )
            
            # Log to MLflow if available
            if self.mlflow_run_id:
                self.mlflow_manager.log_metrics({
                    "drift_latency": latency,
                    "drift_accuracy": accuracy,
                    "drift_success": 1.0 if success else 0.0,
                    "drift_tokens": prompt_tokens + completion_tokens,
                    "drift_quality_score": quality_score,
                    "drift_confidence": confidence,
                    "drift_factor": drift_factor,
                    "drift_latency_score": latency_score,
                    "drift_accuracy_score": accuracy_score,
                    "drift_category_score": category_score,
                    "drift_performance_score": perf_score
                })
                
                # Log drift detection events
                if any([latency_drift, accuracy_drift, category_drift, perf_drift]):
                    log_drift_detection(
                        detector_id="gradual_drift",
                        scores={
                            "latency": latency_score,
                            "accuracy": accuracy_score,
                            "category": category_score,
                            "performance": perf_score
                        },
                        detected=True
                    )
            
            # Store metrics for plotting
            self.metrics_history["timestamps"].append(datetime.now())
            self.metrics_history["latency"].append(latency)
            self.metrics_history["accuracy"].append(accuracy)
            self.metrics_history["token_usage"].append(prompt_tokens + completion_tokens)
            self.metrics_history["error_rate"].append(0 if success else 1)
            self.metrics_history["drift_scores"].append(max(latency_score, accuracy_score, category_score, perf_score))
            
            # Check if retraining is recommended
            if i % 10 == 0:
                retrain = self.performance_detector.should_retrain()
                if retrain:
                    logger.warning(f"Drift detected at sample {i}. Model retraining recommended!")
                    # Log retraining recommendation to MLflow
                    if self.mlflow_run_id:
                        self.mlflow_manager.log_metrics({
                            "retraining_recommended": 1.0,
                            "retraining_sample": float(i)
                        })
                
                logger.info(f"Processed {i} drift samples. Current drift scores - Latency: {latency_score:.4f}, Accuracy: {accuracy_score:.4f}")
            
            # Sleep to simulate real-time operation
            time.sleep(sleep_time)
        
        logger.info("Completed gradual drift simulation")
    
    def simulate_embedding_drift(self, num_samples: int = 100, sleep_time: float = 0.05):
        """
        Simulate drift in embedding vectors.
        
        Args:
            num_samples: Number of embedding samples to generate
            sleep_time: Time to sleep between samples (in seconds)
        """
        logger.info(f"Simulating embedding drift over {num_samples} samples")
        
        # Create embedding drift detector
        embedding_detector = create_embedding_drift_detector("demo_embedding", window_size=50, threshold=0.3)
        
        # Log embedding simulation parameters to MLflow
        if self.mlflow_run_id:
            self.mlflow_manager.log_parameters({
                "embedding_simulation_samples": num_samples,
                "embedding_dimension": 384,
                "embedding_drift_threshold": embedding_detector.threshold
            })
        
        # Generate reference embeddings
        reference_centroid = np.zeros(384)
        for i in range(50):
            # Generate embedding with consistent distribution
            embedding = self._generate_embedding()
            reference_centroid += embedding / 50
            
            # Add reference sample
            embedding_detector.add_reference_sample(embedding)
            
            # Sleep to simulate real-time operation
            time.sleep(sleep_time / 2)
        
        # Simulate gradual drift in embeddings by shifting the distribution
        drift_direction = np.random.normal(0, 1, 384)
        drift_direction = drift_direction / np.linalg.norm(drift_direction)  # Normalize to unit vector
        
        drift_magnitudes = []
        drift_scores = []
        drift_detected_flags = []
        
        for i in range(num_samples):
            # Gradually increase drift magnitude
            drift_magnitude = (i / num_samples) * 1.5  # 0 to 1.5
            drift_magnitudes.append(drift_magnitude)
            
            # Generate drifted embedding
            embedding = self._generate_embedding() + (drift_direction * drift_magnitude)
            
            # Check for drift
            drift_score, drift_detected = embedding_detector.add_current_sample(embedding)
            drift_scores.append(drift_score)
            drift_detected_flags.append(1 if drift_detected else 0)
            
            # Record embedding
            self.model_registry.record_prediction(
                model_id="embedding-demo",
                latency=random.uniform(0.01, 0.05),
                success=True,
                confidence=0.9,
                metadata={"drift_score": drift_score, "drift_detected": drift_detected}
            )
            
            # Log to MLflow if available
            if self.mlflow_run_id:
                self.mlflow_manager.log_metrics({
                    "embedding_drift_score": drift_score,
                    "embedding_drift_magnitude": drift_magnitude,
                    "embedding_drift_detected": 1.0 if drift_detected else 0.0
                })
                
                # Log drift detection events
                if drift_detected:
                    log_drift_detection(
                        detector_id="embedding_drift",
                        scores={"embedding": drift_score},
                        detected=True
                    )
            
            if i % 10 == 0 or drift_detected:
                if drift_detected:
                    logger.warning(f"Embedding drift detected at sample {i}. Drift score: {drift_score:.4f}")
                else:
                    logger.info(f"Processed {i} embedding samples. Current drift score: {drift_score:.4f}")
            
            # Sleep to simulate real-time operation
            time.sleep(sleep_time)
        
        # Plot embedding drift
        plt.figure(figsize=(10, 6))
        plt.plot(range(num_samples), drift_scores, label='Drift Score', color='blue')
        plt.plot(range(num_samples), drift_magnitudes, label='Drift Magnitude', color='green', linestyle='--')
        
        # Highlight drift detections
        for i, flag in enumerate(drift_detected_flags):
            if flag:
                plt.axvline(x=i, color='red', alpha=0.3, linestyle=':')
        
        plt.axhline(y=embedding_detector.threshold, color='red', linestyle='--', label='Threshold')
        plt.title('Embedding Drift Detection')
        plt.xlabel('Sample')
        plt.ylabel('Drift Score / Magnitude')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        plt.tight_layout()
        embedding_drift_path = os.path.join(self.output_dir, 'embedding_drift.png')
        plt.savefig(embedding_drift_path)
        
        # Log to MLflow if available
        if self.mlflow_run_id:
            log_visualization(plt.gcf(), "embedding_drift.png")
        
        plt.close()
        
        logger.info("Completed embedding drift simulation")
    
    def simulate_categorical_drift(self, num_samples: int = 150, sleep_time: float = 0.05):
        """
        Simulate drift in categorical distributions.
        
        Args:
            num_samples: Number of categorical samples to generate
            sleep_time: Time to sleep between samples (in seconds)
        """
        logger.info(f"Simulating categorical drift over {num_samples} samples")
        
        # Create categorical drift detector
        category_detector = create_categorical_drift_detector("demo_doc_types", window_size=50, threshold=0.05)
        
        # Log categorical simulation parameters to MLflow
        if self.mlflow_run_id:
            self.mlflow_manager.log_parameters({
                "categorical_simulation_samples": num_samples,
                "categorical_drift_threshold": category_detector.threshold,
                "categories": "pdf,docx,txt,html,csv"
            })
        
        # Initial distribution of document types
        categories = ["pdf", "docx", "txt", "html", "csv"]
        initial_probs = [0.4, 0.3, 0.15, 0.1, 0.05]  # Initial probability distribution
        
        # Generate reference samples
        for i in range(50):
            category = np.random.choice(categories, p=initial_probs)
            category_detector.add_reference_sample(category)
            
            # Record category distribution
            self.data_quality_registry.record_quality_metric(
                dataset_id="documents-demo",
                metric_name=f"doc_type_{category}",
                score=95,  # Quality score
                data_points=1
            )
            
            # Sleep to simulate real-time operation
            time.sleep(sleep_time / 2)
        
        # Gradually shift distribution
        drift_probs = [0.1, 0.15, 0.05, 0.6, 0.1]  # Target drifted distribution (more HTML)
        
        # Log target distribution to MLflow
        if self.mlflow_run_id:
            self.mlflow_manager.log_parameters({
                "initial_distribution": str(initial_probs),
                "target_distribution": str(drift_probs)
            })
        
        category_counts = {cat: 0 for cat in categories}
        drift_scores = []
        drift_detected_flags = []
        
        for i in range(num_samples):
            # Interpolate between initial and drift distributions
            drift_factor = min(1.0, i / (num_samples * 0.7))  # Complete shift at 70% of samples
            current_probs = [(1 - drift_factor) * p1 + drift_factor * p2 
                             for p1, p2 in zip(initial_probs, drift_probs)]
            
            # Normalize to ensure probabilities sum to 1
            current_probs = [p / sum(current_probs) for p in current_probs]
            
            # Generate category from current distribution
            category = np.random.choice(categories, p=current_probs)
            category_counts[category] += 1
            
            # Check for drift
            drift_score, drift_detected = category_detector.add_current_sample(category)
            drift_scores.append(drift_score)
            drift_detected_flags.append(1 if drift_detected else 0)
            
            # Record category distribution
            self.data_quality_registry.record_quality_metric(
                dataset_id="documents-demo",
                metric_name=f"doc_type_{category}",
                score=90,  # Quality score (slightly lower during drift)
                data_points=1,
                metadata={"drift_score": drift_score, "drift_detected": drift_detected}
            )
            
            # Record pipeline processing for this document type
            self.pipeline_registry.record_job(
                pipeline_id="doc-processing-demo",
                latency=random.uniform(0.1, 0.5),
                success=random.random() < 0.95,
                items_processed=1,
                metadata={"doc_type": category}
            )
            
            # Log to MLflow if available
            if self.mlflow_run_id:
                # Create category frequency metrics
                category_metrics = {f"category_{cat}_freq": category_counts[cat] / sum(category_counts.values()) 
                                   for cat in categories}
                
                # Add drift metrics
                category_metrics.update({
                    "category_drift_score": drift_score,
                    "category_drift_detected": 1.0 if drift_detected else 0.0,
                    "category_drift_factor": drift_factor
                })
                
                self.mlflow_manager.log_metrics(category_metrics)
                
                # Log drift detection events
                if drift_detected:
                    log_drift_detection(
                        detector_id="category_drift",
                        scores={"category": drift_score},
                        detected=True
                    )
            
            if i % 10 == 0 or drift_detected:
                if drift_detected:
                    logger.warning(f"Category distribution drift detected at sample {i}. Drift score: {drift_score:.4f}")
                else:
                    logger.info(f"Processed {i} categorical samples. Current drift score: {drift_score:.4f}")
            
            # Sleep to simulate real-time operation
            time.sleep(sleep_time)
        
        # Plot category distribution
        plt.figure(figsize=(12, 10))
        
        # Plot 1: Drift scores
        plt.subplot(2, 1, 1)
        plt.plot(range(num_samples), drift_scores, label='Drift Score', color='blue')
        
        # Highlight drift detections
        for i, flag in enumerate(drift_detected_flags):
            if flag:
                plt.axvline(x=i, color='red', alpha=0.3, linestyle=':')
        
        plt.axhline(y=category_detector.threshold, color='red', linestyle='--', label='Threshold')
        plt.title('Categorical Drift Detection')
        plt.xlabel('Sample')
        plt.ylabel('Drift Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Category distribution
        plt.subplot(2, 1, 2)
        category_percentages = {cat: count / sum(category_counts.values()) * 100 
                               for cat, count in category_counts.items()}
        
        bars = plt.bar(categories, [category_counts[cat] for cat in categories], color='skyblue')
        
        # Add percentage labels
        for bar, category in zip(bars, categories):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{category_percentages[category]:.1f}%',
                    ha='center', va='bottom', rotation=0)
        
        plt.title('Document Type Distribution')
        plt.xlabel('Document Type')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Save the plot
        plt.tight_layout()
        category_drift_path = os.path.join(self.output_dir, 'category_drift.png')
        plt.savefig(category_drift_path)
        
        # Log to MLflow if available
        if self.mlflow_run_id:
            log_visualization(plt.gcf(), "category_drift.png")
            
            # Log final category distribution as parameters
            self.mlflow_manager.log_parameters({
                f"final_{cat}_percentage": f"{category_percentages[cat]:.1f}%" 
                for cat in categories
            })
        
        plt.close()
        
        logger.info("Completed categorical drift simulation")
    
    def plot_metrics(self):
        """Plot metrics collected during the simulation."""
        logger.info("Generating metrics plots")
        
        # Convert timestamps to relative seconds for better plotting
        start_time = self.metrics_history["timestamps"][0]
        relative_times = [(ts - start_time).total_seconds() for ts in self.metrics_history["timestamps"]]
        
        # Figure 1: Latency and Accuracy
        plt.figure(figsize=(12, 10))
        
        # Plot 1: Latency
        plt.subplot(2, 1, 1)
        plt.plot(relative_times, self.metrics_history["latency"], label='Latency (s)', color='blue')
        plt.title('Model Latency Over Time')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Latency (s)')
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Accuracy
        plt.subplot(2, 1, 2)
        plt.plot(relative_times, self.metrics_history["accuracy"], label='Accuracy', color='green')
        plt.title('Model Accuracy Over Time')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Accuracy')
        plt.ylim(0.5, 1.0)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        model_perf_path = os.path.join(self.output_dir, 'model_performance.png')
        plt.savefig(model_perf_path)
        
        # Log to MLflow if available
        if self.mlflow_run_id:
            log_visualization(plt.gcf(), "model_performance.png")
        plt.close()
        
        # Figure 2: Drift Score and Error Rate
        plt.figure(figsize=(12, 10))
        
        # Plot 1: Drift Score
        plt.subplot(2, 1, 1)
        plt.plot(relative_times, self.metrics_history["drift_scores"], label='Drift Score', color='red')
        plt.title('Drift Score Over Time')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Drift Score')
        plt.ylim(0, 1.0)
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Error Rate (moving average)
        plt.subplot(2, 1, 2)
        
        # Calculate moving average for error rate
        window_size = 20
        error_rate_ma = []
        for i in range(len(self.metrics_history["error_rate"])):
            start_idx = max(0, i - window_size + 1)
            window = self.metrics_history["error_rate"][start_idx:i+1]
            error_rate_ma.append(sum(window) / len(window) if window else 0)
        
        plt.plot(relative_times, error_rate_ma, label='Error Rate (Moving Avg)', color='orange')
        plt.title('Error Rate Over Time (Moving Average)')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Error Rate')
        plt.ylim(0, 0.5)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        drift_error_path = os.path.join(self.output_dir, 'drift_error_rate.png')
        plt.savefig(drift_error_path)
        
        # Log to MLflow if available
        if self.mlflow_run_id:
            log_visualization(plt.gcf(), "drift_error_rate.png")
        plt.close()
        
        # Figure 3: Token Usage
        plt.figure(figsize=(10, 6))
        plt.plot(relative_times, self.metrics_history["token_usage"], label='Token Usage', color='purple')
        plt.title('Token Usage Over Time')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Tokens')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        token_usage_path = os.path.join(self.output_dir, 'token_usage.png')
        plt.savefig(token_usage_path)
        
        # Log to MLflow if available
        if self.mlflow_run_id:
            log_visualization(plt.gcf(), "token_usage.png")
        plt.close()
        
        # Generate Grafana dashboards
        try:
            # Create model performance dashboard
            perf_dashboard_path = os.path.join(self.output_dir, 'model_performance_dashboard.json')
            generate_model_performance_dashboard(
                output_file=perf_dashboard_path,
                title="WebAgent ML Performance Dashboard",
                folder="MLObservabilityDemo"
            )
            
            # Create model drift dashboard
            drift_dashboard_path = os.path.join(self.output_dir, 'model_drift_dashboard.json')
            generate_model_drift_dashboard(
                output_file=drift_dashboard_path,
                title="WebAgent ML Drift Dashboard",
                folder="MLObservabilityDemo"
            )
            
            # Upload dashboards to Grafana if configured
            grafana_url = os.environ.get("GRAFANA_URL")
            grafana_api_key = os.environ.get("GRAFANA_API_KEY")
            
            if grafana_url and grafana_api_key:
                upload_dashboard_to_grafana(
                    dashboard_file=perf_dashboard_path,
                    grafana_url=grafana_url,
                    api_key=grafana_api_key,
                    folder="MLObservabilityDemo"
                )
                
                upload_dashboard_to_grafana(
                    dashboard_file=drift_dashboard_path,
                    grafana_url=grafana_url,
                    api_key=grafana_api_key,
                    folder="MLObservabilityDemo"
                )
                
                logger.info("Uploaded dashboards to Grafana")
            else:
                logger.warning("Grafana URL or API key not set, skipping dashboard upload")
                
        except Exception as e:
            logger.error(f"Error generating Grafana dashboards: {str(e)}")
        
        logger.info(f"Plots saved to {self.output_dir}")
    
    def export_metrics(self):
        """Export metrics to JSON for analysis."""
        # Get all metrics
        all_metrics = get_all_metrics()
        
        # Get drift stats
        drift_stats = get_all_drift_stats()
        
        # Get retraining recommendations
        retraining_dict = should_retrain_any_model()
        
        # Helper function to recursively convert NumPy types to Python native types
        def convert_to_json_safe(obj):
            """Convert NumPy types to Python native types recursively."""
            if hasattr(obj, 'item'):
                return obj.item()  # Convert NumPy scalars to Python scalars
            elif isinstance(obj, dict):
                return {k: convert_to_json_safe(v) for k, v in obj.items()}
            elif isinstance(obj, list) or isinstance(obj, tuple):
                return [convert_to_json_safe(i) for i in obj]
            elif isinstance(obj, (np.integer, np.floating, np.bool_)):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        # Convert all objects to JSON-safe objects
        all_metrics = convert_to_json_safe(all_metrics)
        drift_stats = convert_to_json_safe(drift_stats)
        retraining_dict = convert_to_json_safe(retraining_dict)
        
        # Combine and export
        export_data = {
            "metrics": all_metrics,
            "drift_stats": drift_stats,
            "retraining_recommendations": retraining_dict,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save to file
        export_path = os.path.join(self.output_dir, "observability_metrics.json")
        with open(export_path, "w") as f:
            json.dump(export_data, f, indent=2)
        
        # Log export data to MLflow if available
        if self.mlflow_run_id:
            log_metrics_export(export_data, "observability_metrics.json")
        
        logger.info(f"Metrics exported to {export_path}")
        return export_data
    
    def run_demo(self, skip_plots: bool = False):
        """
        Run the complete observability demo.
        
        Args:
            skip_plots: Skip generating plots
        """
        logger.info("Starting ML observability demo")
        
        try:
            # Simulate normal operations
            self.simulate_normal_operations(num_samples=100, sleep_time=0.02)
            
            # Simulate gradual drift
            self.simulate_gradual_drift(num_samples=150, sleep_time=0.02)
            
            # Simulate embedding drift
            self.simulate_embedding_drift(num_samples=100, sleep_time=0.02)
            
            # Simulate categorical drift
            self.simulate_categorical_drift(num_samples=100, sleep_time=0.02)
            
            # Generate plots
            if not skip_plots:
                self.plot_metrics()
            
            # Export metrics
            metrics_data = self.export_metrics()
            
            # Print summary
            self._print_summary(metrics_data)
            
            # End MLflow run if started
            if self.mlflow_run_id:
                self.mlflow_manager.end_run()
                logger.info(f"Ended MLflow run: {self.mlflow_run_id}")
            
            logger.info("ML observability demo completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error running observability demo: {str(e)}", exc_info=True)
            
            # End MLflow run if started
            if self.mlflow_run_id:
                self.mlflow_manager.end_run()
                logger.warning(f"Ended MLflow run due to error: {self.mlflow_run_id}")
            
            return False
    
    def _print_summary(self, metrics_data: Dict[str, Any]):
        """
        Print a summary of the metrics and drift detection.
        
        Args:
            metrics_data: Metrics data exported from the export_metrics method
        """
        print("\n" + "="*80)
        print("ML OBSERVABILITY DEMO SUMMARY")
        print("="*80)
        
        # Print model metrics
        print("\nMODEL METRICS:")
        model_metrics = metrics_data["metrics"].get("model_metrics", {})
        for model_id, metrics in model_metrics.items():
            print(f"  - {model_id}:")
            print(f"    - Predictions: {metrics.get('predictions', 0)}")
            print(f"    - Success Rate: {metrics.get('success_rate', 0):.2f}%")
            print(f"    - Avg Latency: {metrics.get('avg_latency', 0):.4f}s")
            print(f"    - Avg Confidence: {metrics.get('avg_confidence', 0):.4f}")
        
        # Print pipeline metrics
        print("\nPIPELINE METRICS:")
        pipeline_metrics = metrics_data["metrics"].get("pipeline_metrics", {})
        for pipeline_id, metrics in pipeline_metrics.items():
            print(f"  - {pipeline_id}:")
            print(f"    - Jobs Processed: {metrics.get('jobs_processed', 0)}")
            print(f"    - Success Rate: {metrics.get('success_rate', 0):.2f}%")
            print(f"    - Avg Latency: {metrics.get('avg_latency', 0):.4f}s")
        
        # Print drift stats
        print("\nDRIFT DETECTION:")
        retrain = metrics_data.get("retraining_recommendations", {})
        for detector_name, should_retrain in retrain.items():
            status = "⚠️ RETRAINING RECOMMENDED" if should_retrain else "✓ Normal"
            print(f"  - {detector_name}: {status}")
        
        # Print data quality metrics
        print("\nDATA QUALITY METRICS:")
        data_metrics = metrics_data["metrics"].get("data_quality_metrics", {})
        for dataset_id, metrics in data_metrics.items():
            print(f"  - {dataset_id}:")
            print(f"    - Data Points: {metrics.get('data_points', 0)}")
            print(f"    - Quality Score: {metrics.get('quality_score', 0):.2f}")
        
        print("\nOBSERVABILITY ARTIFACTS:")
        print(f"  - Metrics JSON: {os.path.join(self.output_dir, 'observability_metrics.json')}")
        print(f"  - Performance Plot: {os.path.join(self.output_dir, 'model_performance.png')}")
        print(f"  - Drift Plot: {os.path.join(self.output_dir, 'drift_error_rate.png')}")
        print(f"  - Token Usage Plot: {os.path.join(self.output_dir, 'token_usage.png')}")
        print(f"  - Embedding Drift Plot: {os.path.join(self.output_dir, 'embedding_drift.png')}")
        print(f"  - Category Drift Plot: {os.path.join(self.output_dir, 'category_drift.png')}")
        
        # Print MLflow information if available
        if self.mlflow_run_id:
            print("\nMLFLOW TRACKING:")
            print(f"  - Tracking URI: {self.mlflow_manager.tracking_uri}")
            print(f"  - Run ID: {self.mlflow_run_id}")
            print(f"  - Experiment: MLObservabilityDemo")
            print("  - View in MLflow UI to explore metrics, parameters, and artifacts")
        
        # Print Grafana information if available
        grafana_url = os.environ.get("GRAFANA_URL")
        if grafana_url:
            print("\nGRAFANA DASHBOARDS:")
            print(f"  - Grafana URL: {grafana_url}")
            print(f"  - Performance Dashboard: {os.path.join(self.output_dir, 'model_performance_dashboard.json')}")
            print(f"  - Drift Dashboard: {os.path.join(self.output_dir, 'model_drift_dashboard.json')}")
            if os.environ.get("GRAFANA_API_KEY"):
                print("  - Dashboards have been uploaded to Grafana")
            else:
                print("  - Dashboards generated but not uploaded (API key not provided)")
        
        print("\n" + "="*80)


def main():
    """Main entry point for the observability demo."""
    parser = argparse.ArgumentParser(description="ML Observability Demo")
    parser.add_argument("--output-dir", type=str, default=None, 
                        help="Directory to save output files (default: demos/results)")
    parser.add_argument("--skip-plots", action="store_true", 
                        help="Skip generating plots")
    parser.add_argument("--mlflow-tracking-uri", type=str, default=None,
                        help="MLflow tracking URI (default: from MLFLOW_TRACKING_URI env var)")
    parser.add_argument("--grafana-url", type=str, default=None,
                        help="Grafana URL (default: from GRAFANA_URL env var)")
    parser.add_argument("--grafana-api-key", type=str, default=None,
                        help="Grafana API key (default: from GRAFANA_API_KEY env var)")
    args = parser.parse_args()
    
    # Set environment variables if provided
    if args.mlflow_tracking_uri:
        os.environ["MLFLOW_TRACKING_URI"] = args.mlflow_tracking_uri
    
    if args.grafana_url:
        os.environ["GRAFANA_URL"] = args.grafana_url
        
    if args.grafana_api_key:
        os.environ["GRAFANA_API_KEY"] = args.grafana_api_key
    
    demo = ObservabilityDemo(output_dir=args.output_dir)
    success = demo.run_demo(skip_plots=args.skip_plots)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main()) 