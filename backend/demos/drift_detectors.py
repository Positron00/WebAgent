"""
In-memory drift detectors for the observability demo (ml_observability_demo.py).

Moved here from app/core/drift_detection.py in 3.6.0: the server now computes drift from a shared
store (app/core/drift_store.py) with the pure scoring functions in app/core/drift_detection.py.
These classes keep their old behaviour for the demo only.

This module provides tools for monitoring and detecting drift in model inputs
and outputs over time. It helps identify when models may need to be retrained
or when data quality may be degrading.

Supports:
- Input distribution drift detection
- Output distribution drift detection
- Performance metric tracking over time
- Statistical significance testing for drift
"""

import logging
import time
import json
import math
import threading
from typing import Dict, List, Optional, Any, Callable, Union
from datetime import datetime
import statistics
from collections import defaultdict, deque
import numpy as np
from scipy.stats import ks_2samp, chi2_contingency

from backend.app.core.logger import logger
from backend.app.core.observability import (
    record_data_drift,
    log_to_mlflow
)

class DriftDetector:
    """Base class for drift detection."""
    
    def __init__(self, window_size: int = 1000, name: str = "drift_detector"):
        """
        Initialize the drift detector.
        
        Args:
            window_size: Size of the sliding window for reference data
            name: Name of the drift detector
        """
        self.name = name
        self.window_size = window_size
        self.reference_data = deque(maxlen=window_size)
        self.current_data = deque(maxlen=window_size)
        self.drift_scores = deque(maxlen=100)  # Store last 100 drift scores
        self.last_drift_time = None
        self.lock = threading.RLock()
        
        # Statistics
        self.total_samples = 0
        self.drift_detected_count = 0
        
        logger.info(f"Initialized drift detector: {name} with window size {window_size}")
    
    def add_reference_sample(self, sample: Any):
        """
        Add a sample to the reference distribution.
        
        Args:
            sample: The data point to add
        """
        with self.lock:
            self.reference_data.append(sample)
    
    def add_current_sample(self, sample: Any):
        """
        Add a sample to the current distribution and check for drift.
        
        Args:
            sample: The data point to add
            
        Returns:
            Drift score and whether drift was detected
        """
        with self.lock:
            self.current_data.append(sample)
            self.total_samples += 1
            
            # Only check for drift if we have enough data
            if len(self.reference_data) >= 30 and len(self.current_data) >= 30:
                score, drift_detected = self._compute_drift()
                
                # Record the drift score
                self.drift_scores.append(score)
                
                if drift_detected:
                    self.drift_detected_count += 1
                    self.last_drift_time = datetime.now()
                    
                    # Log drift to MLflow
                    log_to_mlflow(
                        metrics={
                            f"{self.name}_drift_score": score,
                            f"{self.name}_drift_detected": 1
                        },
                        params={
                            "detector_name": self.name,
                            "reference_samples": len(self.reference_data),
                            "current_samples": len(self.current_data)
                        },
                        experiment_name="drift_detection"
                    )
                
                return score, drift_detected
            else:
                return 0.0, False
    
    def _compute_drift(self) -> tuple:
        """
        Compute the drift score and determine if drift is detected.
        
        Returns:
            Tuple of (drift_score, drift_detected)
        """
        raise NotImplementedError("Subclasses must implement _compute_drift")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the drift detector.
        
        Returns:
            Dictionary with drift statistics
        """
        with self.lock:
            avg_drift_score = 0
            if self.drift_scores:
                avg_drift_score = sum(self.drift_scores) / len(self.drift_scores)
            
            return {
                "name": self.name,
                "reference_samples": len(self.reference_data),
                "current_samples": len(self.current_data),
                "total_samples_processed": self.total_samples,
                "drift_detected_count": self.drift_detected_count,
                "last_drift_time": self.last_drift_time.isoformat() if self.last_drift_time else None,
                "avg_drift_score": avg_drift_score,
                "drift_rate": (self.drift_detected_count / max(1, self.total_samples)) * 100
            }
    
    def should_retrain(self) -> bool:
        """
        Determine if the model should be retrained based on drift.
        
        Returns:
            True if retraining is recommended, False otherwise
        """
        with self.lock:
            if not self.drift_scores:
                return False
            
            # If recent drift scores are consistently high
            recent_scores = list(self.drift_scores)[-10:]  # Last 10 scores
            if len(recent_scores) >= 5 and sum(recent_scores) / len(recent_scores) > 0.6:
                return True
            
            # If drift detection rate is high
            if self.total_samples > 100 and (self.drift_detected_count / self.total_samples) > 0.3:
                return True
            
            return False

class NumericalDriftDetector(DriftDetector):
    """Drift detector for numerical data using KS test."""
    
    def __init__(self, window_size: int = 1000, name: str = "numerical_drift", threshold: float = 0.05):
        """
        Initialize the numerical drift detector.
        
        Args:
            window_size: Size of the sliding window for reference data
            name: Name of the drift detector
            threshold: p-value threshold for drift detection (lower means more sensitive)
        """
        super().__init__(window_size=window_size, name=name)
        self.threshold = threshold
    
    def _compute_drift(self) -> tuple:
        """
        Compute drift using Kolmogorov-Smirnov test.
        
        Returns:
            Tuple of (drift_score, drift_detected)
        """
        try:
            # Convert to numpy arrays for KS test
            reference = np.array(list(self.reference_data))
            current = np.array(list(self.current_data))
            
            # Check for valid data
            if np.all(np.isnan(reference)) or np.all(np.isnan(current)):
                return 0.0, False
            
            # Replace any NaN values with median
            reference_median = np.nanmedian(reference)
            current_median = np.nanmedian(current)
            
            reference = np.nan_to_num(reference, nan=reference_median)
            current = np.nan_to_num(current, nan=current_median)
            
            # Run KS test - this compares distributions
            ks_statistic, p_value = ks_2samp(reference, current)
            
            # Drift score is 1 - p_value (higher score = more drift)
            drift_score = 1.0 - p_value
            
            # Drift is detected if p-value is below threshold
            drift_detected = p_value < self.threshold
            
            # For external monitoring
            record_data_drift(f"{self.name}", drift_score)
            
            return drift_score, drift_detected
            
        except Exception as e:
            logger.error(f"Error computing numerical drift: {str(e)}")
            return 0.0, False

class CategoricalDriftDetector(DriftDetector):
    """Drift detector for categorical data using Chi-squared test."""
    
    def __init__(self, window_size: int = 1000, name: str = "categorical_drift", threshold: float = 0.05):
        """
        Initialize the categorical drift detector.
        
        Args:
            window_size: Size of the sliding window for reference data
            name: Name of the drift detector
            threshold: p-value threshold for drift detection (lower means more sensitive)
        """
        super().__init__(window_size=window_size, name=name)
        self.threshold = threshold
        self.categories = set()
    
    def add_reference_sample(self, sample: Any):
        """
        Add a sample to the reference distribution.
        
        Args:
            sample: The categorical value to add
        """
        with self.lock:
            self.reference_data.append(sample)
            self.categories.add(sample)
    
    def add_current_sample(self, sample: Any):
        """
        Add a sample to the current distribution and check for drift.
        
        Args:
            sample: The categorical value to add
            
        Returns:
            Drift score and whether drift was detected
        """
        with self.lock:
            self.current_data.append(sample)
            self.categories.add(sample)
            self.total_samples += 1
            
            # Only check for drift if we have enough data
            if len(self.reference_data) >= 30 and len(self.current_data) >= 30:
                score, drift_detected = self._compute_drift()
                
                # Record the drift score
                self.drift_scores.append(score)
                
                if drift_detected:
                    self.drift_detected_count += 1
                    self.last_drift_time = datetime.now()
                    
                    # Log drift to MLflow
                    log_to_mlflow(
                        metrics={
                            f"{self.name}_drift_score": score,
                            f"{self.name}_drift_detected": 1
                        },
                        params={
                            "detector_name": self.name,
                            "reference_samples": len(self.reference_data),
                            "current_samples": len(self.current_data)
                        },
                        experiment_name="drift_detection"
                    )
                
                return score, drift_detected
            else:
                return 0.0, False
    
    def _compute_drift(self) -> tuple:
        """
        Compute drift using Chi-squared test.
        
        Returns:
            Tuple of (drift_score, drift_detected)
        """
        try:
            # Count occurrences of each category
            reference_counts = defaultdict(int)
            current_counts = defaultdict(int)
            
            for category in self.categories:
                reference_counts[category] = 0
                current_counts[category] = 0
            
            for sample in self.reference_data:
                reference_counts[sample] += 1
            
            for sample in self.current_data:
                current_counts[sample] += 1
            
            # Create contingency table
            table = [
                [reference_counts[cat] for cat in self.categories],
                [current_counts[cat] for cat in self.categories]
            ]
            
            # Run Chi-squared test
            chi2, p_value, dof, expected = chi2_contingency(table)
            
            # Drift score is 1 - p_value (higher score = more drift)
            drift_score = 1.0 - p_value
            
            # Drift is detected if p-value is below threshold
            drift_detected = p_value < self.threshold
            
            # For external monitoring
            record_data_drift(f"{self.name}", drift_score)
            
            return drift_score, drift_detected
            
        except Exception as e:
            logger.error(f"Error computing categorical drift: {str(e)}")
            return 0.0, False

class TextEmbeddingDriftDetector(DriftDetector):
    """Drift detector for text embeddings using cosine similarity."""
    
    def __init__(self, window_size: int = 1000, name: str = "embedding_drift", threshold: float = 0.2):
        """
        Initialize the text embedding drift detector.
        
        Args:
            window_size: Size of the sliding window for reference data
            name: Name of the drift detector
            threshold: Cosine similarity threshold for drift detection (lower means more drift)
        """
        super().__init__(window_size=window_size, name=name)
        self.threshold = threshold
        self.reference_centroid = None
        self.current_centroid = None
    
    def add_reference_sample(self, sample: Union[List[float], np.ndarray]):
        """
        Add a sample to the reference distribution.
        
        Args:
            sample: The embedding vector to add (must be 1D array-like)
        """
        with self.lock:
            # Convert to numpy array
            if not isinstance(sample, np.ndarray):
                sample = np.array(sample, dtype=np.float32)
            
            self.reference_data.append(sample)
            
            # Update reference centroid
            if len(self.reference_data) > 0:
                self.reference_centroid = np.mean([np.array(x) for x in self.reference_data], axis=0)
    
    def add_current_sample(self, sample: Union[List[float], np.ndarray]):
        """
        Add a sample to the current distribution and check for drift.
        
        Args:
            sample: The embedding vector to add (must be 1D array-like)
            
        Returns:
            Drift score and whether drift was detected
        """
        with self.lock:
            # Convert to numpy array
            if not isinstance(sample, np.ndarray):
                sample = np.array(sample, dtype=np.float32)
                
            self.current_data.append(sample)
            self.total_samples += 1
            
            # Update current centroid
            if len(self.current_data) > 0:
                self.current_centroid = np.mean([np.array(x) for x in self.current_data], axis=0)
            
            # Only check for drift if we have enough data
            if (len(self.reference_data) >= 10 and 
                len(self.current_data) >= 10 and 
                self.reference_centroid is not None and 
                self.current_centroid is not None):
                
                score, drift_detected = self._compute_drift()
                
                # Record the drift score
                self.drift_scores.append(score)
                
                if drift_detected:
                    self.drift_detected_count += 1
                    self.last_drift_time = datetime.now()
                    
                    # Log drift to MLflow
                    log_to_mlflow(
                        metrics={
                            f"{self.name}_drift_score": score,
                            f"{self.name}_drift_detected": 1
                        },
                        params={
                            "detector_name": self.name,
                            "reference_samples": len(self.reference_data),
                            "current_samples": len(self.current_data)
                        },
                        experiment_name="drift_detection"
                    )
                
                return score, drift_detected
            else:
                return 0.0, False
    
    def _compute_drift(self) -> tuple:
        """
        Compute drift using cosine similarity between centroids.
        
        Returns:
            Tuple of (drift_score, drift_detected)
        """
        try:
            # Compute cosine similarity between reference and current centroids
            similarity = self._cosine_similarity(self.reference_centroid, self.current_centroid)
            
            # Drift score is 1 - similarity (higher score = more drift)
            drift_score = 1.0 - similarity
            
            # Drift is detected if similarity is below threshold
            drift_detected = similarity < self.threshold
            
            # For external monitoring
            record_data_drift(f"{self.name}", drift_score)
            
            return drift_score, drift_detected
            
        except Exception as e:
            logger.error(f"Error computing embedding drift: {str(e)}")
            return 0.0, False
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.
        
        Args:
            a: First vector
            b: Second vector
            
        Returns:
            Cosine similarity (0-1, where 1 = identical)
        """
        if a is None or b is None:
            return 0.0
            
        if np.all(a == 0) or np.all(b == 0):
            return 0.0
            
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

class ModelPerformanceDriftDetector:
    """Monitor drift in model performance metrics over time."""
    
    def __init__(self, name: str = "model_performance", window_size: int = 100, threshold: float = 0.1):
        """
        Initialize the model performance drift detector.
        
        Args:
            name: Name of the drift detector
            window_size: Size of the sliding window for metrics
            threshold: Threshold for significance in performance changes
        """
        self.name = name
        self.window_size = window_size
        self.threshold = threshold
        
        # Metrics history
        self.accuracy_history = deque(maxlen=window_size)
        self.latency_history = deque(maxlen=window_size)
        self.confidence_history = deque(maxlen=window_size)
        
        # Performance change detection
        self.baseline_accuracy = None
        self.baseline_latency = None
        self.baseline_confidence = None
        
        self.accuracy_trend = []
        self.latency_trend = []
        self.confidence_trend = []
        
        self.lock = threading.RLock()
        
        # Statistics
        self.total_records = 0
        self.drift_detected_count = 0
        self.last_drift_time = None
        
        logger.info(f"Initialized model performance drift detector: {name}")
    
    def add_performance_metrics(self, accuracy: float = None, latency: float = None, confidence: float = None):
        """
        Add performance metrics and check for drift.
        
        Args:
            accuracy: Model accuracy or similar performance metric (0-1)
            latency: Model prediction latency
            confidence: Model prediction confidence (0-1)
            
        Returns:
            Drift score and whether drift was detected
        """
        with self.lock:
            self.total_records += 1
            
            if accuracy is not None:
                self.accuracy_history.append(accuracy)
            
            if latency is not None:
                self.latency_history.append(latency)
            
            if confidence is not None:
                self.confidence_history.append(confidence)
            
            # Establish baseline after 30 records
            if self.total_records == 30:
                if self.accuracy_history:
                    self.baseline_accuracy = sum(self.accuracy_history) / len(self.accuracy_history)
                
                if self.latency_history:
                    self.baseline_latency = sum(self.latency_history) / len(self.latency_history)
                
                if self.confidence_history:
                    self.baseline_confidence = sum(self.confidence_history) / len(self.confidence_history)
            
            # Check for drift after baseline is established
            if (self.baseline_accuracy is not None or 
                self.baseline_latency is not None or 
                self.baseline_confidence is not None):
                
                score, drift_detected = self._compute_drift()
                
                if drift_detected:
                    self.drift_detected_count += 1
                    self.last_drift_time = datetime.now()
                    
                    # Log drift to MLflow
                    log_to_mlflow(
                        metrics={
                            f"{self.name}_drift_score": score,
                            f"{self.name}_drift_detected": 1
                        },
                        params={
                            "detector_name": self.name,
                            "total_records": self.total_records
                        },
                        experiment_name="drift_detection"
                    )
                
                return score, drift_detected
            else:
                return 0.0, False
    
    def _compute_drift(self) -> tuple:
        """
        Compute drift in model performance.
        
        Returns:
            Tuple of (drift_score, drift_detected)
        """
        try:
            drift_score = 0.0
            drift_detected = False
            
            # Update current averages
            current_accuracy = sum(self.accuracy_history) / len(self.accuracy_history) if self.accuracy_history else None
            current_latency = sum(self.latency_history) / len(self.latency_history) if self.latency_history else None
            current_confidence = sum(self.confidence_history) / len(self.confidence_history) if self.confidence_history else None
            
            # Calculate relative changes
            accuracy_change = 0
            latency_change = 0
            confidence_change = 0
            
            if self.baseline_accuracy is not None and current_accuracy is not None:
                accuracy_change = (self.baseline_accuracy - current_accuracy) / max(0.01, self.baseline_accuracy)
                
                # Add to trend (negative = degradation)
                self.accuracy_trend.append(-accuracy_change)
                if len(self.accuracy_trend) > 10:
                    self.accuracy_trend = self.accuracy_trend[-10:]
            
            if self.baseline_latency is not None and current_latency is not None:
                latency_change = (current_latency - self.baseline_latency) / max(0.01, self.baseline_latency)
                
                # Add to trend (positive = degradation)
                self.latency_trend.append(latency_change)
                if len(self.latency_trend) > 10:
                    self.latency_trend = self.latency_trend[-10:]
            
            if self.baseline_confidence is not None and current_confidence is not None:
                confidence_change = (self.baseline_confidence - current_confidence) / max(0.01, self.baseline_confidence)
                
                # Add to trend (negative = degradation)
                self.confidence_trend.append(-confidence_change)
                if len(self.confidence_trend) > 10:
                    self.confidence_trend = self.confidence_trend[-10:]
            
            # Overall drift score - weighted average of absolute changes
            weights = []
            changes = []
            
            if self.baseline_accuracy is not None and current_accuracy is not None:
                weights.append(2.0)  # Accuracy is most important
                changes.append(abs(accuracy_change))
            
            if self.baseline_latency is not None and current_latency is not None:
                weights.append(1.0)
                changes.append(abs(latency_change))
            
            if self.baseline_confidence is not None and current_confidence is not None:
                weights.append(1.0)
                changes.append(abs(confidence_change))
            
            if weights and changes:
                drift_score = sum(w * c for w, c in zip(weights, changes)) / sum(weights)
                
                # Drift detected if score exceeds threshold
                drift_detected = drift_score > self.threshold
                
                # For external monitoring
                record_data_drift(f"{self.name}", drift_score)
            
            return drift_score, drift_detected
            
        except Exception as e:
            logger.error(f"Error computing performance drift: {str(e)}")
            return 0.0, False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the drift detector.
        
        Returns:
            Dictionary with drift statistics
        """
        with self.lock:
            # Calculate trends (positive = improving, negative = degrading)
            accuracy_trend = sum(self.accuracy_trend) / len(self.accuracy_trend) if self.accuracy_trend else 0
            latency_trend = sum(self.latency_trend) / len(self.latency_trend) if self.latency_trend else 0
            confidence_trend = sum(self.confidence_trend) / len(self.confidence_trend) if self.confidence_trend else 0
            
            # Current metrics
            current_accuracy = sum(self.accuracy_history) / len(self.accuracy_history) if self.accuracy_history else None
            current_latency = sum(self.latency_history) / len(self.latency_history) if self.latency_history else None
            current_confidence = sum(self.confidence_history) / len(self.confidence_history) if self.confidence_history else None
            
            return {
                "name": self.name,
                "total_records": self.total_records,
                "drift_detected_count": self.drift_detected_count,
                "last_drift_time": self.last_drift_time.isoformat() if self.last_drift_time else None,
                "baseline": {
                    "accuracy": self.baseline_accuracy,
                    "latency": self.baseline_latency,
                    "confidence": self.baseline_confidence
                },
                "current": {
                    "accuracy": current_accuracy,
                    "latency": current_latency,
                    "confidence": current_confidence
                },
                "trends": {
                    "accuracy": accuracy_trend,
                    "latency": latency_trend,
                    "confidence": confidence_trend
                }
            }
    
    def should_retrain(self) -> bool:
        """
        Determine if the model should be retrained based on performance drift.
        
        Returns:
            True if retraining is recommended, False otherwise
        """
        with self.lock:
            # Check for significant negative accuracy trend
            if self.accuracy_trend and len(self.accuracy_trend) >= 5:
                avg_trend = sum(self.accuracy_trend) / len(self.accuracy_trend)
                if avg_trend < -0.1:  # 10% accuracy drop
                    return True
            
            # Check for significant latency increase
            if self.latency_trend and len(self.latency_trend) >= 5:
                avg_trend = sum(self.latency_trend) / len(self.latency_trend)
                if avg_trend > 0.2:  # 20% latency increase
                    return True
            
            # Check for confidence degradation
            if self.confidence_trend and len(self.confidence_trend) >= 5:
                avg_trend = sum(self.confidence_trend) / len(self.confidence_trend)
                if avg_trend < -0.15:  # 15% confidence drop
                    return True
            
            return False
    
    def reset_baseline(self):
        """Reset the baseline performance metrics using current values."""
        with self.lock:
            if self.accuracy_history:
                self.baseline_accuracy = sum(self.accuracy_history) / len(self.accuracy_history)
            
            if self.latency_history:
                self.baseline_latency = sum(self.latency_history) / len(self.latency_history)
            
            if self.confidence_history:
                self.baseline_confidence = sum(self.confidence_history) / len(self.confidence_history)
            
            # Clear trends after resetting baseline
            self.accuracy_trend = []
            self.latency_trend = []
            self.confidence_trend = []
            
            logger.info(f"Reset baseline for model performance drift detector: {self.name}")
            logger.info(f"New baseline - Accuracy: {self.baseline_accuracy}, Latency: {self.baseline_latency}, Confidence: {self.baseline_confidence}")

# Dictionary of registered drift detectors
_drift_detectors = {}

def get_drift_detector(name: str) -> Optional[DriftDetector]:
    """
    Get a drift detector by name.
    
    Args:
        name: Name of the drift detector
        
    Returns:
        The drift detector if found, None otherwise
    """
    return _drift_detectors.get(name)

def register_drift_detector(detector: Union[DriftDetector, ModelPerformanceDriftDetector]):
    """
    Register a drift detector.
    
    Args:
        detector: The drift detector to register
    """
    _drift_detectors[detector.name] = detector
    logger.info(f"Registered drift detector: {detector.name}")

def create_numerical_drift_detector(name: str, window_size: int = 1000, threshold: float = 0.05) -> NumericalDriftDetector:
    """
    Create and register a numerical drift detector.
    
    Args:
        name: Name of the drift detector
        window_size: Size of the sliding window for reference data
        threshold: p-value threshold for drift detection
        
    Returns:
        The created drift detector
    """
    detector = NumericalDriftDetector(name=name, window_size=window_size, threshold=threshold)
    register_drift_detector(detector)
    return detector

def create_categorical_drift_detector(name: str, window_size: int = 1000, threshold: float = 0.05) -> CategoricalDriftDetector:
    """
    Create and register a categorical drift detector.
    
    Args:
        name: Name of the drift detector
        window_size: Size of the sliding window for reference data
        threshold: p-value threshold for drift detection
        
    Returns:
        The created drift detector
    """
    detector = CategoricalDriftDetector(name=name, window_size=window_size, threshold=threshold)
    register_drift_detector(detector)
    return detector

def create_embedding_drift_detector(name: str, window_size: int = 1000, threshold: float = 0.2) -> TextEmbeddingDriftDetector:
    """
    Create and register an embedding drift detector.
    
    Args:
        name: Name of the drift detector
        window_size: Size of the sliding window for reference data
        threshold: Cosine similarity threshold for drift detection
        
    Returns:
        The created drift detector
    """
    detector = TextEmbeddingDriftDetector(name=name, window_size=window_size, threshold=threshold)
    register_drift_detector(detector)
    return detector

def create_performance_drift_detector(name: str, window_size: int = 100, threshold: float = 0.1) -> ModelPerformanceDriftDetector:
    """
    Create and register a model performance drift detector.
    
    Args:
        name: Name of the drift detector
        window_size: Size of the sliding window for metrics
        threshold: Threshold for significance in performance changes
        
    Returns:
        The created drift detector
    """
    detector = ModelPerformanceDriftDetector(name=name, window_size=window_size, threshold=threshold)
    register_drift_detector(detector)
    return detector

def get_all_drift_stats() -> Dict[str, Dict[str, Any]]:
    """
    Get statistics from all drift detectors.
    
    Returns:
        Dictionary with statistics from all drift detectors
    """
    return {name: detector.get_stats() for name, detector in _drift_detectors.items()}

def should_retrain_any_model() -> Dict[str, bool]:
    """
    Check if any model should be retrained based on drift.
    
    Returns:
        Dictionary mapping detector names to retraining recommendations
    """
    return {name: detector.should_retrain() for name, detector in _drift_detectors.items()} 