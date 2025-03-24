"""
MLflow Integration Module
====================

Provides integration with MLflow for tracking and logging ML observability metrics.
This module handles experiment tracking, metric logging, and artifact storage.
"""

import os
import json
import time
import tempfile
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from app.core.logger import logger
from app.core.config import settings

class MLflowManager:
    """Manages connections and logging to MLflow."""
    
    def __init__(self, tracking_uri: Optional[str] = None):
        """
        Initialize the MLflow connection manager.
        
        Args:
            tracking_uri: URI of the MLflow tracking server (default: from env var MLFLOW_TRACKING_URI)
        """
        self.tracking_uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
        self.initialized = False
        self.client = None
        
        if not MLFLOW_AVAILABLE:
            logger.warning("MLflow not installed. MLflow tracking disabled.")
            return
            
        try:
            if self.tracking_uri:
                mlflow.set_tracking_uri(self.tracking_uri)
                self.client = MlflowClient(tracking_uri=self.tracking_uri)
                self.initialized = True
                logger.info(f"MLflow initialized with tracking URI: {self.tracking_uri}")
            else:
                logger.warning("No MLflow tracking URI provided. MLflow tracking disabled.")
        except Exception as e:
            logger.warning(f"Failed to initialize MLflow: {str(e)}")
    
    def start_run(self, experiment_name: str, run_name: Optional[str] = None) -> Optional[str]:
        """
        Start a new MLflow run.
        
        Args:
            experiment_name: Name of the MLflow experiment
            run_name: Optional name for the run
            
        Returns:
            Run ID if successful, None otherwise
        """
        if not self._check_available():
            return None
            
        try:
            # Get or create experiment
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(
                    experiment_name,
                    artifact_location=settings.MLFLOW_ARTIFACT_LOCATION
                )
            else:
                experiment_id = experiment.experiment_id
                
            # Start run
            active_run = mlflow.start_run(
                experiment_id=experiment_id,
                run_name=run_name or f"run_{int(time.time())}"
            )
            
            logger.info(f"Started MLflow run: {active_run.info.run_id} in experiment: {experiment_name}")
            return active_run.info.run_id
            
        except Exception as e:
            logger.warning(f"Failed to start MLflow run: {str(e)}")
            return None
    
    def log_metrics(self, metrics: Dict[str, Union[float, int]]) -> bool:
        """
        Log metrics to the active MLflow run.
        
        Args:
            metrics: Dictionary of metric names and values
            
        Returns:
            True if successful, False otherwise
        """
        if not self._check_available():
            return False
            
        try:
            mlflow.log_metrics(metrics)
            return True
        except Exception as e:
            logger.warning(f"Failed to log metrics to MLflow: {str(e)}")
            return False
    
    def log_parameters(self, params: Dict[str, Any]) -> bool:
        """
        Log parameters to the active MLflow run.
        
        Args:
            params: Dictionary of parameter names and values
            
        Returns:
            True if successful, False otherwise
        """
        if not self._check_available():
            return False
            
        try:
            # Convert non-string parameters to strings
            string_params = {k: str(v) for k, v in params.items()}
            mlflow.log_params(string_params)
            return True
        except Exception as e:
            logger.warning(f"Failed to log parameters to MLflow: {str(e)}")
            return False
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> bool:
        """
        Log an artifact to the active MLflow run.
        
        Args:
            local_path: Path to the local file to log
            artifact_path: Optional destination directory for the artifact
            
        Returns:
            True if successful, False otherwise
        """
        if not self._check_available():
            return False
            
        try:
            mlflow.log_artifact(local_path, artifact_path)
            return True
        except Exception as e:
            logger.warning(f"Failed to log artifact to MLflow: {str(e)}")
            return False
    
    def log_dict(self, dictionary: Dict[str, Any], artifact_file: str) -> bool:
        """
        Log a dictionary as a JSON artifact.
        
        Args:
            dictionary: Dictionary to log
            artifact_file: Name of the artifact file
            
        Returns:
            True if successful, False otherwise
        """
        if not self._check_available():
            return False
            
        try:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(dictionary, f, indent=2)
                temp_path = f.name
            
            # Log the file as an artifact
            success = self.log_artifact(temp_path, os.path.dirname(artifact_file))
            
            # Clean up
            os.remove(temp_path)
            
            return success
        except Exception as e:
            logger.warning(f"Failed to log dictionary to MLflow: {str(e)}")
            return False
    
    def log_figure(self, figure, artifact_file: str) -> bool:
        """
        Log a matplotlib figure as an artifact.
        
        Args:
            figure: Matplotlib figure object
            artifact_file: Name of the artifact file
            
        Returns:
            True if successful, False otherwise
        """
        if not self._check_available():
            return False
            
        try:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                temp_path = f.name
            
            # Save the figure
            figure.savefig(temp_path)
            
            # Log the file as an artifact
            success = self.log_artifact(temp_path, os.path.dirname(artifact_file))
            
            # Clean up
            os.remove(temp_path)
            
            return success
        except Exception as e:
            logger.warning(f"Failed to log figure to MLflow: {str(e)}")
            return False
    
    def end_run(self) -> bool:
        """
        End the active MLflow run.
        
        Returns:
            True if successful, False otherwise
        """
        if not self._check_available():
            return False
            
        try:
            mlflow.end_run()
            return True
        except Exception as e:
            logger.warning(f"Failed to end MLflow run: {str(e)}")
            return False
    
    def _check_available(self) -> bool:
        """
        Check if MLflow is available and initialized.
        
        Returns:
            True if available, False otherwise
        """
        if not MLFLOW_AVAILABLE:
            logger.warning("MLflow not installed. MLflow tracking disabled.")
            return False
            
        if not self.initialized:
            logger.warning("MLflow not initialized. MLflow tracking disabled.")
            return False
            
        return True


# Global instance
_mlflow_manager = None

def get_mlflow_manager() -> MLflowManager:
    """
    Get or create the MLflow manager instance.
    
    Returns:
        MLflow manager instance
    """
    global _mlflow_manager
    if _mlflow_manager is None:
        _mlflow_manager = MLflowManager(tracking_uri=settings.MLFLOW_TRACKING_URI)
    return _mlflow_manager

def log_model_metrics(model_id: str, metrics: Dict[str, Any]) -> bool:
    """
    Log model metrics to MLflow.
    
    Args:
        model_id: ID of the model
        metrics: Dictionary of metrics
        
    Returns:
        True if successful, False otherwise
    """
    manager = get_mlflow_manager()
    
    # Start run if needed
    if not mlflow.active_run():
        run_id = manager.start_run(f"model_{model_id}")
        if not run_id:
            return False
    
    # Prepare metrics (ensure they are numbers)
    numeric_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float, np.number)):
            numeric_metrics[key] = float(value)
    
    # Log metrics
    success = manager.log_metrics(numeric_metrics)
    
    # Log non-numeric values as parameters
    params = {k: v for k, v in metrics.items() if k not in numeric_metrics}
    if params:
        manager.log_parameters(params)
    
    return success

def log_drift_detection(detector_id: str, scores: Dict[str, float], detected: bool) -> bool:
    """
    Log drift detection results to MLflow.
    
    Args:
        detector_id: ID of the drift detector
        scores: Dictionary of drift scores
        detected: Whether drift was detected
        
    Returns:
        True if successful, False otherwise
    """
    manager = get_mlflow_manager()
    
    # Check if we need to start a run - only if no run is active
    if not mlflow.active_run():
        logger.info(f"No active MLflow run found, starting new run for drift detection: {detector_id}")
        run_id = manager.start_run(f"drift_{detector_id}")
        if not run_id:
            return False
        # Mark that we created this run and need to end it
        created_run = True
    else:
        # Use existing run
        created_run = False
        logger.debug(f"Using existing MLflow run for drift detection: {detector_id}")
    
    try:
        # Prepare metrics
        metrics = {f"drift_score_{k}": v for k, v in scores.items()}
        metrics["drift_detected"] = 1 if detected else 0
        
        # Add detector ID as a tag
        manager.log_parameters({"detector_id": detector_id})
        
        # Log metrics
        success = manager.log_metrics(metrics)
        
        # If we created this run, end it
        if created_run:
            manager.end_run()
            
        return success
    except Exception as e:
        logger.warning(f"Failed to log drift detection to MLflow: {str(e)}")
        # Clean up if we created the run
        if created_run:
            manager.end_run()
        return False

def log_visualization(figure, artifact_name: str) -> bool:
    """
    Log a visualization to MLflow.
    
    Args:
        figure: Matplotlib figure
        artifact_name: Name of the artifact
        
    Returns:
        True if successful, False otherwise
    """
    manager = get_mlflow_manager()
    
    # Check if we need to start a run
    if not mlflow.active_run():
        logger.warning("No active MLflow run found for logging visualization. Using existing MLflow manager.")
        # Since logging visualization is usually part of a larger workflow,
        # we won't create a new run here but will log to the existing manager
    
    return manager.log_figure(figure, artifact_name)

def log_metrics_export(metrics_data: Dict[str, Any], artifact_name: str = "metrics_export.json") -> bool:
    """
    Log metrics export to MLflow.
    
    Args:
        metrics_data: Dictionary of metrics data
        artifact_name: Name of the artifact
        
    Returns:
        True if successful, False otherwise
    """
    manager = get_mlflow_manager()
    
    # Check if we need to start a run
    if not mlflow.active_run():
        logger.warning("No active MLflow run found for logging metrics export. Using existing MLflow manager.")
        # Since logging metrics export is usually part of a larger workflow,
        # we won't create a new run here but will log to the existing manager
    
    return manager.log_dict(metrics_data, artifact_name) 