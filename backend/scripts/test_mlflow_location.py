#!/usr/bin/env python
"""
MLflow Directory Location Test
=============================

This script tests that new MLflow experiments are created in the correct
backend/mlruns directory.

Usage:
    python test_mlflow_location.py
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime

# Add the project root to the Python path
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = BACKEND_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import required modules
import mlflow
from backend.app.core.config import settings
from backend.app.core.mlflow_utils import get_or_create_experiment

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("mlflow-test")

def main():
    # Print information about the configured MLflow settings
    logger.info(f"MLflow settings:")
    logger.info(f"  Tracking URI: {settings.MLFLOW_TRACKING_URI}")
    logger.info(f"  Artifact Location: {settings.MLFLOW_ARTIFACT_LOCATION}")
    
    # Create a test experiment with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"test_experiment_{timestamp}"
    
    logger.info(f"Creating test experiment: {experiment_name}")
    experiment_id = get_or_create_experiment(experiment_name)
    
    # Get the experiment and check its location
    experiment = mlflow.get_experiment(experiment_id)
    
    logger.info(f"Experiment created with ID: {experiment_id}")
    logger.info(f"Experiment artifact location: {experiment.artifact_location}")
    
    # Check if the location is correct
    if experiment.artifact_location == settings.MLFLOW_ARTIFACT_LOCATION:
        logger.info("SUCCESS: Experiment was created with the correct artifact location")
    else:
        logger.error(f"FAILURE: Experiment was created with incorrect location: {experiment.artifact_location}")
        logger.error(f"Expected: {settings.MLFLOW_ARTIFACT_LOCATION}")
    
    # Create a test run and log some metrics
    with mlflow.start_run(experiment_id=experiment_id, run_name=f"test_run_{timestamp}"):
        logger.info("Creating test run and logging metrics")
        mlflow.log_param("test_param", "test_value")
        mlflow.log_metric("test_metric", 1.0)
    
    # Check for the mlruns directory
    expected_dir = os.path.join(PROJECT_ROOT, "backend", "mlruns")
    if os.path.exists(expected_dir):
        logger.info(f"SUCCESS: backend/mlruns directory exists at: {expected_dir}")
    else:
        logger.error(f"FAILURE: backend/mlruns directory does not exist at: {expected_dir}")
    
    # Check if there's a mlruns directory in the root
    root_dir = os.path.join(PROJECT_ROOT, "mlruns")
    if os.path.exists(root_dir):
        logger.warning(f"WARNING: mlruns directory still exists in the project root: {root_dir}")
    else:
        logger.info("SUCCESS: No mlruns directory in the project root")

if __name__ == "__main__":
    main() 