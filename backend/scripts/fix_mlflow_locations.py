#!/usr/bin/env python
"""
MLflow Directory Location Fixer
===============================

This script checks all MLflow experiments and identifies those with incorrect
artifact locations. It can optionally delete experiments with incorrect locations
so they can be recreated with the correct backend/mlruns path.

Usage:
    python fix_mlflow_locations.py [--delete-incorrect]

Options:
    --delete-incorrect    Delete experiments with incorrect artifact locations
                         (WARNING: This will delete all runs and artifacts for these experiments)
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add the project root to the Python path
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = BACKEND_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import required modules
import mlflow
from backend.app.core.config import settings
from backend.app.core.mlflow_utils import check_experiment_locations

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("mlflow-fix")

def delete_experiment(experiment_id, name):
    """
    Delete an MLflow experiment.
    
    Args:
        experiment_id: The ID of the experiment to delete
        name: The name of the experiment (for logging)
    
    Returns:
        bool: True if deletion was successful, False otherwise
    """
    try:
        mlflow.delete_experiment(experiment_id)
        logger.info(f"Successfully deleted experiment '{name}' (ID: {experiment_id})")
        return True
    except Exception as e:
        logger.error(f"Failed to delete experiment '{name}' (ID: {experiment_id}): {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Fix MLflow experiment locations")
    parser.add_argument("--delete-incorrect", action="store_true", 
                        help="Delete experiments with incorrect artifact locations")
    args = parser.parse_args()
    
    # Print information about the configured MLflow settings
    logger.info(f"MLflow settings:")
    logger.info(f"  Tracking URI: {settings.MLFLOW_TRACKING_URI}")
    logger.info(f"  Artifact Location: {settings.MLFLOW_ARTIFACT_LOCATION}")
    
    # Check experiment locations
    incorrect_experiments = check_experiment_locations()
    
    if not incorrect_experiments:
        logger.info("All experiments are using the correct artifact location. Nothing to fix.")
        return
    
    logger.info(f"Found {len(incorrect_experiments)} experiments with incorrect artifact locations:")
    for exp in incorrect_experiments:
        logger.info(f"  - '{exp['name']}' (ID: {exp['id']})")
        logger.info(f"    Current: {exp['current_location']}")
        logger.info(f"    Expected: {exp['expected_location']}")
    
    if args.delete_incorrect:
        logger.warning("Deleting experiments with incorrect artifact locations...")
        delete_count = 0
        
        for exp in incorrect_experiments:
            if delete_experiment(exp['id'], exp['name']):
                delete_count += 1
        
        logger.info(f"Deleted {delete_count} of {len(incorrect_experiments)} incorrect experiments.")
        logger.info("You can now recreate these experiments and they will use the correct artifact location.")
    else:
        logger.info("To delete these experiments, run the script with the --delete-incorrect flag.")
        logger.info("WARNING: Deleting experiments will delete all runs and artifacts for these experiments.")

if __name__ == "__main__":
    main() 