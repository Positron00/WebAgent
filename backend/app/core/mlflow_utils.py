"""
MLflow utility functions for the WebAgent project.
Ensures consistent MLflow configuration across the application.
"""
import logging
import mlflow
from app.core.config import settings
import os

logger = logging.getLogger(__name__)

def get_or_create_experiment(experiment_name):
    """
    Get or create an MLflow experiment with proper artifact location configuration.
    
    This function ensures all experiments are created with the backend/mlruns
    path as specified in settings.MLFLOW_ARTIFACT_LOCATION.
    
    Args:
        experiment_name: Name of the experiment to get or create
        
    Returns:
        experiment_id: The ID of the created or retrieved experiment
    """
    # Set the tracking URI from settings
    tracking_uri = settings.MLFLOW_TRACKING_URI
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        logger.debug(f"Using MLflow tracking URI: {tracking_uri}")
    
    # Set the default artifact root
    os.environ["MLFLOW_DEFAULT_ARTIFACT_ROOT"] = settings.MLFLOW_DEFAULT_ARTIFACT_ROOT
    logger.debug(f"Setting MLFLOW_DEFAULT_ARTIFACT_ROOT to {settings.MLFLOW_DEFAULT_ARTIFACT_ROOT}")
    
    # First check if the experiment already exists
    experiment = mlflow.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        # Create a new experiment with the correct artifact location
        logger.info(f"Creating new MLflow experiment '{experiment_name}' with artifact location: {settings.MLFLOW_ARTIFACT_LOCATION}")
        experiment_id = mlflow.create_experiment(
            experiment_name,
            artifact_location=settings.MLFLOW_ARTIFACT_LOCATION
        )
    else:
        experiment_id = experiment.experiment_id
        # Check if the existing experiment has the correct artifact location
        if experiment.artifact_location != settings.MLFLOW_ARTIFACT_LOCATION:
            logger.warning(
                f"Experiment '{experiment_name}' exists but has artifact location "
                f"'{experiment.artifact_location}' instead of '{settings.MLFLOW_ARTIFACT_LOCATION}'. "
                f"Cannot change the artifact location of an existing experiment."
            )
    
    # Set as the active experiment
    mlflow.set_experiment(experiment_name)
    
    return experiment_id

def check_experiment_locations():
    """
    Check all existing MLflow experiments and identify those with incorrect artifact locations.
    
    This function prints information about any experiments that are not using the
    correct backend/mlruns path as specified in settings.MLFLOW_ARTIFACT_LOCATION.
    
    Returns:
        incorrect_experiments: List of experiments with incorrect artifact locations
    """
    # Set the tracking URI from settings
    tracking_uri = settings.MLFLOW_TRACKING_URI
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    
    # Get all experiments
    experiments = mlflow.search_experiments()
    incorrect_experiments = []
    
    for experiment in experiments:
        if experiment.artifact_location != settings.MLFLOW_ARTIFACT_LOCATION:
            incorrect_experiments.append({
                "id": experiment.experiment_id,
                "name": experiment.name,
                "current_location": experiment.artifact_location,
                "expected_location": settings.MLFLOW_ARTIFACT_LOCATION
            })
            logger.warning(
                f"Experiment '{experiment.name}' (ID: {experiment.experiment_id}) is using "
                f"artifact location '{experiment.artifact_location}' instead of "
                f"'{settings.MLFLOW_ARTIFACT_LOCATION}'"
            )
    
    if incorrect_experiments:
        logger.warning(
            f"Found {len(incorrect_experiments)} experiments with incorrect artifact locations. "
            f"To fix this, you need to delete these experiments and recreate them. "
            f"Note that this will delete all runs and artifacts for these experiments."
        )
    else:
        logger.info("All experiments are using the correct artifact location.")
    
    return incorrect_experiments 