#!/usr/bin/env python
"""
Models Configuration

This module provides configuration settings for model services.
"""
import os
import logging
from typing import Dict, List, Any
import yaml

logger = logging.getLogger(__name__)

# Default directory for model files
DEFAULT_MODELS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 
    "..", 
    "model_files"
)

# Default model configuration
DEFAULT_MODEL_CONFIG = {
    "host": "localhost",
    "device": "cpu",
    "capabilities": ["text-generation"],
    "parameters": {
        "quantization": None,
        "max_batch_size": 1,
        "max_input_length": 2048,
        "max_output_length": 2048,
        "timeout": 60
    }
}

# Helper function to create ports for models
def get_base_port(starting_port: int = 8500):
    """Get the base port for model services"""
    return starting_port

def load_models_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load model service configuration from YAML file
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dict: Configuration dictionary for model services
    """
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "models.yaml"
        )
    
    if not os.path.exists(config_path):
        logger.warning(f"Configuration file not found: {config_path}")
        return {"models": {}}
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded model configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading model configuration: {e}")
        return {"models": {}}

def save_models_config(config: Dict[str, Any], config_path: str = None) -> bool:
    """
    Save model service configuration to YAML file
    
    Args:
        config: Configuration dictionary
        config_path: Path to save the configuration file
        
    Returns:
        bool: True if saved successfully
    """
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "models.yaml"
        )
    
    try:
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        logger.info(f"Saved model configuration to {config_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving model configuration: {e}")
        return False

def create_default_config_if_not_exists():
    """Create default configuration file if it doesn't exist"""
    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "models.yaml"
    )
    
    if not os.path.exists(config_path):
        logger.info("Creating default model configuration")
        
        # Create default configuration
        default_config = {
            "models": {
                # No models by default
            }
        }
        
        # Save default configuration
        save_models_config(default_config, config_path)
        logger.info("Created default model configuration")
        
        # Create model_files directory if it doesn't exist
        if not os.path.exists(DEFAULT_MODELS_DIR):
            os.makedirs(DEFAULT_MODELS_DIR, exist_ok=True)
            logger.info(f"Created model files directory: {DEFAULT_MODELS_DIR}")

# Create default configuration file if it doesn't exist
create_default_config_if_not_exists() 