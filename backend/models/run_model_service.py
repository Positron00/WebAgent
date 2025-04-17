#!/usr/bin/env python
"""
Run Model Service Script

This script loads and runs a model service based on the specified configuration.
"""
import logging
import traceback
import argparse
import os
import sys

logger = logging.getLogger(__name__)

def setup_logging():
    """Set up logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("model_service.log")
        ]
    )

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Run a model service')
    parser.add_argument("--model-id", required=True, help="ID of the model to run")
    parser.add_argument("--config", help="Path to the configuration file")
    return parser.parse_args()

def run_model_service(model_id: str, config_path: str = None):
    """
    Load and run a model service
    
    Args:
        model_id: ID of the model to run
        config_path: Path to the configuration file
    """
    try:
        from backend.models.config import load_models_config, DEFAULT_MODEL_CONFIG
        from backend.models.services import ModelServiceConfig, TransformerModelService
        from backend.models.services.core_transformer_service import CoreTransformerModelService
        
        # Load configuration
        config = load_models_config(config_path)
        
        if model_id not in config["models"]:
            logger.error(f"Model {model_id} not found in configuration")
            return False
        
        # Get model configuration
        model_config = config["models"][model_id]
        model_type = model_config.get("model_type", "llm")
        
        # Create default configuration for the service
        default_config = DEFAULT_MODEL_CONFIG.copy()
        default_config.update(model_config)
        
        # Create model service configuration
        service_config = ModelServiceConfig(
            model_id=model_id,
            display_name=model_config.get("display_name", model_id),
            model_type=model_type,
            host=model_config.get("host", "localhost"),
            port=model_config.get("port", 8500),
            model_path=model_config.get("model_path"),
            device=model_config.get("device", "cpu"),
            capabilities=model_config.get("capabilities", ["text-generation"]),
            parameters=model_config.get("parameters", {})
        )
        
        # Run the service
        logger.info(f"Starting model service: {model_id}")
        logger.info(f"Configuration: {service_config}")
        
        if model_id == "core-transformer":
            # Use our custom Transformer implementation
            service = CoreTransformerModelService(service_config)
            service.run()
            return True
        elif model_type in ["llm", "embedding"]:
            # Use the Hugging Face-based implementation
            service = TransformerModelService(service_config)
            service.run()
            return True
        else:
            logger.error(f"Unsupported model type: {model_type}")
            return False
            
    except Exception as e:
        logger.error(f"Error running model service: {e}")
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    args = parse_args()
    run_model_service(args.model_id, args.config) 