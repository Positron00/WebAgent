#!/usr/bin/env python
"""
Run Model Manager

This script starts the Model Manager which loads and manages all model services.
It serves as the entry point for the model microservices framework.
"""
import argparse
import logging
import os
import sys
import signal
import time
from pathlib import Path

# Make sure backend is in the path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run the Model Manager")
    parser.add_argument("--config", help="Path to the models configuration file")
    parser.add_argument("--gateway-host", default="localhost", help="Host for the API gateway")
    parser.add_argument("--gateway-port", type=int, default=8000, help="Port for the API gateway")
    parser.add_argument("--models", nargs="+", help="Specific models to load (default: all in config)")
    return parser.parse_args()

def signal_handler(sig, frame):
    """Handle termination signals"""
    logger.info("Received termination signal. Shutting down...")
    sys.exit(0)

def run_manager():
    """Run the model manager"""
    args = parse_args()
    
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        from backend.models.model_manager import ModelManager
        from backend.models.config import load_models_config
        
        # Load configuration first if provided
        if args.config:
            config = load_models_config(args.config)
        
        # Create and start the model manager
        manager = ModelManager()
        
        # Start the API gateway
        manager.start_gateway(host=args.gateway_host, port=args.gateway_port)
        
        # Load models from configuration
        if args.models:
            for model_id in args.models:
                manager.load_model_service(model_id)
        else:
            # Load all models from configuration
            config = load_models_config(args.config)
            
            if "models" in config:
                for model_id in config["models"]:
                    manager.load_model_service(model_id)
            else:
                logger.warning("No models found in configuration")
        
        # Keep the main thread running
        logger.info("Model Manager is running. Press Ctrl+C to exit.")
        while True:
            time.sleep(1)
            
    except Exception as e:
        logger.error(f"Error running Model Manager: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    run_manager() 