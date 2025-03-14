#!/usr/bin/env python
"""
Model Manager - Orchestrate model services

This module provides functionality to manage the lifecycle of model services,
including starting, stopping, and monitoring them.
"""
import logging
import os
import signal
import subprocess
import sys
import time
from typing import Dict, List, Optional, Any
import threading
import json
import asyncio
from pathlib import Path

from .config import load_models_config, DEFAULT_MODEL_CONFIG
from .services import ModelServiceConfig, TransformerModelService
from .registry import get_model_registry, ModelInfo
from .api.gateway import ModelAPIGateway

logger = logging.getLogger(__name__)

class ModelProcessInfo:
    """Information about a running model process"""
    def __init__(self, model_id: str, process: subprocess.Popen, config: Dict[str, Any]):
        self.model_id = model_id
        self.process = process
        self.config = config
        self.start_time = time.time()
        self.exit_code = None
        self.restart_count = 0
        
    def is_running(self) -> bool:
        """Check if the process is still running"""
        if self.process.poll() is not None:
            self.exit_code = self.process.returncode
            return False
        return True

class ModelManager:
    """Manager for model services"""
    
    def __init__(self):
        """Initialize the model manager"""
        self.config = load_models_config()
        self.processes: Dict[str, ModelProcessInfo] = {}
        self.gateway: Optional[ModelAPIGateway] = None
        self.gateway_thread: Optional[threading.Thread] = None
        self.running = False
        self._init_registry()
    
    def _init_registry(self) -> None:
        """Initialize the model registry"""
        # This initializes the singleton registry
        registry = get_model_registry()
        logger.info("Model registry initialized")
    
    def start_gateway(self, host: str = "localhost", port: int = 8000) -> None:
        """
        Start the API gateway
        
        Args:
            host: Host to run the gateway on
            port: Port to run the gateway on
        """
        if self.gateway_thread is not None and self.gateway_thread.is_alive():
            logger.warning("Gateway already running")
            return
        
        self.gateway = ModelAPIGateway(host=host, port=port)
        
        def run_gateway():
            try:
                logger.info(f"Starting API gateway on {host}:{port}")
                self.gateway.run()
            except Exception as e:
                logger.exception(f"Error running API gateway: {e}")
        
        self.gateway_thread = threading.Thread(target=run_gateway, daemon=True)
        self.gateway_thread.start()
        logger.info("API gateway started")
    
    def load_model_service(self, model_id: str) -> bool:
        """
        Load and start a model service
        
        Args:
            model_id: ID of the model to load
            
        Returns:
            bool: True if the model was started successfully
        """
        if model_id not in self.config["models"]:
            logger.error(f"Model {model_id} not found in configuration")
            return False
        
        # Get model configuration
        model_config = self.config["models"][model_id]
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
        
        # Create and run the model service
        if model_type in ["llm", "embedding"]:
            # Run the service
            service = TransformerModelService(service_config)
            service.run()
            return True
        else:
            logger.error(f"Unsupported model type: {model_type}")
            return False
    
    def start_model_service_subprocess(self, model_id: str) -> bool:
        """
        Start a model service as a separate process
        
        Args:
            model_id: ID of the model to start
            
        Returns:
            bool: True if the model was started successfully
        """
        if model_id not in self.config["models"]:
            logger.error(f"Model {model_id} not found in configuration")
            return False
        
        if model_id in self.processes and self.processes[model_id].is_running():
            logger.warning(f"Model service {model_id} is already running")
            return True
        
        # Get model configuration
        model_config = self.config["models"][model_id]
        model_type = model_config.get("model_type", "llm")
        
        # Get path to the runner script
        current_dir = os.path.dirname(os.path.abspath(__file__))
        runner_script = os.path.join(current_dir, "run_model_service.py")
        
        # Create command
        cmd = [
            sys.executable,
            runner_script,
            "--model-id", model_id,
            "--config", os.path.join(current_dir, "config", "models.yaml")
        ]
        
        # Start process
        logger.info(f"Starting model service for {model_id}")
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            # Store process info
            self.processes[model_id] = ModelProcessInfo(
                model_id=model_id,
                process=process,
                config=model_config
            )
            
            # Start log thread
            self._start_log_thread(model_id)
            
            logger.info(f"Started model service for {model_id} (PID: {process.pid})")
            return True
            
        except Exception as e:
            logger.exception(f"Error starting model service for {model_id}: {e}")
            return False
    
    def _start_log_thread(self, model_id: str) -> None:
        """
        Start a thread to read logs from the process
        
        Args:
            model_id: ID of the model
        """
        process_info = self.processes.get(model_id)
        if not process_info:
            return
        
        def read_logs():
            while process_info.is_running():
                # Read from stdout
                line = process_info.process.stdout.readline()
                if line:
                    logger.info(f"[{model_id}] {line.strip()}")
                
                # Read from stderr
                err_line = process_info.process.stderr.readline()
                if err_line:
                    logger.error(f"[{model_id}] {err_line.strip()}")
                
                # Small sleep to avoid high CPU usage
                time.sleep(0.1)
                
            logger.info(f"Process for {model_id} exited with code {process_info.exit_code}")
        
        log_thread = threading.Thread(target=read_logs, daemon=True)
        log_thread.start()
    
    def start_all_services(self) -> None:
        """Start all model services defined in the configuration"""
        self.running = True
        
        # Start API gateway
        self.start_gateway()
        
        # Start model services
        for model_id in self.config["models"]:
            self.start_model_service_subprocess(model_id)
        
        # Start monitoring thread
        threading.Thread(target=self._monitor_services, daemon=True).start()
    
    def stop_model_service(self, model_id: str) -> bool:
        """
        Stop a model service
        
        Args:
            model_id: ID of the model to stop
            
        Returns:
            bool: True if the model was stopped successfully
        """
        if model_id not in self.processes:
            logger.warning(f"Model service {model_id} is not running")
            return False
        
        # Get process info
        process_info = self.processes[model_id]
        
        # Send SIGTERM
        logger.info(f"Stopping model service {model_id}")
        try:
            process_info.process.terminate()
            
            # Wait for process to terminate
            for _ in range(5):  # Wait up to 5 seconds
                if not process_info.is_running():
                    break
                time.sleep(1)
            
            # Force kill if still running
            if process_info.is_running():
                logger.warning(f"Model service {model_id} did not terminate, sending SIGKILL")
                process_info.process.kill()
            
            # Update registry
            registry = get_model_registry()
            registry.update_model_status(model_id, "offline")
            
            logger.info(f"Stopped model service {model_id}")
            return True
            
        except Exception as e:
            logger.exception(f"Error stopping model service {model_id}: {e}")
            return False
    
    def stop_all_services(self) -> None:
        """Stop all running model services"""
        self.running = False
        
        # Stop all model services
        for model_id in list(self.processes.keys()):
            self.stop_model_service(model_id)
    
    def _monitor_services(self) -> None:
        """Monitor running model services and restart if needed"""
        while self.running:
            for model_id in list(self.processes.keys()):
                process_info = self.processes[model_id]
                
                # Check if process is still running
                if not process_info.is_running():
                    logger.warning(f"Model service {model_id} stopped unexpectedly")
                    
                    # Auto-restart if needed
                    if self.running and process_info.restart_count < 3:
                        logger.info(f"Restarting model service {model_id}")
                        process_info.restart_count += 1
                        self.start_model_service_subprocess(model_id)
                    else:
                        logger.error(f"Model service {model_id} failed to start after multiple attempts")
            
            # Check every 10 seconds
            time.sleep(10)
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the status of all model services
        
        Returns:
            Dict: Status information
        """
        registry = get_model_registry()
        models = registry.get_all_models()
        
        status = {
            "gateway": {
                "running": self.gateway_thread is not None and self.gateway_thread.is_alive(),
            },
            "models": {}
        }
        
        for model in models:
            process_info = self.processes.get(model.model_id)
            model_status = {
                "id": model.model_id,
                "display_name": model.display_name,
                "status": model.status,
                "endpoint": f"{model.endpoint.host}:{model.endpoint.port}" if model.endpoint else None,
                "process_running": process_info is not None and process_info.is_running(),
                "pid": process_info.process.pid if process_info else None,
                "restart_count": process_info.restart_count if process_info else 0,
                "uptime": time.time() - process_info.start_time if process_info else 0
            }
            status["models"][model.model_id] = model_status
        
        return status


# Singleton instance of the model manager
_manager_instance = None

def get_model_manager() -> ModelManager:
    """Get the singleton instance of the model manager"""
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = ModelManager()
    return _manager_instance 