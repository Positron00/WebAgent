#!/usr/bin/env python
"""
Base Model Service - Abstract base class for model services

This module provides a foundation for all model service implementations
with common functionality for model loading, inference, and health checks.
"""
import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any
import json
import signal
import threading
from datetime import datetime

from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from ..registry import ModelInfo, ModelEndpoint, get_model_registry

logger = logging.getLogger(__name__)

class CompletionRequest(BaseModel):
    """Common request format for model completions"""
    prompt: str = Field(..., description="The input prompt for the model")
    max_tokens: int = Field(100, description="Maximum number of tokens to generate")
    temperature: float = Field(0.7, description="The randomness of the generation")
    top_p: float = Field(0.9, description="Nucleus sampling parameter")
    stop: Optional[List[str]] = Field(None, description="Stop sequences to end generation")
    
    model_config = {
        "protected_namespaces": ()
    }

class CompletionResponse(BaseModel):
    """Common response format for model completions"""
    text: str = Field(..., description="The generated text")
    model: str = Field(..., description="The model used for generation")
    usage: Dict[str, int] = Field(..., description="Token usage information")
    finish_reason: str = Field("stop", description="The reason generation stopped")
    
    model_config = {
        "protected_namespaces": ()
    }

class ModelServiceConfig(BaseModel):
    """Configuration for a model service"""
    model_id: str = Field(..., description="Unique identifier for the model")
    display_name: str = Field(..., description="Display name for the model")
    model_type: str = Field(..., description="Type of model")
    host: str = Field("localhost", description="Host to run the service on")
    port: int = Field(..., description="Port to run the service on")
    model_path: str = Field(..., description="Path to the model files")
    device: str = Field("cpu", description="Device to run the model on (cpu, cuda)")
    capabilities: List[str] = Field(default_factory=list, description="Model capabilities")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Additional model parameters")
    
    model_config = {
        "protected_namespaces": ()
    }

class BaseModelService(ABC):
    """Base class for all model services"""
    
    def __init__(self, config: ModelServiceConfig):
        """Initialize the model service"""
        self.config = config
        self.model = None
        self.tokenizer = None
        self.app = FastAPI(
            title=f"{config.display_name} API",
            description=f"API for the {config.display_name} model",
            version="1.0.0"
        )
        self.status = "initializing"
        self.start_time = datetime.now()
        self.request_count = 0
        self.error_count = 0
        self.last_error = None
        self._setup_api()
        self._setup_cors()
        self._register_in_registry()
        self._setup_shutdown_handler()
    
    def _setup_api(self) -> None:
        """Set up the FastAPI routes"""
        
        @self.app.get("/", tags=["Status"])
        async def root():
            """Return service information"""
            return {
                "name": self.config.display_name,
                "model_id": self.config.model_id,
                "model_type": self.config.model_type,
                "status": self.status,
                "uptime": str(datetime.now() - self.start_time),
                "requests": self.request_count,
                "errors": self.error_count
            }
        
        @self.app.get("/health", tags=["Status"])
        async def health_check():
            """Health check endpoint"""
            if self.status != "ready":
                return Response(
                    content=json.dumps({"status": self.status}),
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    media_type="application/json"
                )
            return {"status": "ready", "model_id": self.config.model_id}
        
        @self.app.post("/v1/completions", response_model=CompletionResponse, tags=["Inference"])
        async def generate_completion(request: CompletionRequest, background_tasks: BackgroundTasks):
            """Generate text completion"""
            if self.status != "ready":
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=f"Service is not ready: {self.status}"
                )
            
            self.request_count += 1
            try:
                start_time = time.time()
                result = await self.generate(
                    prompt=request.prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    stop=request.stop
                )
                logger.info(f"Generation took {time.time() - start_time:.2f}s")
                return result
            except Exception as e:
                self.error_count += 1
                self.last_error = str(e)
                logger.exception(f"Error during generation: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Error during generation: {str(e)}"
                )
            
        @self.app.get("/metrics", tags=["Status"])
        async def metrics():
            """Return service metrics"""
            return {
                "status": self.status,
                "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
                "request_count": self.request_count,
                "error_count": self.error_count,
                "last_error": self.last_error,
                "model_id": self.config.model_id,
                "model_type": self.config.model_type
            }
    
    def _setup_cors(self) -> None:
        """Set up CORS middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # In production, restrict this to specific domains
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def _register_in_registry(self) -> None:
        """Register this service in the model registry"""
        registry = get_model_registry()
        
        endpoint = ModelEndpoint(
            service_name=f"{self.config.model_id}-service",
            host=self.config.host,
            port=self.config.port,
            url_path="/v1/completions"
        )
        
        model_info = ModelInfo(
            model_id=self.config.model_id,
            display_name=self.config.display_name,
            model_type=self.config.model_type,
            description=f"Model service for {self.config.display_name}",
            endpoint=endpoint,
            capabilities=set(self.config.capabilities),
            status=self.status
        )
        
        registry.register_model(model_info)
        logger.info(f"Registered model {self.config.model_id} in registry")
    
    def _update_status(self, status: str) -> None:
        """Update the service status"""
        self.status = status
        registry = get_model_registry()
        registry.update_model_status(self.config.model_id, status)
        logger.info(f"Updated status of {self.config.model_id} to {status}")
    
    def _setup_shutdown_handler(self) -> None:
        """Set up signal handlers for graceful shutdown"""
        def handle_exit(signum, frame):
            """Handle shutdown signal"""
            logger.info(f"Received signal {signum}, shutting down...")
            self._cleanup()
            registry = get_model_registry()
            registry.update_model_status(self.config.model_id, "offline")
            os._exit(0)
        
        signal.signal(signal.SIGINT, handle_exit)
        signal.signal(signal.SIGTERM, handle_exit)
    
    async def startup(self) -> None:
        """Start the model service"""
        logger.info(f"Starting {self.config.model_id} service...")
        try:
            self._update_status("loading")
            await self.load_model()
            self._update_status("ready")
            logger.info(f"{self.config.model_id} service is ready")
        except Exception as e:
            self._update_status("error")
            self.last_error = str(e)
            logger.exception(f"Error starting {self.config.model_id} service: {e}")
            raise
    
    def run(self) -> None:
        """Run the model service"""
        # Start the model loading in a background thread
        threading.Thread(target=self._async_startup, daemon=True).start()
        
        # Run the FastAPI app
        uvicorn.run(
            self.app,
            host=self.config.host,
            port=self.config.port,
            log_level="info"
        )
    
    def _async_startup(self) -> None:
        """Run startup in a background thread"""
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.startup())
        except Exception as e:
            logger.exception(f"Error in async startup: {e}")
        finally:
            loop.close()
    
    def _cleanup(self) -> None:
        """Clean up resources before shutdown"""
        try:
            logger.info(f"Cleaning up resources for {self.config.model_id}...")
            self.cleanup()
            logger.info(f"Cleanup completed for {self.config.model_id}")
        except Exception as e:
            logger.exception(f"Error during cleanup: {e}")
    
    @abstractmethod
    async def load_model(self) -> None:
        """
        Load the model into memory
        
        This method should be implemented by each model service class.
        """
        pass
    
    @abstractmethod
    async def generate(
        self, 
        prompt: str, 
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[List[str]] = None
    ) -> CompletionResponse:
        """
        Generate text completion
        
        This method should be implemented by each model service class.
        
        Args:
            prompt: The input prompt
            max_tokens: Maximum number of tokens to generate
            temperature: The randomness of the generation
            top_p: Nucleus sampling parameter
            stop: Stop sequences to end generation
            
        Returns:
            CompletionResponse: The generated completion
        """
        pass
    
    def cleanup(self) -> None:
        """
        Clean up resources before shutdown
        
        This method can be overridden by model service classes to perform
        cleanup operations (e.g., free GPU memory).
        """
        pass 