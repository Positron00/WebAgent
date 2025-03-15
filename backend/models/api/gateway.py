#!/usr/bin/env python
"""
Model API Gateway

This module provides a central API gateway that routes requests to the appropriate
model service and handles service discovery and load balancing.
"""
import logging
import time
from typing import Dict, List, Optional, Any, Union
import json
import asyncio
import os
from datetime import datetime
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from pydantic import BaseModel, Field

from ..registry import get_model_registry, ModelInfo

logger = logging.getLogger(__name__)

class CompletionRequest(BaseModel):
    """Request format for completions"""
    model: str = Field(..., description="Model ID to use for completion")
    prompt: str = Field(..., description="The input prompt for the model")
    max_tokens: int = Field(100, description="Maximum number of tokens to generate")
    temperature: float = Field(0.7, description="The randomness of the generation")
    top_p: float = Field(0.9, description="Nucleus sampling parameter")
    stop: Optional[List[str]] = Field(None, description="Stop sequences to end generation")

class CompletionResponse(BaseModel):
    """Response format for completions"""
    text: str = Field(..., description="The generated text")
    model: str = Field(..., description="The model used for generation")
    usage: Dict[str, int] = Field(..., description="Token usage information")
    finish_reason: str = Field("stop", description="The reason generation stopped")

class ModelListResponse(BaseModel):
    """Response format for model listing"""
    models: List[Dict[str, Any]] = Field(..., description="List of available models")

class ModelAPIGateway:
    """API gateway for model services"""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8000):
        """Initialize the API gateway"""
        self.host = host
        self.port = port
        self.health_check_task = None
        
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Start health check task on startup
            task = asyncio.create_task(self._health_check_loop())
            yield
            # Cancel health check task on shutdown
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        self.app = FastAPI(
            title="Model API Gateway",
            description="API Gateway for model services",
            version="1.0.0",
            lifespan=lifespan
        )
        self.start_time = datetime.now()
        self.request_count = 0
        self.error_count = 0
        self._configure_routes()
        self._configure_middleware()
    
    def _configure_routes(self) -> None:
        """Set up the FastAPI routes"""
        
        @self.app.get("/", tags=["Status"])
        async def root():
            """Return service information"""
            registry = get_model_registry()
            models = registry.get_all_models()
            
            return {
                "name": "Model API Gateway",
                "status": "online",
                "uptime": str(datetime.now() - self.start_time),
                "requests": self.request_count,
                "errors": self.error_count,
                "models_available": len(models),
                "models_online": sum(1 for m in models if m.status == "ready")
            }
        
        @self.app.get("/health", tags=["Status"])
        async def health_check():
            """Health check endpoint"""
            registry = get_model_registry()
            models = registry.get_all_models()
            models_online = sum(1 for m in models if m.status == "ready")
            
            return {
                "status": "online",
                "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
                "models_total": len(models),
                "models_online": models_online
            }
        
        @self.app.get("/v1/models", response_model=ModelListResponse, tags=["Models"])
        async def list_models():
            """List available models"""
            registry = get_model_registry()
            models = registry.get_all_models()
            
            model_list = []
            for model in models:
                model_data = {
                    "id": model.model_id,
                    "name": model.display_name,
                    "type": model.model_type,
                    "status": model.status,
                    "capabilities": list(model.capabilities),
                    "description": model.description
                }
                model_list.append(model_data)
            
            return {"models": model_list}
        
        @self.app.post("/v1/completions", response_model=CompletionResponse, tags=["Generation"])
        async def generate_completion(request: CompletionRequest, background_tasks: BackgroundTasks):
            """Generate text completion"""
            self.request_count += 1
            
            try:
                # Get the model
                registry = get_model_registry()
                model = registry.get_model(request.model)
                
                if model is None:
                    self.error_count += 1
                    raise HTTPException(
                        status_code=404,
                        detail=f"Model not found: {request.model}"
                    )
                
                if model.status != "ready":
                    self.error_count += 1
                    raise HTTPException(
                        status_code=503,
                        detail=f"Model not ready: {request.model} (status: {model.status})"
                    )
                
                # Forward the request to the model service
                result = await self._forward_request(
                    model=model,
                    endpoint="/v1/completions",
                    data={
                        "prompt": request.prompt,
                        "max_tokens": request.max_tokens,
                        "temperature": request.temperature,
                        "top_p": request.top_p,
                        "stop": request.stop
                    }
                )
                
                return result
                
            except HTTPException:
                raise
            except Exception as e:
                self.error_count += 1
                logger.exception(f"Error during completion: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Error during completion: {str(e)}"
                )
            
        @self.app.get("/metrics", tags=["Status"])
        async def metrics():
            """Return service metrics"""
            registry = get_model_registry()
            models = registry.get_all_models()
            
            metrics_data = {
                "service": {
                    "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
                    "request_count": self.request_count,
                    "error_count": self.error_count,
                    "models_total": len(models),
                    "models_online": sum(1 for m in models if m.status == "ready")
                },
                "models": {}
            }
            
            for model in models:
                metrics_data["models"][model.model_id] = {
                    "status": model.status,
                    "last_check": model.last_check.isoformat() if model.last_check else None
                }
            
            return metrics_data
    
    def _configure_middleware(self) -> None:
        """Configure middleware for the API"""
        # CORS middleware for cross-origin requests
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    async def _health_check_loop(self) -> None:
        """Continuously check the health of model services"""
        while True:
            await self._check_model_services()
            await asyncio.sleep(30)  # Check every 30 seconds
    
    async def _check_model_services(self) -> None:
        """Check the health of all model services"""
        registry = get_model_registry()
        models = registry.get_all_models()
        
        for model in models:
            try:
                # Skip models with no endpoint
                if not model.endpoint:
                    continue
                
                # Check the health of the model service
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        f"http://{model.endpoint.host}:{model.endpoint.port}/health",
                        timeout=5.0
                    )
                    
                    if response.status_code == 200:
                        new_status = response.json().get("status", "unknown")
                        if new_status != model.status:
                            registry.update_model_status(model.model_id, new_status)
                    else:
                        # Service is responding but not healthy
                        if model.status != "error":
                            registry.update_model_status(model.model_id, "error")
                            
            except Exception as e:
                # Service is not responding
                if model.status != "offline":
                    registry.update_model_status(model.model_id, "offline")
                    logger.warning(f"Model service {model.model_id} is offline: {e}")
    
    async def _forward_request(
        self,
        model: ModelInfo,
        endpoint: str,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Forward a request to a model service
        
        Args:
            model: Model information
            endpoint: API endpoint to call
            data: Request data
            
        Returns:
            Dict: Response from the model service
        """
        try:
            url = f"http://{model.endpoint.host}:{model.endpoint.port}{endpoint}"
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url,
                    json=data,
                    timeout=60.0  # Longer timeout for model inference
                )
                
                if response.status_code != 200:
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=f"Error from model service: {response.text}"
                    )
                
                return response.json()
                
        except httpx.RequestError as e:
            logger.exception(f"Error forwarding request to {model.model_id}: {e}")
            raise HTTPException(
                status_code=503,
                detail=f"Error communicating with model service: {str(e)}"
            )
    
    def run(self) -> None:
        """Run the API gateway"""
        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )

def run_gateway(host: str = "localhost", port: int = 8000):
    """Run the API gateway"""
    gateway = ModelAPIGateway(host=host, port=port)
    gateway.run()

if __name__ == "__main__":
    print("Starting Model API Gateway...")
    run_gateway() 