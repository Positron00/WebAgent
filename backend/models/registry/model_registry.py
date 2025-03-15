#!/usr/bin/env python
"""
Model Registry - Keeps track of all available models in the system.

This registry stores information about model endpoints, status, and capabilities
and provides a discovery mechanism for clients to locate appropriate models.
"""
import logging
from typing import Dict, List, Optional, Set
import os
import json
from datetime import datetime
from pydantic import BaseModel, Field, validator
import threading

logger = logging.getLogger(__name__)

class ModelEndpoint(BaseModel):
    """Information about a model endpoint"""
    service_name: str = Field(..., description="Service name for the model")
    host: str = Field(..., description="Host address where the model is running")
    port: int = Field(..., description="Port number for the model service")
    url_path: str = Field("/v1/completions", description="API endpoint path")
    
    model_config = {
        "protected_namespaces": ()
    }
    
    @property
    def endpoint_url(self) -> str:
        """Generate the full URL for the model endpoint"""
        return f"http://{self.host}:{self.port}{self.url_path}"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format"""
        try:
            # Try using model_dump (Pydantic v2)
            result = self.model_dump(exclude={"url_path"})
        except AttributeError:
            # Fallback to dict (Pydantic v1)
            result = self.dict(exclude={"url_path"})
            
        result["url_path"] = self.url_path
        result["endpoint_url"] = self.endpoint_url
        return result

class ModelInfo(BaseModel):
    """Information about a model"""
    model_id: str = Field(..., description="Unique identifier for the model")
    display_name: str = Field(..., description="Display name for the model")
    model_type: str = Field(..., description="Type of model (e.g., 'llm', 'embedding', 'classification')")
    description: str = Field("", description="Description of the model")
    version: str = Field("1.0.0", description="Model version")
    endpoint: ModelEndpoint = Field(..., description="Endpoint information")
    parameters: Dict = Field(default_factory=dict, description="Model parameters")
    capabilities: Set[str] = Field(default_factory=set, description="Model capabilities")
    status: str = Field("offline", description="Model status (online/offline/error)")
    last_check: Optional[datetime] = Field(None, description="Last status check time")
    
    model_config = {
        "protected_namespaces": ()
    }
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format"""
        try:
            # Try using model_dump (Pydantic v2)
            result = self.model_dump(exclude={"endpoint", "capabilities", "last_check"})
        except AttributeError:
            # Fallback to dict (Pydantic v1)
            result = self.dict(exclude={"endpoint", "capabilities", "last_check"})
            
        result["endpoint"] = self.endpoint.to_dict()
        result["capabilities"] = list(self.capabilities)
        result["last_check"] = self.last_check.isoformat() if self.last_check else None
        return result

class ModelRegistry:
    """Registry that maintains information about all available models"""
    
    def __init__(self, registry_path: str = None):
        """Initialize the registry"""
        self.models: Dict[str, ModelInfo] = {}
        self._lock = threading.RLock()
        self.registry_path = registry_path or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            "models_registry.json"
        )
        self._load_registry()
    
    def register_model(self, model_info: ModelInfo) -> bool:
        """
        Register a new model or update an existing one
        
        Args:
            model_info: Information about the model
            
        Returns:
            bool: True if the registration was successful
        """
        with self._lock:
            self.models[model_info.model_id] = model_info
            self._save_registry()
            logger.info(f"Registered model: {model_info.model_id} ({model_info.display_name})")
            return True
    
    def unregister_model(self, model_id: str) -> bool:
        """
        Unregister a model from the registry
        
        Args:
            model_id: ID of the model to unregister
            
        Returns:
            bool: True if the model was unregistered, False if not found
        """
        with self._lock:
            if model_id in self.models:
                del self.models[model_id]
                self._save_registry()
                logger.info(f"Unregistered model: {model_id}")
                return True
            return False
    
    def get_model(self, model_id: str) -> Optional[ModelInfo]:
        """
        Get information about a specific model
        
        Args:
            model_id: ID of the model to retrieve
            
        Returns:
            ModelInfo or None: Information about the model, or None if not found
        """
        return self.models.get(model_id)
    
    def get_all_models(self) -> List[ModelInfo]:
        """
        Get information about all registered models
        
        Returns:
            List[ModelInfo]: List of all registered models
        """
        return list(self.models.values())
    
    def get_models_by_type(self, model_type: str) -> List[ModelInfo]:
        """
        Get all models of a specific type
        
        Args:
            model_type: Type of models to retrieve
            
        Returns:
            List[ModelInfo]: List of models of the specified type
        """
        return [model for model in self.models.values() if model.model_type == model_type]
    
    def get_models_by_capability(self, capability: str) -> List[ModelInfo]:
        """
        Get all models that have a specific capability
        
        Args:
            capability: Capability to filter by
            
        Returns:
            List[ModelInfo]: List of models with the specified capability
        """
        return [
            model for model in self.models.values() 
            if capability in model.capabilities
        ]
    
    def update_model_status(self, model_id: str, status: str) -> bool:
        """
        Update the status of a model
        
        Args:
            model_id: ID of the model to update
            status: New status ('online', 'offline', or 'error')
            
        Returns:
            bool: True if the status was updated, False if model not found
        """
        with self._lock:
            if model_id in self.models:
                self.models[model_id].status = status
                self.models[model_id].last_check = datetime.now()
                self._save_registry()
                return True
            return False
    
    def _save_registry(self) -> None:
        """Save the registry to disk"""
        try:
            with open(self.registry_path, 'w') as f:
                registry_data = {
                    model_id: model.to_dict() 
                    for model_id, model in self.models.items()
                }
                json.dump(registry_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save model registry: {e}")
    
    def _load_registry(self) -> None:
        """Load the registry from disk"""
        try:
            if os.path.exists(self.registry_path):
                with open(self.registry_path, 'r') as f:
                    registry_data = json.load(f)
                    
                for model_id, model_data in registry_data.items():
                    endpoint_data = model_data.pop("endpoint")
                    model_data["endpoint"] = ModelEndpoint(**endpoint_data)
                    
                    # Handle capabilities (convert list back to set)
                    capabilities = model_data.pop("capabilities", [])
                    
                    # Handle last_check datetime
                    last_check = model_data.pop("last_check", None)
                    if last_check:
                        model_data["last_check"] = datetime.fromisoformat(last_check)
                    
                    # Remove model_id from model_data to prevent duplicate argument
                    model_data.pop("model_id", None)
                    
                    model_info = ModelInfo(
                        model_id=model_id,
                        capabilities=set(capabilities),
                        **model_data
                    )
                    self.models[model_id] = model_info
                    
                logger.info(f"Loaded {len(self.models)} models from registry")
        except Exception as e:
            logger.error(f"Failed to load model registry: {e}")


# Singleton instance of the model registry
_registry_instance = None

def get_model_registry() -> ModelRegistry:
    """Get the singleton instance of the model registry"""
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = ModelRegistry()
    return _registry_instance 