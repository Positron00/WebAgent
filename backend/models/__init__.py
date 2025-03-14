"""
Models Package

This package provides a microservices framework for managing and running different types of models.
It includes model discovery, configuration, and a REST API for interacting with the models.
"""

from backend.models.config import (
    load_models_config,
    DEFAULT_MODEL_CONFIG,
)
from backend.models.registry import (
    ModelRegistry,
    get_model_registry,
    ModelInfo,
    ModelEndpoint,
)
from backend.models.model_manager import ModelManager
from backend.models.run_model_service import run_model_service

__all__ = [
    # Configuration
    'load_models_config',
    'DEFAULT_MODEL_CONFIG',
    
    # Registry
    'ModelRegistry',
    'get_model_registry',
    'ModelInfo',
    'ModelEndpoint',
    
    # Management
    'ModelManager',
    'run_model_service',
] 