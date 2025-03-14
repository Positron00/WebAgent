"""
Model Registry Package

This package provides functionality for tracking and discovering model services.
"""

from .model_registry import ModelRegistry, ModelInfo, ModelEndpoint, get_model_registry

__all__ = ['ModelRegistry', 'ModelInfo', 'ModelEndpoint', 'get_model_registry'] 