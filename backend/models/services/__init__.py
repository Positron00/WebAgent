"""
Model Services Package

This package provides service implementations for various models.
"""

from .base_model_service import (
    BaseModelService, 
    ModelServiceConfig, 
    CompletionRequest, 
    CompletionResponse
)
from .transformer_model_service import TransformerModelService

try:
    from .core_transformer_service import CoreTransformerModelService
except ImportError as exc:
    import logging

    logging.getLogger(__name__).warning(
        "Core transformer model service unavailable: %s", exc
    )
    CoreTransformerModelService = None  # type: ignore[misc, assignment]

__all__ = [
    'BaseModelService', 
    'ModelServiceConfig', 
    'CompletionRequest', 
    'CompletionResponse',
    'TransformerModelService'
] 