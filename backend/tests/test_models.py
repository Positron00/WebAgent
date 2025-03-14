"""
Test modules for the model microservices framework.

This module contains tests for:
- Model Registry
- Model API Gateway
- Model Services
- Model Manager
"""
import unittest
import os
import sys
import json
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.models.registry.model_registry import (
    ModelRegistry,
    get_model_registry,
    ModelInfo,
    ModelEndpoint,
)
from backend.models.config.models_config import load_models_config, DEFAULT_MODEL_CONFIG
from backend.models.api.gateway import ModelAPIGateway, CompletionRequest, CompletionResponse, ModelListResponse


class AsyncTestCase(unittest.TestCase):
    """Base class for async test cases"""
    
    def run_async(self, coro):
        """Run a coroutine in the event loop"""
        return asyncio.run(coro)


class TestModelRegistry(unittest.TestCase):
    """Test the model registry functionality"""

    def setUp(self):
        """Set up test fixtures"""
        # Create a fresh registry for each test
        self.registry = ModelRegistry()
    
    def test_singleton_pattern(self):
        """Test that get_model_registry returns the same instance"""
        registry1 = get_model_registry()
        registry2 = get_model_registry()
        self.assertIs(registry1, registry2)
    
    def test_register_model(self):
        """Test registering a model"""
        endpoint = ModelEndpoint(
            service_name="test-service",
            host="localhost",
            port=8500,
            url_path="/v1/completions"
        )
        
        model_info = ModelInfo(
            model_id="test-model",
            display_name="Test Model",
            model_type="llm",
            endpoint=endpoint,
            capabilities={"text-generation"},
            status="ready"
        )
        
        self.registry.register_model(model_info)
        
        # Check if model is registered
        self.assertIn("test-model", self.registry.models)
        
        # Check if model info is correct
        retrieved_info = self.registry.get_model("test-model")
        self.assertEqual(retrieved_info.model_id, "test-model")
        self.assertEqual(retrieved_info.model_type, "llm")
        self.assertEqual(retrieved_info.endpoint.host, "localhost")
        self.assertEqual(retrieved_info.endpoint.port, 8500)
    
    def test_unregister_model(self):
        """Test unregistering a model"""
        # First register a model
        endpoint = ModelEndpoint(
            service_name="test-service",
            host="localhost",
            port=8500,
            url_path="/v1/completions"
        )
        
        model_info = ModelInfo(
            model_id="test-model",
            display_name="Test Model",
            model_type="llm",
            endpoint=endpoint,
            capabilities={"text-generation"},
            status="ready"
        )
        
        self.registry.register_model(model_info)
        
        # Now unregister it
        self.registry.unregister_model("test-model")
        
        # Check if model is unregistered
        self.assertNotIn("test-model", self.registry.models)
    
    def test_update_model_status(self):
        """Test updating model status"""
        # First register a model
        endpoint = ModelEndpoint(
            service_name="test-service",
            host="localhost",
            port=8500,
            url_path="/v1/completions"
        )
        
        model_info = ModelInfo(
            model_id="test-model",
            display_name="Test Model",
            model_type="llm",
            endpoint=endpoint,
            capabilities={"text-generation"},
            status="loading"
        )
        
        self.registry.register_model(model_info)
        
        # Update status
        self.registry.update_model_status("test-model", "ready")
        
        # Check if status is updated
        updated_info = self.registry.get_model("test-model")
        self.assertEqual(updated_info.status, "ready")


class TestModelConfig(unittest.TestCase):
    """Test model configuration functionality"""
    
    def test_default_model_config(self):
        """Test default model configuration"""
        # Check if default configuration has expected keys
        self.assertIn("parameters", DEFAULT_MODEL_CONFIG)
        self.assertIn("device", DEFAULT_MODEL_CONFIG)
        self.assertIn("capabilities", DEFAULT_MODEL_CONFIG)


class TestModelAPIGateway(AsyncTestCase):
    """Test the model API gateway"""
    
    def setUp(self):
        """Set up the test fixtures"""
        # Create a mock registry
        self.mock_registry = MagicMock()
        
        # Create a gateway with the mock registry
        with patch("backend.models.api.gateway.get_model_registry", return_value=self.mock_registry):
            self.gateway = ModelAPIGateway(host="localhost", port=8000)
    
    def test_api_setup(self):
        """Test that the API routes are set up correctly"""
        # Check that the app has routes
        self.assertTrue(len(self.gateway.app.routes) > 0)
        
        # Check that the expected endpoints exist
        route_paths = [route.path for route in self.gateway.app.routes]
        self.assertIn("/", route_paths)
        self.assertIn("/health", route_paths)
        self.assertIn("/v1/models", route_paths)
        self.assertIn("/v1/completions", route_paths)


if __name__ == "__main__":
    unittest.main() 