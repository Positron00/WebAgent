"""
Simplified Tests for the model manager.

This module contains basic tests for the ModelManager class.
"""
import unittest
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.models.model_manager import ModelManager, ModelProcessInfo


class TestModelProcessInfo(unittest.TestCase):
    """Test the ModelProcessInfo class"""
    
    def setUp(self):
        """Set up test fixtures"""
        mock_process = MagicMock()
        mock_process.pid = 12345
        mock_process.poll.return_value = None  # Process is running
        
        self.process_info = ModelProcessInfo(
            model_id="test-model",
            process=mock_process,
            config={
                "model_id": "test-model",
                "model_type": "transformer",
                "host": "localhost",
                "port": 8501
            }
        )
    
    def test_is_running_true(self):
        """Test is_running when process is running"""
        # Mock process.poll to return None (process is running)
        self.process_info.process.poll.return_value = None
        
        # Check if is_running returns True
        self.assertTrue(self.process_info.is_running())
    
    def test_is_running_false(self):
        """Test is_running when process is not running"""
        # Mock process.poll to return exit code (process is not running)
        self.process_info.process.poll.return_value = 1
        # Reset exit_code to ensure it gets set by the method
        self.process_info.exit_code = None
        # Ensure returncode is set to the same value
        self.process_info.process.returncode = 1
        
        # Check if is_running returns False
        self.assertFalse(self.process_info.is_running())
        
        # Check if exit_code was set
        self.assertEqual(self.process_info.exit_code, 1)


class TestModelManager(unittest.TestCase):
    """Test the model manager functionality"""

    def setUp(self):
        """Set up test fixtures"""
        # Create a mock config
        self.mock_config = {
            "gateway": {
                "host": "localhost",
                "port": 8000
            },
            "models": {
                "test-model": {
                    "model_id": "test-model",
                    "display_name": "Test Model",
                    "model_type": "transformer",
                    "model_path": "/path/to/model",
                    "host": "localhost",
                    "port": 8501,
                    "parameters": {
                        "max_tokens": 100,
                        "temperature": 0.7
                    }
                }
            }
        }
        
        # Mock load_models_config
        self.config_patch = patch("backend.models.model_manager.load_models_config", 
                                 return_value=self.mock_config)
        self.mock_load_config = self.config_patch.start()
        
        # Mock registry
        self.registry_patch = patch("backend.models.model_manager.get_model_registry")
        self.mock_registry_func = self.registry_patch.start()
        self.mock_registry = MagicMock()
        self.mock_registry_func.return_value = self.mock_registry
    
    def tearDown(self):
        """Clean up after tests"""
        self.config_patch.stop()
        self.registry_patch.stop()
    
    def test_initialization(self):
        """Test model manager initialization"""
        manager = ModelManager()
        
        # Check if manager is initialized correctly
        self.assertEqual(manager.config, self.mock_config)
        self.assertIsNone(manager.gateway)
        self.assertIsNone(manager.gateway_thread)
        self.assertFalse(manager.running)
        self.assertEqual(len(manager.processes), 0)


if __name__ == "__main__":
    unittest.main() 