"""
Tests for the model manager.

This module contains tests for the ModelManager class that orchestrates model services.
"""
import unittest
import os
import sys
import json
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
        mock_process.returncode = None
        
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
        
        # Check if poll was called
        self.process_info.process.poll.assert_called_once()
    
    def test_is_running_false(self):
        """Test is_running when process is not running"""
        # Mock process.poll to return exit code (process is not running)
        self.process_info.process.poll.return_value = 1
        self.process_info.process.returncode = 1
        self.process_info.exit_code = None  # Reset to ensure it gets set
        
        # Check if is_running returns False
        self.assertFalse(self.process_info.is_running())
        
        # Check if poll was called
        self.process_info.process.poll.assert_called_once()
        
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
        
        # Create patches
        self.subprocess_patch = patch("backend.models.model_manager.subprocess")
        self.mock_subprocess = self.subprocess_patch.start()
        
        # Mock subprocess.Popen
        self.mock_process = MagicMock()
        self.mock_process.pid = 12345
        self.mock_process.poll.return_value = None  # Process is running
        self.mock_subprocess.Popen.return_value = self.mock_process
        
        # Mock os.kill
        self.os_patch = patch("backend.models.model_manager.os")
        self.mock_os = self.os_patch.start()
        self.mock_os.SIGTERM = 15  # Define SIGTERM constant
        
        # Mock load_models_config
        self.config_patch = patch("backend.models.model_manager.load_models_config", 
                                 return_value=self.mock_config)
        self.mock_load_config = self.config_patch.start()
        
        # Mock path exists
        self.path_patch = patch("backend.models.model_manager.Path.exists", return_value=True)
        self.mock_path_exists = self.path_patch.start()
        
        # Mock registry
        self.registry_patch = patch("backend.models.model_manager.get_model_registry")
        self.mock_registry_func = self.registry_patch.start()
        self.mock_registry = MagicMock()
        self.mock_registry_func.return_value = self.mock_registry
        
        # Mock the _start_log_thread method to prevent infinite loop
        self.log_thread_patch = patch.object(ModelManager, "_start_log_thread")
        self.mock_start_log = self.log_thread_patch.start()
    
    def tearDown(self):
        """Clean up after tests"""
        self.subprocess_patch.stop()
        self.os_patch.stop()
        self.config_patch.stop()
        self.path_patch.stop()
        self.registry_patch.stop()
        self.log_thread_patch.stop()
    
    def test_initialization(self):
        """Test model manager initialization"""
        manager = ModelManager()
        
        # Check if manager is initialized correctly
        self.assertEqual(manager.config, self.mock_config)
        self.assertTrue(manager.running)
        self.assertEqual(len(manager.processes), 0)
    
    def test_start_gateway(self):
        """Test starting the API gateway"""
        # Setup mock gateway
        mock_gateway = MagicMock()
        
        # Patch the ModelAPIGateway class
        with patch("backend.models.model_manager.ModelAPIGateway", return_value=mock_gateway) as mock_api_gateway, \
             patch("backend.models.model_manager.threading.Thread") as mock_thread:
            
            # Create a mock thread instance
            mock_thread_instance = MagicMock()
            mock_thread.return_value = mock_thread_instance
            
            # Create the manager and start the gateway
            manager = ModelManager()
            manager.start_gateway(host="localhost", port=8000)
            
            # Check if gateway was created correctly
            mock_api_gateway.assert_called_once_with("localhost", 8000)
            self.assertEqual(manager.gateway, mock_gateway)
            
            # Check if thread was created and started
            mock_thread.assert_called_once()
            mock_thread_instance.start.assert_called_once()
    
    def test_start_model_service_subprocess(self):
        """Test starting a model service subprocess"""
        manager = ModelManager()
        
        # Test starting a model service subprocess
        result = manager.start_model_service_subprocess("test-model")
        
        # Check if subprocess was started correctly
        self.assertTrue(result)
        self.mock_subprocess.Popen.assert_called_once()
        self.assertIn("test-model", manager.processes)
        self.mock_start_log.assert_called_once_with("test-model")
    
    def test_stop_model_service(self):
        """Test stopping a model service"""
        manager = ModelManager()
        
        # Add a mock process to the manager
        mock_process = MagicMock()
        mock_process.pid = 12345
        process_info = ModelProcessInfo(
            model_id="test-model",
            process=mock_process,
            config=self.mock_config["models"]["test-model"]
        )
        manager.processes["test-model"] = process_info
        
        # Test stopping a model service
        result = manager.stop_model_service("test-model")
        
        # Check if model service was stopped correctly
        self.assertTrue(result)
        self.mock_os.kill.assert_called_with(12345, 15)  # 15 is SIGTERM


if __name__ == "__main__":
    unittest.main() 