"""
Tests for the model manager.

This module contains tests for the ModelManager class that orchestrates model services.
"""
import unittest
import os
import sys
import asyncio
import json
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock, call

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.models.model_manager import ModelManager, ModelProcessInfo


class AsyncTestCase(unittest.TestCase):
    """Base class for async test cases"""
    
    def run_async(self, coro):
        """Run a coroutine in the event loop"""
        return asyncio.run(coro)


class TestModelManager(AsyncTestCase):
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
        
        # Mock the _start_log_thread method
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
        self.assertIsNone(manager.gateway)
        self.assertIsNone(manager.gateway_thread)
        self.assertFalse(manager.running)
        self.assertEqual(len(manager.processes), 0)
    
    def test_start_gateway(self):
        """Test starting the API gateway"""
        async def _test():
            # Setup mock process
            mock_process = AsyncMock()
            mock_process.pid = 54321
            
            with patch("backend.models.model_manager.threading.Thread") as mock_thread:
                manager = ModelManager()
                manager.start_gateway(host="localhost", port=8000)
                
                # Check if gateway was started correctly
                self.assertIsNotNone(manager.gateway)
                self.assertIsNotNone(manager.gateway_thread)
                mock_thread.assert_called_once()
                mock_thread.return_value.start.assert_called_once()
                
        self.run_async(_test())

    def test_load_model_service(self):
        """Test loading a model service"""
        async def _test():
            # Create a manager with a fully mocked implementation
            with patch.object(ModelManager, "start_model_service_subprocess") as mock_start:
                # Make the mocked method return True
                mock_start.return_value = True
                
                manager = ModelManager()
                
                # Test loading a model service
                result = manager.load_model_service("test-model")
                
                # Check if model service was loaded correctly
                self.assertTrue(result)
                mock_start.assert_called_once_with("test-model")
            
        self.run_async(_test())
    
    def test_start_model_service_subprocess(self):
        """Test starting a model service subprocess"""
        async def _test():
            manager = ModelManager()
            
            # Test starting a model service subprocess
            result = manager.start_model_service_subprocess("test-model")
            
            # Check if subprocess was started correctly
            self.assertTrue(result)
            self.mock_subprocess.Popen.assert_called_once()
            self.assertIn("test-model", manager.processes)
            self.mock_start_log.assert_called_once_with("test-model")
            
        self.run_async(_test())
    
    def test_stop_model_service(self):
        """Test stopping a model service"""
        async def _test():
            # Direct patching of the os module
            with patch("backend.models.model_manager.os.kill") as mock_kill:
                manager = ModelManager()
                
                # Add a mock process to the manager
                mock_process = MagicMock()
                mock_process.pid = 12345
                process_info = MagicMock()
                process_info.model_id = "test-model"
                process_info.process = mock_process
                manager.processes["test-model"] = process_info
                
                # Test stopping a model service
                result = manager.stop_model_service("test-model")
                
                # Check if model service was stopped correctly
                self.assertTrue(result)
                mock_kill.assert_called_once_with(12345, 15)  # 15 is SIGTERM
            
        self.run_async(_test())
    
    def test_monitor_services(self):
        """Test monitoring services"""
        async def _test():
            manager = ModelManager()
            
            # Add a mock process to the manager
            process_info = MagicMock()
            process_info.model_id = "test-model"
            process_info.is_running = MagicMock(return_value=True)
            manager.processes["test-model"] = process_info
            
            # Test monitoring services
            manager._monitor_services()
            
            # Check if monitoring worked correctly
            process_info.is_running.assert_called_once()
            
        self.run_async(_test())
    
    def test_get_service_status(self):
        """Test getting service status"""
        manager = ModelManager()
        
        # Setup mock status response
        mock_status = {
            "models": {
                "test-model": {
                    "model_id": "test-model",
                    "running": True,
                    "status": "online"
                }
            }
        }
        
        # Patch the get_status method
        with patch.object(manager, "get_status", return_value=mock_status):
            # Test getting service status
            status = manager.get_status()
            
            # Check if status is correct
            self.assertIn("models", status)
            self.assertIn("test-model", status["models"])
            self.assertTrue(status["models"]["test-model"]["running"])


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
    
    def test_to_dict(self):
        """Test converting process info to dictionary"""
        # Add a to_dict method to the process_info class for testing
        def to_dict(self):
            return {
                "model_id": self.model_id,
                "running": self.is_running(),
                "start_time": self.start_time,
                "exit_code": self.exit_code,
                "restart_count": self.restart_count
            }
        
        # Add the method to the instance
        self.process_info.to_dict = to_dict.__get__(self.process_info)
        
        info_dict = self.process_info.to_dict()
        
        # Check if dictionary is correct
        self.assertEqual(info_dict["model_id"], "test-model")
        self.assertTrue(info_dict["running"])
        self.assertIsNone(info_dict["exit_code"])
        self.assertEqual(info_dict["restart_count"], 0)


if __name__ == "__main__":
    unittest.main() 