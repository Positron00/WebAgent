"""
Tests for the ModelProcessInfo class.

This module contains tests for the ModelProcessInfo class that tracks model process information.
"""
import unittest
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.models.model_manager import ModelProcessInfo


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


if __name__ == "__main__":
    unittest.main() 