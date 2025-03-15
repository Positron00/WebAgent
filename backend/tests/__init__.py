"""
Test Package
===========

This package contains test modules for the WebAgent platform.
"""

# Set up path for consistent imports
import os
import sys
from pathlib import Path

# Add the backend directory to the path
backend_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if backend_dir.name == 'backend' and str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

# Function to get the root directory of the project
def get_project_root() -> Path:
    """Returns the absolute path to the project root directory."""
    return Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import common test utilities so they can be imported from tests directly
from .mock_utils import create_mock_agent, create_test_workflow_state, setup_test_environment 