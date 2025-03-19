"""
Diagnostics Test Suite

This package contains tests for the diagnostic tools in WebAgent.

These tests verify the functionality of the system diagnostic tools,
including health checks, monitoring systems, and testing frameworks.
"""

import os
import sys

# Add the parent directory to the path so that imports work correctly
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Set up path for consistent imports
import os
import sys
from pathlib import Path

# Add the backend directory to the path
backend_dir = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if backend_dir.name == 'backend' and str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

# Function to get the root directory of the project
def get_project_root() -> Path:
    """Returns the absolute path to the project root directory."""
    return Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) 