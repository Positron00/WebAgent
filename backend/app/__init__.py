"""
WebAgent App Package
==================

This package contains the core components of the WebAgent platform.
"""

# Import the path setup to ensure imports work correctly
import os
import sys
from pathlib import Path

# Add the backend directory to the path if needed
backend_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if backend_dir.name == 'backend' and str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

# Import version information
__version__ = "2.5.8"  # Released on 2025-03-14
