"""
WebAgent Diagnostics Module

This module provides diagnostic tools for the WebAgent platform:

- runners/: Diagnostic runner scripts to execute different diagnostic tests
  - run_all.py: Run all diagnostics and tests
  - run_diagnostics.py: Run system diagnostics only
  - run_tests.py: Run test suite only

- system_check.py: Basic system health and dependency check
- core.py: Core diagnostic functionality and utilities

Diagnostic tests have been moved to backend/tests/diagnostics/

Can be run from:
- WebAgent root: python -m backend.diagnostics.runners.run_all
- Backend directory: python diagnostics/runners/run_all.py
"""

import logging
import sys
import os
from pathlib import Path

# Try to import path_setup for more robust import handling
current_dir = Path(__file__).resolve().parent
backend_dir = current_dir.parent

# Add current directory to sys.path
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Try to use path_setup if available
try:
    # Try to import from the backend directory
    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))
        
    from path_setup import setup_path
    setup_path()
except ImportError:
    # Fallback to manual path setup
    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))
    
    # Add current working directory to path
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)

# Configure logger
logger = logging.getLogger(__name__)
logger.info("Diagnostics module initialized")

# Import key components for easier access
try:
    # Try relative import first
    from .core import get_diagnostics, print_diagnostics_report, run_all_tests
except ImportError:
    try:
        # Try absolute import if relative fails
        from backend.diagnostics.core import get_diagnostics, print_diagnostics_report, run_all_tests
    except ImportError:
        logger.error("Failed to import diagnostics core components. Import errors may occur.")
        # Define placeholder functions to avoid errors
        def get_diagnostics():
            raise ImportError("Diagnostics core module could not be imported")
            
        def print_diagnostics_report(*args, **kwargs):
            raise ImportError("Diagnostics core module could not be imported")
            
        def run_all_tests(*args, **kwargs):
            raise ImportError("Diagnostics core module could not be imported") 