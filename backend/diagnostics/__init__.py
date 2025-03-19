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

def setup_paths():
    """
    Set up the Python path to find modules regardless of where the script is run from.
    
    This function can be used by any diagnostic script to ensure proper import paths
    regardless of whether the script is run:
    - Directly from the backend directory
    - From the WebAgent root directory
    - As a module with python -m
    
    Returns:
        backend_dir: Path object for the backend directory
    """
    script_dir = Path(os.path.abspath(os.path.dirname(__file__)))
    working_dir = Path(os.getcwd())
    
    # If we're in a 'diagnostics' directory, go up one level to find backend
    if script_dir.name == 'diagnostics':
        backend_dir = script_dir.parent
    else:
        backend_dir = working_dir
        
        # Try to find backend directory
        if working_dir.name == 'backend':
            backend_dir = working_dir
        elif 'backend' in working_dir.parts:
            # Navigate up until we find backend
            backend_dir = working_dir
            while backend_dir.name != 'backend' and backend_dir != backend_dir.parent:
                backend_dir = backend_dir.parent
        else:
            # We might be in the project root
            backend_dir = working_dir / 'backend'
            if not backend_dir.exists():
                # Last resort - try relative to script
                backend_dir = script_dir.parent
    
    # Add backend dir to path
    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))
        
    # Add project root (parent of backend) if relevant
    project_root = backend_dir.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
        
    return backend_dir

# Exports
__all__ = ['setup_paths'] 