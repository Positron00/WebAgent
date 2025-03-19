"""
WebAgent Tests Package
=====================

This package contains all the test modules and utilities for the WebAgent backend.
"""

import sys
import os
from pathlib import Path

def setup_paths():
    """
    Set up the Python path to find modules regardless of where the script is run from.
    
    This function can be used by any test script to ensure proper import paths
    regardless of whether the script is run:
    - Directly from the backend directory
    - From the WebAgent root directory
    - As a module with python -m
    
    Returns:
        backend_dir: Path object for the backend directory
    """
    script_dir = Path(os.path.abspath(os.path.dirname(__file__)))
    working_dir = Path(os.getcwd())
    
    # If we're in a 'tests' directory, go up one level to find backend
    if script_dir.name == 'tests':
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

def import_helper(module_path, fallback_path=None):
    """
    Helper function to import modules in a way that works from either 
    the WebAgent root or backend directory.
    
    Args:
        module_path: Path to the module using app/backend.app style
        fallback_path: Alternative path to try if first one fails
        
    Returns:
        Imported module or None if import failed
    """
    try:
        # Try first path
        module_parts = module_path.split('.')
        if module_parts[0] == 'app':
            # Try without backend prefix first
            try:
                return __import__('.'.join(module_parts), fromlist=['*'])
            except ImportError:
                # Try with backend prefix
                return __import__('backend.' + '.'.join(module_parts), fromlist=['*'])
        else:
            # Try as-is first
            try:
                return __import__(module_path, fromlist=['*'])
            except ImportError as e:
                if fallback_path:
                    return __import__(fallback_path, fromlist=['*'])
                raise e
    except ImportError as e:
        if fallback_path:
            try:
                return __import__(fallback_path, fromlist=['*'])
            except ImportError:
                return None
        return None

# Function to get the root directory of the project
def get_project_root() -> Path:
    """Returns the absolute path to the project root directory."""
    return Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Make test utilities available at module level
try:
    from tests.mock_utils import create_mock_agent, create_test_workflow_state, setup_test_environment
except ImportError:
    try:
        from backend.tests.mock_utils import create_mock_agent, create_test_workflow_state, setup_test_environment
    except ImportError:
        # Define placeholder functions in case imports fail
        def create_mock_agent(*args, **kwargs):
            raise ImportError("Mock utilities could not be imported")
            
        def create_test_workflow_state(*args, **kwargs):
            raise ImportError("Mock utilities could not be imported")
            
        def setup_test_environment(*args, **kwargs):
            raise ImportError("Mock utilities could not be imported") 