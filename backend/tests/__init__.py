"""
WebAgent Tests Package

This package contains all tests for the WebAgent platform:
- Unit tests: focused on testing individual components
- Integration tests: testing multiple components working together
- Diagnostics tests: testing diagnostic capabilities

Tests can be run from:
- WebAgent root: python -m pytest backend/tests
- Backend directory: python -m pytest tests
"""

import sys
import os
from pathlib import Path

# Configure paths for imports
def setup_test_paths():
    """
    Set up the Python path correctly for tests to run from either
    the WebAgent root directory or the backend directory.
    """
    # Get the current directory (should be the tests directory)
    current_dir = Path(__file__).resolve().parent
    
    # The backend directory should be the parent of the tests directory
    backend_dir = current_dir.parent
    
    # Add backend to path if not already there
    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))
    
    # Try to use path_setup if available
    try:
        from path_setup import setup_path
        setup_path()
        return True
    except ImportError:
        # Make sure current directory is in path
        cwd = os.getcwd()
        if cwd not in sys.path:
            sys.path.insert(0, cwd)
        return False

# Run the path setup when the module is imported
setup_test_paths()

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