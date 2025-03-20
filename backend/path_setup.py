"""
Path Setup Utility
=================

This module helps set up the Python path correctly for importing modules within this project.
It allows code to be run either from the backend directory or from the WebAgent root directory.
"""

import os
import sys
from pathlib import Path
import importlib.util


def setup_path():
    """
    Add the appropriate directories to the Python path for imports to work correctly.
    
    This function:
    1. If running from WebAgent root directory, adds backend/ to the path
    2. Makes sure the current directory is in the path
    3. Sets up path to allow imports from either 'app' or 'backend.app'
    """
    # Get the current directory
    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    
    # If current_dir is called 'backend', we're in the backend directory
    # If parent dir contains a 'backend' directory, we're likely in the WebAgent root
    
    # Check if we're in the backend directory already
    if current_dir.name == 'backend' and str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
        
        # Also add the app directory to the path directly
        app_dir = current_dir / 'app'
        if app_dir.exists() and str(app_dir) not in sys.path:
            sys.path.insert(0, str(app_dir))
    
    # Otherwise, check if we need to add the backend directory to the path
    parent_dir = current_dir.parent
    backend_dir = parent_dir / 'backend'
    if backend_dir.exists() and str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))
        
        # Also add the app directory to the path directly
        app_dir = backend_dir / 'app'
        if app_dir.exists() and str(app_dir) not in sys.path:
            sys.path.insert(0, str(app_dir))
    
    # Always make sure current directory is in path
    if str(os.getcwd()) not in sys.path:
        sys.path.insert(0, str(os.getcwd()))
    
    # Create a backend.app fallback if needed
    try:
        import backend.app  # noqa: F401
    except ImportError:
        try:
            import app
            # If app exists but backend.app doesn't, create a fallback
            if importlib.util.find_spec("app") and not importlib.util.find_spec("backend.app"):
                sys.modules["backend.app"] = app
        except ImportError:
            pass


# Automatically setup the path when this module is imported
setup_path()


if __name__ == '__main__':
    setup_path()
    print("Python path updated. Current path:")
    for path in sys.path:
        print(f"  - {path}") 