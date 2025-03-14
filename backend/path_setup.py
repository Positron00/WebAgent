"""
Path Setup Utility
=================

This module helps set up the Python path correctly for importing modules within this project.
It allows code to be run either from the backend directory or from the WebAgent root directory.
"""

import os
import sys
from pathlib import Path


def setup_path():
    """
    Add the appropriate directories to the Python path for imports to work correctly.
    
    This function:
    1. If running from WebAgent root directory, adds backend/ to the path
    2. Makes sure the current directory is in the path
    """
    # Get the current directory
    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    
    # If current_dir is called 'backend', we're in the backend directory
    # If parent dir contains a 'backend' directory, we're likely in the WebAgent root
    
    # Check if we're in the backend directory already
    if current_dir.name == 'backend' and current_dir not in sys.path:
        sys.path.insert(0, str(current_dir))
    
    # Otherwise, check if we need to add the backend directory to the path
    parent_dir = current_dir.parent
    backend_dir = parent_dir / 'backend'
    if backend_dir.exists() and backend_dir not in sys.path:
        sys.path.insert(0, str(backend_dir))
    
    # Always make sure current directory is in path
    if str(os.getcwd()) not in sys.path:
        sys.path.insert(0, str(os.getcwd()))


# Automatically setup the path when this module is imported
setup_path()


if __name__ == '__main__':
    setup_path()
    print("Python path updated. Current path:")
    for path in sys.path:
        print(f"  - {path}") 