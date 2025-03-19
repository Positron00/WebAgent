#!/usr/bin/env python
"""
WebAgent System Check
====================

This is a convenience script to run the diagnostics suite.
It forwards to the full diagnostics runner.

Can be run from:
- WebAgent root: python -m backend.diagnostics.system_check
- Backend directory: python diagnostics/system_check.py
"""
import sys
import os
from pathlib import Path

# Determine if we're in the backend directory or the project root
current_dir = Path(os.path.abspath(os.path.dirname(__file__)))
backend_dir = current_dir.parent

# Try to import path_setup from the backend directory
try:
    # First, add the backend directory to the path if it's not already there
    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))
    
    # Try to import the path_setup module
    from path_setup import setup_path
    setup_path()
except ImportError:
    # Add project root to path for imports
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

# Try both import styles
try:
    try:
        # Try absolute import first
        from backend.diagnostics.core import print_diagnostics_report
    except ImportError:
        # Try relative import if absolute fails
        from diagnostics.core import print_diagnostics_report
except ImportError:
    print("Error: Could not import diagnostics module.")
    print("Make sure you're running this script from either the WebAgent root or backend directory.")
    sys.exit(1)

def main():
    """Run the system diagnostic checks and print a report."""
    print("\nRunning WebAgent System Check...\n")
    print_diagnostics_report(run_checks=True)
    print("\nSystem check complete.\n")

if __name__ == "__main__":
    main() 