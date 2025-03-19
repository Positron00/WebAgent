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
import importlib

# First, try to import from current package
try:
    from diagnostics import setup_paths
except ImportError:
    try:
        # Try as absolute import
        from backend.diagnostics import setup_paths
    except ImportError:
        # If all else fails, do inline path setup
        from pathlib import Path
        
        def setup_paths():
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
                    backend_dir = working_dir
                    while backend_dir.name != 'backend' and backend_dir != backend_dir.parent:
                        backend_dir = backend_dir.parent
                else:
                    backend_dir = working_dir / 'backend'
                    if not backend_dir.exists():
                        backend_dir = script_dir.parent
            
            # Add backend dir to path
            if str(backend_dir) not in sys.path:
                sys.path.insert(0, str(backend_dir))
                
            # Add project root
            project_root = backend_dir.parent
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))
                
            return backend_dir

# Set up paths
backend_dir = setup_paths()

def import_diagnostics_module():
    """Dynamically import the diagnostics core module."""
    modules_to_try = [
        'diagnostics.core',
        'backend.diagnostics.core',
        '.core',
    ]
    
    for module_name in modules_to_try:
        try:
            return importlib.import_module(module_name)
        except ImportError:
            continue
    
    print("Error: Could not import diagnostics module.")
    print("Make sure you're running this script from either the WebAgent root or backend directory.")
    sys.exit(1)

def main():
    """Run the system diagnostic checks and print a report."""
    # Try to use path_setup if available
    try:
        path_setup = importlib.import_module('path_setup')
        if hasattr(path_setup, 'setup_path'):
            path_setup.setup_path()
    except ImportError:
        pass
    
    # Import core diagnostics
    diagnostics_core = import_diagnostics_module()
    print_diagnostics_report = getattr(diagnostics_core, 'print_diagnostics_report')
    
    print("\nRunning WebAgent System Check...\n")
    
    # Ensure the app module can be found
    app_imported = False
    for module_name in ['app', 'backend.app']:
        try:
            importlib.import_module(module_name)
            app_imported = True
            break
        except ImportError:
            continue
    
    if not app_imported:
        print("Warning: Could not import app module. Some checks may fail.")
    
    print_diagnostics_report(run_checks=True)
    print("\nSystem check complete.\n")

if __name__ == "__main__":
    main() 