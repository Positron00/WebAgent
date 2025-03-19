#!/usr/bin/env python
"""
Test Runner for WebAgent

This script runs the WebAgent test suite, which is now located in backend/tests.
The test suite includes unit tests, diagnostic tests, and integration tests.

Run specific tests with:
python -m backend.diagnostics.runners.run_tests --test-file=test_langgraph.py
OR
cd backend && python diagnostics/runners/run_tests.py --test-file=test_langgraph.py
"""
import os
import sys
import argparse
import subprocess
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("test_runner")

# Determine if we're in the backend directory or the project root
current_dir = Path(os.path.abspath(os.path.dirname(__file__)))
backend_dir = current_dir.parent.parent

# Try to import path_setup from the backend directory
try:
    # First, add the backend directory to the path if it's not already there
    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))
    
    # Try to import the path_setup module
    from path_setup import setup_path
    setup_path()
    path_setup_imported = True
except ImportError:
    path_setup_imported = False
    # Fallback method to add project directories to path
    project_root = Path(__file__).resolve().parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # Make sure the current directory is in the path
    if str(os.getcwd()) not in sys.path:
        sys.path.insert(0, str(os.getcwd()))

def run_tests(test_file=None, test_dir=None, verbose=False):
    """
    Run the WebAgent test suite.
    
    Args:
        test_file: Optional specific test file to run
        test_dir: Optional specific test directory to run tests from
        verbose: Whether to display verbose output
    
    Returns:
        True if all tests pass, False otherwise
    """
    # Determine the correct base directory regardless of where we're running from
    # First try to find the backend directory from the current file location
    file_base_dir = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    # Check if we're already in the backend directory
    if file_base_dir.name == 'backend':
        base_dir = file_base_dir
    else:
        # Look for the backend directory in the current working directory
        cwd = Path(os.getcwd())
        if cwd.name == 'backend':
            base_dir = cwd
        elif (cwd / 'backend').exists():
            base_dir = cwd / 'backend'
        else:
            # Fallback to using the file location
            base_dir = file_base_dir
    
    tests_dir = base_dir / "tests"
    
    if not tests_dir.exists():
        logger.error(f"Tests directory not found: {tests_dir}")
        logger.error("Make sure you are running this script from the WebAgent root or backend directory")
        return False
    
    logger.info(f"Running tests from {tests_dir}")
    
    # Build the pytest command
    command = ["python", "-m", "pytest"]
    
    if verbose:
        command.append("-v")
    
    # Add test file or directory if specified
    if test_file:
        if test_dir:
            # If both file and directory are specified, look in that directory
            command.append(str(tests_dir / test_dir / test_file))
        else:
            # Otherwise look in the main tests directory
            command.append(str(tests_dir / test_file))
    elif test_dir:
        # If only directory is specified, run all tests in that directory
        command.append(str(tests_dir / test_dir))
    else:
        # Run all tests
        command.append(str(tests_dir))
    
    # Try to add code coverage if available
    try:
        import pytest_cov
        command.extend(["--cov=backend", "--cov-report=term"])
        logger.info("Running with coverage reporting enabled")
    except ImportError:
        logger.warning("pytest-cov not installed - running without coverage reporting")
    
    # Run the tests
    logger.info(f"Running command: {' '.join(command)}")
    
    # Determine the correct working directory for subprocess.run
    if base_dir.parent.name == 'WebAgent' or (base_dir.parent / 'backend').exists():
        # If the parent is the WebAgent root, use that as working directory
        cwd = str(base_dir.parent)
    else:
        # Otherwise use the backend directory
        cwd = str(base_dir)
    
    result = subprocess.run(command, cwd=cwd)
    
    # Check if tests passed
    if result.returncode == 0:
        logger.info("All tests passed!")
        return True
    else:
        logger.error(f"Tests failed with return code {result.returncode}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run WebAgent tests")
    parser.add_argument("--test-file", help="Specific test file to run")
    parser.add_argument("--test-dir", help="Specific test directory to run tests from")
    parser.add_argument("--verbose", action="store_true", help="Display verbose output")
    
    # Only parse known args to allow calling from other scripts
    args, _ = parser.parse_known_args()
    
    success = run_tests(args.test_file, args.test_dir, args.verbose)
    if not success:
        sys.exit(1) 