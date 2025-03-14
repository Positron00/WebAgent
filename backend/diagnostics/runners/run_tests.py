#!/usr/bin/env python
"""
WebAgent Test Runner
===================

This script runs the WebAgent test suite to verify the functionality of all components.
It provides various options for which tests to run and how to display the results.

Usage:
    python -m backend.diagnostics.runners.run_tests [options]

Options:
    --verbose, -v         Increase verbosity
    --test-dir=<dir>      Directory containing tests to run (default: backend/tests)
    --test-file=<file>    Specific test file to run
    --test-func=<func>    Specific test function to run
    --no-capture          Don't capture stdout/stderr
    --export-junit=<file> Export results to a JUnit XML file
"""
import os
import sys
import argparse
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Make sure pytest is available
try:
    import pytest
except ImportError:
    print("Error: pytest is not installed. Please install it with:")
    print("  pip install pytest pytest-xdist pytest-cov")
    sys.exit(1)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="WebAgent Test Runner")
    parser.add_argument("--verbose", "-v", action="count", default=0, 
                        help="Increase verbosity (can be used multiple times)")
    parser.add_argument("--test-dir", type=str, default="backend/tests",
                        help="Directory containing tests to run")
    parser.add_argument("--test-file", type=str, help="Specific test file to run")
    parser.add_argument("--test-func", type=str, help="Specific test function to run")
    parser.add_argument("--no-capture", action="store_true", help="Don't capture stdout/stderr")
    parser.add_argument("--export-junit", type=str, help="Export results to a JUnit XML file")
    parser.add_argument("--cov", action="store_true", help="Enable coverage reporting")
    parser.add_argument("--cov-report", type=str, choices=["term", "html", "xml"], default="term",
                        help="Coverage report format")
    
    args = parser.parse_args()
    
    # Set up test environment
    os.environ["TESTING"] = "True"
    os.environ.setdefault("WEBAGENT_ENV", "test")
    
    # Build pytest arguments
    pytest_args = []
    
    # Set verbosity
    if args.verbose == 0:
        pytest_args.append("-q")
    elif args.verbose == 1:
        pytest_args.append("-v")
    elif args.verbose >= 2:
        pytest_args.append("-vv")
    
    # Don't capture stdout/stderr if requested
    if args.no_capture:
        pytest_args.append("-s")
    
    # Export JUnit XML if requested
    if args.export_junit:
        pytest_args.extend(["--junitxml", args.export_junit])
    
    # Enable coverage if requested
    if args.cov:
        pytest_args.extend(["--cov=backend", f"--cov-report={args.cov_report}"])
    
    # Determine test target
    if args.test_file and args.test_func:
        target = f"{args.test_file}::{args.test_func}"
    elif args.test_file:
        target = args.test_file
    else:
        target = args.test_dir
    
    pytest_args.append(target)
    
    print(f"Running tests with arguments: {' '.join(pytest_args)}")
    result = pytest.main(pytest_args)
    
    # Report status
    if result == 0:
        print("\n✅ All tests passed!")
    else:
        print(f"\n❌ Tests failed with exit code {result}")
    
    return result

if __name__ == "__main__":
    sys.exit(main()) 