#!/usr/bin/env python
"""
WebAgent Test & Diagnostics Runner
=================================

This script runs all tests and diagnostics for the WebAgent platform.
It provides a comprehensive health check and test report.

Usage:
    python -m backend.diagnostics.runners.run_all [options]

Options:
    --skip-tests           Skip running tests
    --skip-diagnostics     Skip running diagnostics
    --check-network        Check network connectivity
    --check-llm            Check LLM availability (may make API calls)
    --export-json=<file>   Export results to a JSON file
    --export-junit=<file>  Export test results to a JUnit XML file
    --verbose, -v          Increase verbosity
"""
import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import test and diagnostics runners
from backend.diagnostics.core import get_diagnostics, print_diagnostics_report
from backend.app.core.logger import setup_logger

def main():
    """Run all tests and diagnostics."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="WebAgent Test & Diagnostics Runner")
    parser.add_argument("--skip-tests", action="store_true", help="Skip running tests")
    parser.add_argument("--skip-diagnostics", action="store_true", help="Skip running diagnostics")
    parser.add_argument("--check-network", action="store_true", help="Check network connectivity")
    parser.add_argument("--check-llm", action="store_true", help="Check LLM availability")
    parser.add_argument("--export-json", type=str, help="Export results to a JSON file")
    parser.add_argument("--export-junit", type=str, help="Export test results to a JUnit XML file")
    parser.add_argument("--verbose", "-v", action="count", default=0, 
                        help="Increase verbosity (can be used multiple times)")
    parser.add_argument("--test-dir", type=str, default="backend/diagnostics/tests",
                        help="Directory containing tests to run")
    
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logger("diagnostics_runner")
    logger.info("Starting WebAgent diagnostics and tests")
    
    # Set up the environment
    os.environ.setdefault("WEBAGENT_ENV", "test")
    os.environ["TESTING"] = "True"
    
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tests": None,
        "diagnostics": None
    }
    
    # Run tests if not skipped
    if not args.skip_tests:
        logger.info("Running tests...")
        test_result = run_tests(args)
        results["tests"] = {
            "passed": test_result == 0,
            "exit_code": test_result
        }
    else:
        logger.info("Tests skipped.")
    
    # Run diagnostics if not skipped
    if not args.skip_diagnostics:
        logger.info("Running diagnostics...")
        diag_results = run_diagnostics(args)
        results["diagnostics"] = diag_results
    else:
        logger.info("Diagnostics skipped.")
    
    # Export results if requested
    if args.export_json:
        with open(args.export_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults exported to {args.export_json}")
    
    # Return success if both tests and diagnostics passed (or were skipped)
    tests_ok = args.skip_tests or results["tests"]["passed"]
    diag_ok = args.skip_diagnostics or not has_critical_errors(results.get("diagnostics", {}))
    
    return 0 if tests_ok and diag_ok else 1

def run_tests(args):
    """Run tests using pytest."""
    try:
        import pytest
    except ImportError:
        print("Error: pytest is not installed. Please install it with:")
        print("  pip install pytest pytest-xdist pytest-cov")
        return 1
    
    # Build pytest arguments
    pytest_args = []
    
    # Set verbosity
    if args.verbose == 0:
        pytest_args.append("-q")
    elif args.verbose == 1:
        pytest_args.append("-v")
    elif args.verbose >= 2:
        pytest_args.append("-vv")
    
    # Export JUnit XML if requested
    if args.export_junit:
        pytest_args.extend(["--junitxml", args.export_junit])
    
    # Set test directory
    pytest_args.append(args.test_dir)
    
    print(f"Running tests with arguments: {' '.join(pytest_args)}")
    result = pytest.main(pytest_args)
    
    # Report status
    if result == 0:
        print("\n✅ All tests passed!")
    else:
        print(f"\n❌ Tests failed with exit code {result}")
    
    return result

def run_diagnostics(args) -> Dict[str, Any]:
    """Run diagnostics and return results."""
    # Get diagnostics instance
    diagnostics = get_diagnostics()
    
    # Selectively disable certain checks
    if not args.check_network:
        diagnostics.check_network_connectivity = lambda: {"status": "skipped"}
        
    if not args.check_llm:
        diagnostics.check_language_model_availability = lambda: {"status": "skipped"}
    
    # Run all checks
    results = diagnostics.run_all_checks()
    
    # Print report
    print_diagnostics_report(run_checks=False)  # We already ran checks
    
    return results

def has_critical_errors(diagnostics_results) -> bool:
    """Check if diagnostics results contain critical errors."""
    if not diagnostics_results:
        return False
    
    # Check for workflow errors
    if "workflow" in diagnostics_results and "error" in diagnostics_results["workflow"]:
        return True
    
    # Check for agent errors
    if "agents" in diagnostics_results and "error" in diagnostics_results["agents"]:
        return True
    
    return False

if __name__ == "__main__":
    sys.exit(main()) 