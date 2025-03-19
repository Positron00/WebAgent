#!/usr/bin/env python
"""
Run All Diagnostics and Tests for WebAgent

This script runs all the diagnostics and tests for the WebAgent platform:
1. System diagnostics check
2. Test suite execution

The test suite is now located in backend/tests/ and includes:
- Unit tests in backend/tests/unit/
- Diagnostic tests in backend/tests/diagnostics/
- Integration tests in backend/tests/

Usage:
    python -m backend.diagnostics.runners.run_all [options]
    OR
    cd backend && python diagnostics/runners/run_all.py [options]

Options:
    --skip-tests           Skip running tests
    --skip-diagnostics     Skip running diagnostics
    --verbose, -v          Increase verbosity
    --test-dir             Specific test directory to run
    --test-file            Specific test file to run
"""
import os
import sys
import time
import argparse
import logging
import subprocess
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("webagent_diagnostics")

# First, try to import the setup_paths function from the diagnostics package
try:
    # Try importing from the current package
    from diagnostics import setup_paths
except ImportError:
    try:
        # Try as absolute import
        from backend.diagnostics import setup_paths
    except ImportError:
        # If all imports fail, define the function inline
        def setup_paths():
            script_dir = Path(os.path.abspath(os.path.dirname(__file__)))
            # Navigate up two levels to find backend directory (runners -> diagnostics -> backend)
            backend_dir = script_dir.parent.parent
            project_root = backend_dir.parent if backend_dir.name == 'backend' else backend_dir
            
            # Add the necessary paths to sys.path
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))
            if str(backend_dir) not in sys.path:
                sys.path.insert(0, str(backend_dir))
                
            return backend_dir

# Set up paths
backend_dir = setup_paths()
project_root = backend_dir.parent if backend_dir.name == 'backend' else backend_dir

# Load environment variables from .env.local file if it exists
env_file = project_root / '.env.local'
if env_file.exists():
    try:
        logger.debug(f"Loading environment variables from {env_file}")
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip().strip('"\'')
        logger.debug("Environment variables loaded successfully")
    except Exception as e:
        logger.warning(f"Error loading environment variables from {env_file}: {e}")

# Try to import path_setup from the backend directory
try:
    # Try to import the path_setup module
    from path_setup import setup_path
    setup_path()
    path_setup_imported = True
except ImportError:
    path_setup_imported = False

# Try both import styles - absolute and relative
try:
    try:
        # Try absolute import
        from backend.diagnostics.runners.run_diagnostics import main as run_diagnostics
        from backend.diagnostics.runners.run_tests import run_tests
    except ImportError:
        # Try relative import
        from diagnostics.runners.run_diagnostics import main as run_diagnostics
        from diagnostics.runners.run_tests import run_tests
except ImportError as e:
    print(f"Error importing diagnostic runners: {e}")
    print("Make sure you are running this script from the WebAgent root or backend directory")
    sys.exit(1)

def main():
    """Run all diagnostics and tests for WebAgent."""
    parser = argparse.ArgumentParser(description="Run all WebAgent diagnostics and tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Display verbose output")
    parser.add_argument("--skip-diagnostics", action="store_true", help="Skip diagnostic checks")
    parser.add_argument("--skip-tests", action="store_true", help="Skip tests")
    parser.add_argument("--test-dir", type=str, help="Specific test directory to run (relative to tests/)")
    parser.add_argument("--test-file", type=str, help="Specific test file to run")
    parser.add_argument("--tests-only", "-t", action="store_true", help="Only run tests (alias for --skip-diagnostics)")
    
    args = parser.parse_args()
    
    # If --tests-only is provided, set skip_diagnostics to True
    if args.tests_only:
        args.skip_diagnostics = True
    
    overall_success = True
    start_time = time.time()
    
    logger.info("===== WebAgent Comprehensive Diagnostics and Tests =====")
    
    # Run diagnostics
    if not args.skip_diagnostics:
        logger.info("\n===== Running System Diagnostics =====")
        try:
            diagnostics_success = run_diagnostics() == 0  # Convert return code to boolean success
            if not diagnostics_success:
                logger.error("❌ Diagnostics failed!")
                overall_success = False
            else:
                logger.info("✅ Diagnostics passed!")
        except Exception as e:
            logger.error(f"❌ Diagnostics error: {str(e)}")
            overall_success = False
    else:
        logger.info("Skipping diagnostics as requested")
    
    # Run tests
    if not args.skip_tests:
        logger.info("\n===== Running Test Suite =====")
        
        try:
            # Run all tests or specified directory/file
            test_success = run_tests(test_file=args.test_file, test_dir=args.test_dir, verbose=args.verbose)
            
            if not test_success:
                logger.error("❌ Tests failed!")
                overall_success = False
            else:
                logger.info("✅ Tests passed!")
        except Exception as e:
            logger.error(f"❌ Tests error: {str(e)}")
            overall_success = False
    else:
        logger.info("Skipping tests as requested")
    
    # Report overall status
    elapsed_time = time.time() - start_time
    logger.info(f"\n===== Diagnostics and Tests completed in {elapsed_time:.2f} seconds =====")
    
    if overall_success:
        logger.info("✅ All checks passed successfully!")
        return 0
    else:
        logger.error("❌ Some checks failed. See above for details.")
        return 1

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