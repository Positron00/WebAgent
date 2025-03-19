#!/usr/bin/env python
"""
Run WebAgent Diagnostics

This script runs the diagnostics for the WebAgent platform, providing information on:
1. System information
2. Python environment 
3. Installed dependencies
4. Environment variables
5. Model services status

Usage:
    python -m backend.diagnostics.runners.run_diagnostics [options]
    OR
    cd backend && python diagnostics/runners/run_diagnostics.py [options]

Options:
    --verbose, -v        Increase verbosity (display DEBUG level logs)
    --quiet, -q          Decrease verbosity (display only warnings and errors)
    --no-color           Disable colored output
    --output-file=FILE   Write diagnostics to a file
    --output-format=FMT  Output format (text, json, markdown)
    --run-tests          Also run diagnostic tests
"""
import os
import sys
import time
import argparse
import logging
import datetime
from pathlib import Path

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

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("webagent_diagnostics")

# Try to use path_setup if available
try:
    from path_setup import setup_path
    setup_path()
except ImportError:
    pass

# Try to import the core diagnostics module
try:
    try:
        # Try absolute import first
        from backend.diagnostics.core import print_diagnostics_report, get_diagnostics
    except ImportError:
        # Try relative import if absolute fails
        from diagnostics.core import print_diagnostics_report, get_diagnostics
except ImportError as e:
    logger.error(f"Error importing diagnostics module: {e}")
    logger.error("Make sure you're running this script from either the WebAgent root or the backend directory")
    sys.exit(1)

# Load environment variables from .env.local file if it exists
env_file = backend_dir / '.env.local'
if env_file.exists():
    try:
        print(f"Loading environment variables from {env_file}")
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip().strip('"\'')
        print("Environment variables loaded successfully")
    except Exception as e:
        print(f"Error loading environment variables from {env_file}: {e}")

# Try to load configuration to ensure environment variables are set
try:
    # Try with absolute imports first
    try:
        from backend.app.core.loadEnvYAML import load_config
        load_config()  # This will load and export environment variables
    except ImportError:
        # Try with relative imports
        try:
            from app.core.loadEnvYAML import load_config
            load_config()  # This will load and export environment variables
        except ImportError:
            # If neither works, we'll continue without configuration
            print("Warning: Could not load configuration. Some environment variables may be missing.")
except Exception as e:
    print(f"Warning: Error loading configuration: {str(e)}")

# Try to import the logger - handle case if not available
try:
    try:
        from app.core.logger import setup_logger
    except ImportError:
        try:
            from backend.app.core.logger import setup_logger
        except ImportError:
            # Fallback logger setup if the app module is not available
            import logging
            def setup_logger(name):
                logger = logging.getLogger(name)
                logger.setLevel(logging.INFO)
                if not logger.handlers:
                    handler = logging.StreamHandler()
                    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
                    logger.addHandler(handler)
                return logger
except ImportError:
    # If all else fails, define a simple logger
    import logging
    def setup_logger(name):
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            logger.addHandler(handler)
        return logger

# Set up logging
logger = setup_logger("webagent.diagnostics_runner")

def main():
    """
    Main entry point for the diagnostics script.
    
    Returns:
        0 for success, 1 for failure
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="WebAgent Diagnostics Runner")
    parser.add_argument("--check-network", action="store_true", help="Check network connectivity")
    parser.add_argument("--check-llm", action="store_true", help="Check language model availability")
    parser.add_argument("--export-json", type=str, help="Export results to a JSON file")
    parser.add_argument("--format", choices=["text", "json", "markdown"], default="text", 
                        help="Output format (default: text)")
    
    # Only parse known args to allow calling from other scripts with additional args
    args, _ = parser.parse_known_args()
    
    # Set up logging
    logger.info("Starting WebAgent diagnostics")
    
    # Get diagnostics instance
    diagnostics = get_diagnostics()
    
    # Selectively disable certain checks
    if not args.check_network:
        diagnostics.check_network_connectivity = lambda: {"status": "skipped"}
        
    if not args.check_llm:
        diagnostics.check_language_model_availability = lambda: {"status": "skipped"}
    
    # Run all checks
    results = diagnostics.run_all_checks()
    
    # Print report based on format
    if args.format == "text":
        print_diagnostics_report(run_checks=False)  # We already ran checks
    elif args.format == "json":
        print(json.dumps(results, indent=2))
    elif args.format == "markdown":
        print_markdown_report(results)
    
    # Export to JSON if requested
    if args.export_json:
        with open(args.export_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults exported to {args.export_json}")
    
    logger.info("Diagnostics completed")
    
    # Return success if no critical errors
    if "workflow" in results and "error" in results["workflow"]:
        return 1
    if "agents" in results and "error" in results["agents"]:
        return 1
    return 0

def print_markdown_report(results):
    """Print a markdown-formatted report."""
    print("# WebAgent Diagnostics Report\n")
    print(f"**Version:** {results.get('system_info', {}).get('webagent_version', 'unknown')}")
    print(f"**Timestamp:** {results.get('timestamp', 'unknown')}")
    print(f"**Environment:** {results.get('environment', {}).get('webagent_env', 'unknown')}\n")
    
    # System Info
    print("## System Information")
    if "system_info" in results:
        info = results["system_info"]
        print(f"- **OS:** {info.get('os')} {info.get('os_version')}")
        print(f"- **Python:** {info.get('python_version').split()[0]}")
        print(f"- **CPU Count:** {info.get('cpu_count')}")
        print(f"- **Platform:** {info.get('platform')}")
    
    # Environment
    if "environment" in results:
        env = results["environment"]
        print("\n## Environment Configuration")
        print(f"- **Debug Mode:** {env.get('debug_mode')}")
        print(f"- **LLM Provider:** {env.get('llm_provider')}")
        
        # Check API keys
        env_vars = env.get("environment_variables", {})
        print("\n### API Keys")
        for var, status in env_vars.items():
            print(f"- **{var}:** {'✓ Set' if status.get('set') else '✗ Not Set'}")
    
    # Agents
    if "agents" in results:
        agents = results["agents"]
        print("\n## Agent Status")
        for agent, status in agents.items():
            if isinstance(status, dict) and "status" in status:
                avg_time = status.get("average_execution_time", "N/A")
                avg_time = f"{avg_time:.2f}s" if isinstance(avg_time, (int, float)) else avg_time
                print(f"- **{agent}:** {status.get('status')} (Avg time: {avg_time})")
            else:
                print(f"- **{agent}:** {status}")
    
    # Workflow
    if "workflow" in results:
        workflow = results["workflow"]
        print("\n## Workflow")
        if "error" in workflow:
            print(f"- **Error:** {workflow.get('error')}")
        else:
            print(f"- **Nodes:** {', '.join(workflow.get('nodes', []))}")
            print(f"- **Compiled:** {workflow.get('compiled', False)}")
            print(f"- **Node Count:** {workflow.get('node_count', 0)}")
            
            missing = workflow.get("missing_critical_nodes", [])
            if missing:
                print(f"- **Missing Critical Nodes:** {', '.join(missing)}")
    
    # Network Connectivity
    if "network_connectivity" in results:
        connectivity = results["network_connectivity"]
        print("\n## Network Connectivity")
        for service, status in connectivity.items():
            if status.get("status") == "available":
                latency = status.get("latency_ms", 0)
                print(f"- **{service}:** ✓ Available (Latency: {latency:.1f}ms)")
            else:
                print(f"- **{service}:** ✗ Unavailable ({status.get('error', 'Unknown error')})")
    
    # LLM Availability
    if "llm_availability" in results:
        llm = results["llm_availability"]
        print("\n## Language Model Status")
        if "service_status" in llm:
            print(f"- **Provider:** {llm['service_status'].get('provider', 'Unknown')}")
            print(f"- **Default Model:** {llm['service_status'].get('default_model', 'Unknown')}")
        if "api_check" in llm:
            print(f"- **API Check:** {llm['api_check'].get('status', 'Unknown')}")
    
    # Execution Times
    if "execution_time" in results:
        times = results["execution_time"]
        print("\n## Diagnostic Execution Times")
        print(f"- **Total:** {times.get('total_seconds', 0):.2f} seconds")

if __name__ == "__main__":
    sys.exit(main()) 