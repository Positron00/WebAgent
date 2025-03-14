#!/usr/bin/env python
"""
WebAgent Diagnostics Runner
===========================

This script runs the WebAgent diagnostics suite to check the health of all components.
It provides various options for which checks to run and how to display the results.

Usage:
    python -m backend.diagnostics.runners.run_diagnostics [options]

Options:
    --check-network       Check network connectivity to essential services
    --check-llm           Check language model availability (may make API calls)
    --export-json=<file>  Export results to a JSON file
    --format=<format>     Output format: text, json, markdown (default: text)
"""
import os
import sys
import json
import argparse
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import diagnostics
from backend.diagnostics.core import get_diagnostics, print_diagnostics_report
from backend.app.core.logger import setup_logger

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="WebAgent Diagnostics Runner")
    parser.add_argument("--check-network", action="store_true", help="Check network connectivity")
    parser.add_argument("--check-llm", action="store_true", help="Check language model availability")
    parser.add_argument("--export-json", type=str, help="Export results to a JSON file")
    parser.add_argument("--format", choices=["text", "json", "markdown"], default="text", 
                        help="Output format (default: text)")
    
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logger("diagnostics_runner")
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