"""
Diagnostics Utilities
====================

This module provides diagnostic functions for the WebAgent platform.
Use these utilities to check the health and status of various components.
"""
import os
import sys
import time
import json
import logging
import platform
from typing import Dict, Any, List
import importlib
import traceback
from datetime import datetime

# Import WebAgent modules
from app.core.config import settings
from app.core.metrics import get_metric_statistics

# Set up logging
logger = logging.getLogger(__name__)

class SystemDiagnostics:
    """System diagnostics utility class."""
    
    def __init__(self):
        """Initialize the diagnostics utility."""
        self.check_results = {}
        
    def run_all_checks(self) -> Dict[str, Any]:
        """
        Run all diagnostic checks and return results.
        
        Returns:
            Dictionary with all check results
        """
        logger.info("Running all diagnostic checks")
        
        # Run all checks
        self.check_system_info()
        self.check_dependencies()
        self.check_environment()
        self.check_agents()
        self.check_workflow()
        self.check_performance_metrics()
        
        # Add timestamp
        self.check_results["timestamp"] = datetime.now().isoformat()
        
        return self.check_results
    
    def check_system_info(self) -> Dict[str, Any]:
        """
        Check system information.
        
        Returns:
            Dictionary with system information
        """
        logger.info("Checking system information")
        
        system_info = {
            "os": platform.system(),
            "os_version": platform.version(),
            "python_version": sys.version,
            "webagent_version": settings.VERSION,
            "hostname": platform.node(),
            "cpu_count": os.cpu_count(),
            "current_time": datetime.now().isoformat(),
            "timezone": time.tzname
        }
        
        self.check_results["system_info"] = system_info
        return system_info
    
    def check_dependencies(self) -> Dict[str, Any]:
        """
        Check dependencies and their versions.
        
        Returns:
            Dictionary with dependency information
        """
        logger.info("Checking dependencies")
        
        # List of key dependencies to check
        dependencies = [
            "langchain", "langgraph", "fastapi",
            "pydantic", "uvicorn", "openai",
            "anthropic", "pytest", "numpy"
        ]
        
        # Check each dependency
        dependency_info = {}
        for dep in dependencies:
            try:
                module = importlib.import_module(dep)
                version = getattr(module, "__version__", "unknown")
                dependency_info[dep] = {"status": "installed", "version": version}
            except ImportError:
                dependency_info[dep] = {"status": "not_installed", "version": None}
        
        self.check_results["dependencies"] = dependency_info
        return dependency_info
    
    def check_environment(self) -> Dict[str, Any]:
        """
        Check environment configuration.
        
        Returns:
            Dictionary with environment information
        """
        logger.info("Checking environment configuration")
        
        # Get information from settings
        environment_info = {
            "webagent_env": settings.WEBAGENT_ENV,
            "debug_mode": settings.DEBUG_MODE,
            "api_base_url": settings.API_V1_STR,
            "log_level": settings.LOG_LEVEL,
            "request_size_limit": settings.REQUEST_SIZE_LIMIT,
            "llm_provider": settings.LLM_PROVIDER,
            "langsmith_enabled": settings.get("LANGSMITH_ENABLED", False),
        }
        
        # Check for required environment variables
        env_vars = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "LANGSMITH_API_KEY"]
        env_var_status = {}
        
        for var in env_vars:
            env_var_status[var] = {
                "set": var in os.environ,
                "value_length": len(os.environ.get(var, "")) if var in os.environ else 0
            }
        
        environment_info["environment_variables"] = env_var_status
        
        self.check_results["environment"] = environment_info
        return environment_info
    
    def check_agents(self) -> Dict[str, List[str]]:
        """
        Check available agents and their status.
        
        Returns:
            Dictionary with agent information
        """
        logger.info("Checking agents")
        
        # Import supervisor to get agent status
        try:
            from app.agents.supervisor import get_supervisor_agent
            supervisor = get_supervisor_agent()
            agent_status = supervisor.get_agent_status()
        except Exception as e:
            logger.error(f"Error checking agents: {str(e)}")
            agent_status = {"error": str(e)}
        
        self.check_results["agents"] = agent_status
        return agent_status
    
    def check_workflow(self) -> Dict[str, Any]:
        """
        Check workflow configuration.
        
        Returns:
            Dictionary with workflow information
        """
        logger.info("Checking workflow")
        
        try:
            from app.graph.workflows import get_agent_workflow
            workflow = get_agent_workflow()
            
            workflow_info = {
                "nodes": list(workflow.nodes),
                "compiled": hasattr(workflow, "_compiled") and workflow._compiled,
                "root_node": "__start__" in workflow.nodes
            }
            
        except Exception as e:
            logger.error(f"Error checking workflow: {str(e)}")
            workflow_info = {"error": str(e), "traceback": traceback.format_exc()}
        
        self.check_results["workflow"] = workflow_info
        return workflow_info
    
    def check_performance_metrics(self) -> Dict[str, Any]:
        """
        Check performance metrics.
        
        Returns:
            Dictionary with performance metrics
        """
        logger.info("Checking performance metrics")
        
        try:
            # Get metrics from the metrics module
            metrics = get_metric_statistics()
            
            # Add process memory usage
            import psutil
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            
            metrics["process"] = {
                "memory_rss": memory_info.rss,  # Resident Set Size
                "memory_vms": memory_info.vms,  # Virtual Memory Size
                "cpu_percent": process.cpu_percent(interval=0.1),
                "threads": len(process.threads())
            }
            
        except Exception as e:
            logger.error(f"Error checking metrics: {str(e)}")
            metrics = {"error": str(e)}
        
        self.check_results["metrics"] = metrics
        return metrics

# Singleton instance
_diagnostics_instance = None

def get_diagnostics() -> SystemDiagnostics:
    """
    Get the diagnostics instance (singleton pattern).
    
    Returns:
        SystemDiagnostics instance
    """
    global _diagnostics_instance
    if _diagnostics_instance is None:
        _diagnostics_instance = SystemDiagnostics()
    return _diagnostics_instance

def print_diagnostics_report(run_checks=True) -> None:
    """
    Print a diagnostic report to the console.
    
    Args:
        run_checks: Whether to run checks before printing
    """
    diagnostics = get_diagnostics()
    
    if run_checks:
        diagnostics.run_all_checks()
    
    # Print the report
    report = diagnostics.check_results
    
    print("\n========== WebAgent Diagnostics Report ==========")
    print(f"WebAgent v{settings.VERSION} | {report.get('timestamp', datetime.now().isoformat())}")
    print(f"Environment: {settings.WEBAGENT_ENV}")
    print("=" * 50)
    
    # System Info
    if "system_info" in report:
        info = report["system_info"]
        print("\n## System Information")
        print(f"OS: {info.get('os')} {info.get('os_version')}")
        print(f"Python: {info.get('python_version')}")
        print(f"CPU Count: {info.get('cpu_count')}")
    
    # Environment
    if "environment" in report:
        env = report["environment"]
        print("\n## Environment Configuration")
        print(f"Debug Mode: {env.get('debug_mode')}")
        print(f"LLM Provider: {env.get('llm_provider')}")
        
        # Check API keys
        env_vars = env.get("environment_variables", {})
        for var, status in env_vars.items():
            print(f"{var}: {'✓ Set' if status.get('set') else '✗ Not Set'}")
    
    # Agents
    if "agents" in report:
        agents = report["agents"]
        print("\n## Agent Status")
        for agent, status in agents.items():
            if isinstance(status, dict) and "status" in status:
                print(f"{agent}: {status.get('status')}")
            else:
                print(f"{agent}: {status}")
    
    # Workflow
    if "workflow" in report:
        workflow = report["workflow"]
        print("\n## Workflow")
        if "error" in workflow:
            print(f"Error: {workflow.get('error')}")
        else:
            print(f"Nodes: {', '.join(workflow.get('nodes', []))}")
            print(f"Compiled: {workflow.get('compiled', False)}")
    
    # Metrics
    if "metrics" in report and "process" in report["metrics"]:
        process = report["metrics"]["process"]
        print("\n## Performance Metrics")
        print(f"Memory (RSS): {process.get('memory_rss') / (1024*1024):.2f} MB")
        print(f"CPU Usage: {process.get('cpu_percent')}%")
        print(f"Threads: {process.get('threads')}")
    
    print("\n========== End Diagnostics Report ==========\n")

if __name__ == "__main__":
    # Run diagnostics if this module is executed directly
    print_diagnostics_report() 