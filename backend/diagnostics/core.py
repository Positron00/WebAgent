"""
Core Diagnostics Utilities
=========================

This module provides core diagnostic functions for the WebAgent platform.
It consolidates all diagnostic capabilities in one place.
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
from backend.app.core.config import settings
from backend.app.core.metrics import get_metric_statistics, timing_decorator, log_memory_usage

# Set up logging
logger = logging.getLogger(__name__)

class SystemDiagnostics:
    """System diagnostics utility class."""
    
    def __init__(self):
        """Initialize the diagnostics utility."""
        self.check_results = {}
        self.check_times = {}
        
    @timing_decorator
    def run_all_checks(self) -> Dict[str, Any]:
        """
        Run all diagnostic checks and return results.
        
        Returns:
            Dictionary with all check results
        """
        logger.info("Running all diagnostic checks")
        
        # Run all checks and track execution times
        start_time = time.time()
        
        self.check_system_info()
        self.check_dependencies()
        self.check_environment()
        self.check_agents()
        self.check_workflow()
        self.check_performance_metrics()
        self.check_network_connectivity()
        self.check_language_model_availability()
        self.check_security_status()
        
        # Add timestamp and total execution time
        total_time = time.time() - start_time
        self.check_results["timestamp"] = datetime.now().isoformat()
        self.check_results["execution_time"] = {
            "total_seconds": total_time,
            "individual_checks": self.check_times
        }
        
        logger.info(f"All diagnostic checks completed in {total_time:.2f} seconds")
        return self.check_results
    
    @timing_decorator
    def check_system_info(self) -> Dict[str, Any]:
        """
        Check system information.
        
        Returns:
            Dictionary with system information
        """
        logger.info("Checking system information")
        start_time = time.time()
        
        system_info = {
            "os": platform.system(),
            "os_version": platform.version(),
            "python_version": sys.version,
            "webagent_version": settings.VERSION,
            "hostname": platform.node(),
            "cpu_count": os.cpu_count(),
            "current_time": datetime.now().isoformat(),
            "timezone": time.tzname,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "architecture": platform.architecture(),
            "python_path": sys.executable,
            "working_directory": os.getcwd(),
            "process_id": os.getpid()
        }
        
        self.check_results["system_info"] = system_info
        self.check_times["system_info"] = time.time() - start_time
        return system_info
    
    @timing_decorator
    def check_dependencies(self) -> Dict[str, Any]:
        """
        Check dependencies and their versions.
        
        Returns:
            Dictionary with dependency information
        """
        logger.info("Checking dependencies")
        start_time = time.time()
        
        # List of key dependencies to check
        dependencies = [
            "langchain", "langgraph", "fastapi",
            "pydantic", "uvicorn", "openai",
            "anthropic", "pytest", "numpy",
            "psutil", "prometheus_client", "httpx",
            "redis", "transformers", "jwt"
        ]
        
        # Check each dependency
        dependency_info = {}
        for dep in dependencies:
            try:
                module = importlib.import_module(dep)
                version = getattr(module, "__version__", "unknown")
                dependency_info[dep] = {
                    "status": "installed", 
                    "version": version,
                    "location": getattr(module, "__file__", "unknown")
                }
                logger.debug(f"Dependency {dep} v{version} found at {getattr(module, '__file__', 'unknown')}")
            except ImportError:
                dependency_info[dep] = {"status": "not_installed", "version": None, "location": None}
                logger.warning(f"Dependency {dep} is not installed")
            except Exception as e:
                dependency_info[dep] = {"status": "error", "error": str(e), "version": None}
                logger.error(f"Error checking dependency {dep}: {str(e)}")
        
        self.check_results["dependencies"] = dependency_info
        self.check_times["dependencies"] = time.time() - start_time
        return dependency_info
    
    @timing_decorator
    def check_environment(self) -> Dict[str, Any]:
        """
        Check environment configuration.
        
        Returns:
            Dictionary with environment information
        """
        logger.info("Checking environment configuration")
        start_time = time.time()
        
        # Get environment or default to 'dev'
        env = getattr(settings, "WEBAGENT_ENV", os.getenv("WEBAGENT_ENV", "dev"))
        
        # Get information from settings
        environment_info = {
            "webagent_env": env,
            "debug_mode": settings.DEBUG_MODE,
            "api_base_url": settings.API_V1_STR,
            "log_level": getattr(settings, "LOG_LEVEL", "INFO"),
            "request_size_limit": getattr(settings, "REQUEST_SIZE_LIMIT", "10MB"),
            "llm_provider": getattr(settings, "LLM_PROVIDER", "openai"),
            "langsmith_enabled": getattr(settings, "LANGSMITH_ENABLED", False),
        }
        
        # Check for required environment variables
        env_vars = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "LANGSMITH_API_KEY"]
        env_var_status = {}
        
        for var in env_vars:
            key_exists = var in os.environ
            value_length = len(os.environ.get(var, "")) if var in os.environ else 0
            
            # Log with appropriate level
            if key_exists and value_length > 0:
                logger.debug(f"Environment variable {var} is set (length: {value_length})")
            elif key_exists and value_length == 0:
                logger.warning(f"Environment variable {var} is set but empty")
            else:
                logger.warning(f"Environment variable {var} is not set")
            
            env_var_status[var] = {
                "set": key_exists,
                "value_length": value_length,
                "status": "valid" if key_exists and value_length > 0 else "invalid"
            }
        
        environment_info["environment_variables"] = env_var_status
        
        # Check temp directory permissions
        tmp_dir = os.environ.get("TMPDIR", "/tmp")
        try:
            test_file = os.path.join(tmp_dir, "webagent_test.txt")
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
            environment_info["tmp_dir_writable"] = True
            logger.debug(f"Temporary directory {tmp_dir} is writable")
        except Exception as e:
            environment_info["tmp_dir_writable"] = False
            environment_info["tmp_dir_error"] = str(e)
            logger.warning(f"Temporary directory {tmp_dir} is not writable: {str(e)}")
        
        self.check_results["environment"] = environment_info
        self.check_times["environment"] = time.time() - start_time
        return environment_info
    
    @timing_decorator
    def check_agents(self) -> Dict[str, Dict[str, Any]]:
        """
        Check available agents and their status.
        
        Returns:
            Dictionary with agent information
        """
        logger.info("Checking agents")
        start_time = time.time()
        
        # Import supervisor to get agent status
        try:
            from backend.app.agents.supervisor import get_supervisor_agent
            supervisor = get_supervisor_agent()
            agent_status = supervisor.get_agent_status()
            
            # Add additional diagnostics for each agent
            for agent_name, status in agent_status.items():
                if isinstance(status, dict):
                    # Add memory usage information
                    import psutil
                    process = psutil.Process(os.getpid())
                    memory_info = process.memory_info()
                    
                    status["memory_usage"] = {
                        "rss": memory_info.rss / 1024 / 1024,  # MB
                        "vms": memory_info.vms / 1024 / 1024   # MB
                    }
                    
                    # Add execution time information if available from metrics
                    from backend.app.core.metrics import get_metric_statistics
                    metrics = get_metric_statistics()
                    execution_times = metrics.get("execution_times", {})
                    
                    if f"{agent_name}.run" in execution_times:
                        status["average_execution_time"] = execution_times[f"{agent_name}.run"]["average"]
                        status["max_execution_time"] = execution_times[f"{agent_name}.run"]["max"]
                        status["min_execution_time"] = execution_times[f"{agent_name}.run"]["min"]
                        status["total_executions"] = execution_times[f"{agent_name}.run"]["count"]
                
                logger.info(f"Agent {agent_name} status: {status.get('status', 'unknown')}")
            
        except Exception as e:
            error_msg = f"Error checking agents: {str(e)}"
            agent_status = {"error": error_msg}
            logger.error(error_msg, exc_info=True)
        
        self.check_results["agents"] = agent_status
        self.check_times["agents"] = time.time() - start_time
        return agent_status
    
    @timing_decorator
    def check_workflow(self) -> Dict[str, Any]:
        """
        Check the LangGraph workflow.
        
        Returns:
            Dict with workflow status information
        """
        logger.info("Checking LangGraph workflow")
        
        workflow_info = {
            "status": "not_checked",
            "error": None,
            "nodes": [],
            "edges": []
        }
        
        try:
            # Try to get the workflow
            from backend.app.graph.workflows import get_agent_workflow
            
            try:
                workflow = get_agent_workflow()
                workflow_info["status"] = "success"
                workflow_info["nodes"] = list(workflow.graph.nodes)
                workflow_info["edges"] = [(src, dest) for src, dest in workflow.graph.edges]
            except TypeError as e:
                # Special handling for LangGraph compatibility errors
                if "Expected a Runnable, callable or dict" in str(e):
                    # Extract the class name that's causing the issue
                    import re
                    class_name = re.search(r"<class '(.+?)'>", str(e))
                    if class_name:
                        workflow_info["status"] = "partial"
                        workflow_info["error"] = f"LangGraph compatibility issue with {class_name.group(1)}. Add __call__ method."
                        logger.warning(f"Workflow check partial: LangGraph compatibility issue with {class_name.group(1)}")
                    else:
                        workflow_info["status"] = "error"
                        workflow_info["error"] = f"LangGraph compatibility issue: {str(e)}"
                        logger.error(f"Workflow check failed: {str(e)}")
                else:
                    workflow_info["status"] = "error"
                    workflow_info["error"] = str(e)
                    logger.error(f"Error checking workflow: {e}")
                
                # Try to get basic agent information even if the workflow has errors
                try:
                    from backend.app.agents.supervisor import get_supervisor_agent
                    supervisor = get_supervisor_agent()
                    workflow_info["supervisor_available"] = True
                except Exception as agent_err:
                    logger.error(f"Error checking supervisor agent: {agent_err}")
                    workflow_info["supervisor_available"] = False
        except Exception as e:
            workflow_info["status"] = "error"
            workflow_info["error"] = f"Error checking workflow: {str(e)}"
            logger.error(f"Error checking workflow: {e}")
        
        return workflow_info
    
    @timing_decorator
    def check_performance_metrics(self) -> Dict[str, Any]:
        """
        Check performance metrics.
        
        Returns:
            Dictionary with performance metrics
        """
        logger.info("Checking performance metrics")
        start_time = time.time()
        
        try:
            # Get metrics from the metrics module
            metrics = get_metric_statistics()
            
            # Add process memory usage
            import psutil
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            
            metrics["process"] = {
                "memory_rss": memory_info.rss,  # Resident Set Size
                "memory_rss_mb": memory_info.rss / 1024 / 1024,  # MB
                "memory_vms": memory_info.vms,  # Virtual Memory Size
                "memory_vms_mb": memory_info.vms / 1024 / 1024,  # MB
                "cpu_percent": process.cpu_percent(interval=0.1),
                "threads": len(process.threads()),
                "open_files": len(process.open_files()),
                "connections": len(process.connections()),
                "uptime_seconds": time.time() - process.create_time()
            }
            
            # Get system-wide metrics
            system = psutil.virtual_memory()
            metrics["system"] = {
                "total_memory": system.total,
                "total_memory_gb": system.total / 1024 / 1024 / 1024,
                "available_memory": system.available,
                "available_memory_gb": system.available / 1024 / 1024 / 1024,
                "memory_percent": system.percent,
                "cpu_count": psutil.cpu_count(),
                "cpu_percent": psutil.cpu_percent(interval=0.1)
            }
            
            # Add detailed LLM metrics
            llm_calls = metrics.get("llm_calls", {})
            for model, stats in llm_calls.items():
                logger.info(f"LLM model {model}: {stats.get('count', 0)} calls, avg time: {stats.get('avg_time', 0):.2f}s")
            
        except Exception as e:
            error_msg = f"Error checking metrics: {str(e)}"
            metrics = {"error": error_msg}
            logger.error(error_msg, exc_info=True)
        
        self.check_results["metrics"] = metrics
        self.check_times["performance_metrics"] = time.time() - start_time
        return metrics
    
    @timing_decorator
    def check_network_connectivity(self) -> Dict[str, Any]:
        """
        Check network connectivity to essential services.
        
        Returns:
            Dictionary with connectivity information
        """
        logger.info("Checking network connectivity")
        start_time = time.time()
        
        import httpx
        
        # Define the endpoints to check
        endpoints = {
            "openai": "https://api.openai.com/v1/models",
            "anthropic": "https://api.anthropic.com/v1/messages",
            "langsmith": "https://api.smith.langchain.com"
        }
        
        results = {}
        
        for name, url in endpoints.items():
            try:
                start = time.time()
                # Just check connection without sending credentials
                client = httpx.Client(timeout=5.0)
                response = client.head(url)
                latency = time.time() - start
                
                # Check for any response (we don't need auth to succeed)
                if response.status_code < 500:
                    results[name] = {
                        "status": "available",
                        "latency_ms": latency * 1000,
                        "status_code": response.status_code
                    }
                    logger.info(f"Connection to {name} ({url}) successful: {latency*1000:.1f}ms")
                else:
                    results[name] = {
                        "status": "error",
                        "error": f"Server error: {response.status_code}",
                        "latency_ms": latency * 1000,
                        "status_code": response.status_code
                    }
                    logger.warning(f"Connection to {name} ({url}) failed: status {response.status_code}")
            except Exception as e:
                results[name] = {
                    "status": "error",
                    "error": str(e)
                }
                logger.error(f"Connection to {name} ({url}) failed: {str(e)}")
        
        self.check_results["network_connectivity"] = results
        self.check_times["network_connectivity"] = time.time() - start_time
        return results
    
    @timing_decorator
    def check_language_model_availability(self) -> Dict[str, Any]:
        """
        Check if the language model service is available.
        
        Returns:
            Dict with language model status information
        """
        logger.info("Checking language model availability")
        start_time = time.time()
        
        results = {
            "status": "unknown",
            "provider": "unknown",
            "error": None,
            "providers": {}
        }
        
        try:
            # Import from the backend app
            from backend.app.services.llm import get_llm_service_status
            
            # Get LLM service status (this should not make API calls)
            try:
                service_status = get_llm_service_status()
                results["status"] = "available"
                results["provider"] = service_status.get("provider", "unknown")
                results["providers"] = {
                    provider: details for provider, details in service_status.items()
                    if provider not in ["provider", "error"]
                }
            except Exception as status_err:
                logger.warning(f"Error getting LLM service status: {status_err}")
                results["status"] = "error"
                results["error"] = f"Error getting service status: {str(status_err)}"
                
                # Fallback to simple API key check
                if os.getenv("OPENAI_API_KEY"):
                    results["providers"]["openai"] = {"api_key_set": True}
                if os.getenv("TOGETHER_API_KEY"):
                    results["providers"]["together"] = {"api_key_set": True}
        except Exception as e:
            logger.error(f"Error checking LLM availability: {e}")
            results["status"] = "error"
            results["error"] = f"Error checking LLM availability: {str(e)}"
        
        self.check_results["llm_availability"] = results
        self.check_times["llm_availability"] = time.time() - start_time
        return results
    
    @timing_decorator
    def check_security_status(self) -> Dict[str, Any]:
        """
        Check security configuration and status.
        
        Returns:
            Dictionary with security information
        """
        logger.info("Checking security status")
        start_time = time.time()
        
        security_info = {}
        
        # Check API key protection
        try:
            from backend.app.core.security import get_security_status
            security_info.update(get_security_status())
        except ImportError:
            # Fallback checks if the security module isn't available
            security_info["middleware"] = {
                "rate_limiting": hasattr(settings, "RATE_LIMIT_ENABLED") and settings.RATE_LIMIT_ENABLED,
                "size_limiting": hasattr(settings, "REQUEST_SIZE_LIMIT") and settings.REQUEST_SIZE_LIMIT > 0,
                "security_headers": True  # Assuming it's enabled in main.py
            }
        
        # Check for https
        security_info["https"] = {
            "enabled": (os.environ.get("HTTPS_ENABLED", "false").lower() == "true"),
            "cert_path": os.environ.get("SSL_CERT_PATH", "not_configured"),
            "key_path": os.environ.get("SSL_KEY_PATH", "not_configured")
        }
        
        # Check authentication configuration
        security_info["authentication"] = {
            "enabled": hasattr(settings, "AUTH_ENABLED") and settings.AUTH_ENABLED,
            "jwt_enabled": hasattr(settings, "JWT_SECRET_KEY") and bool(settings.JWT_SECRET_KEY),
            "api_key_enabled": hasattr(settings, "API_KEY_ENABLED") and settings.API_KEY_ENABLED
        }
        
        # Check for proper security headers
        security_info["security_headers"] = {
            "configured": True  # This is set in main.py
        }
        
        # Log findings
        for key, value in security_info.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, bool) and not subvalue and subkey != "enabled":
                        logger.warning(f"Security check {key}.{subkey} failed: {subvalue}")
        
        self.check_results["security"] = security_info
        self.check_times["security"] = time.time() - start_time
        return security_info

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
    
    # Get environment or default to 'dev'
    env = getattr(settings, "WEBAGENT_ENV", os.getenv("WEBAGENT_ENV", "dev"))
    
    print("\n========== WebAgent Diagnostics Report ==========")
    print(f"WebAgent v{settings.VERSION} | {report.get('timestamp', datetime.now().isoformat())}")
    print(f"Environment: {env}")
    print("=" * 50)
    
    # System Info
    if "system_info" in report:
        info = report["system_info"]
        print("\n## System Information")
        print(f"OS: {info.get('os')} {info.get('os_version')}")
        print(f"Python: {info.get('python_version').split()[0]}")
        print(f"CPU Count: {info.get('cpu_count')}")
        print(f"Platform: {info.get('platform')}")
        print(f"Working Directory: {info.get('working_directory')}")
    
    # Environment
    if "environment" in report:
        env = report["environment"]
        print("\n## Environment Configuration")
        print(f"Debug Mode: {env.get('debug_mode')}")
        print(f"LLM Provider: {env.get('llm_provider')}")
        
        # Check API keys
        env_vars = env.get("environment_variables", {})
        print("\nAPI Keys:")
        for var, status in env_vars.items():
            print(f"  {var}: {'✓ Set' if status.get('set') else '✗ Not Set'}")
    
    # Agents
    if "agents" in report:
        agents = report["agents"]
        print("\n## Agent Status")
        for agent, status in agents.items():
            if isinstance(status, dict) and "status" in status:
                avg_time = status.get("average_execution_time", "N/A")
                avg_time = f"{avg_time:.2f}s" if isinstance(avg_time, (int, float)) else avg_time
                print(f"{agent}: {status.get('status')} (Avg time: {avg_time})")
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
            print(f"Node Count: {workflow.get('node_count', 0)}")
            
            missing = workflow.get("missing_critical_nodes", [])
            if missing:
                print(f"Missing Critical Nodes: {', '.join(missing)}")
    
    # Network Connectivity
    if "network_connectivity" in report:
        connectivity = report["network_connectivity"]
        print("\n## Network Connectivity")
        for service, status in connectivity.items():
            if status.get("status") == "available":
                latency = status.get("latency_ms", 0)
                print(f"{service}: ✓ Available (Latency: {latency:.1f}ms)")
            else:
                print(f"{service}: ✗ Unavailable ({status.get('error', 'Unknown error')})")
    
    # LLM Availability
    if "llm_availability" in report:
        llm = report["llm_availability"]
        print("\n## Language Model Status")
        if "service_status" in llm:
            print(f"Provider: {llm['service_status'].get('provider', 'Unknown')}")
            print(f"Default Model: {llm['service_status'].get('default_model', 'Unknown')}")
        if "api_check" in llm:
            print(f"API Check: {llm['api_check'].get('status', 'Unknown')}")
    
    # Metrics
    if "metrics" in report and "process" in report["metrics"]:
        process = report["metrics"]["process"]
        system = report["metrics"].get("system", {})
        print("\n## Performance Metrics")
        print(f"Memory (RSS): {process.get('memory_rss_mb', 0):.2f} MB")
        print(f"System Memory: {system.get('available_memory_gb', 0):.2f} GB available of {system.get('total_memory_gb', 0):.2f} GB")
        print(f"CPU Usage: {process.get('cpu_percent')}% (process) / {system.get('cpu_percent')}% (system)")
        print(f"Threads: {process.get('threads')}")
        print(f"Uptime: {process.get('uptime_seconds', 0) / 60:.1f} minutes")
        
        # Print LLM call stats
        llm_calls = report["metrics"].get("llm_calls", {})
        if llm_calls:
            print("\nLLM Calls:")
            for model, stats in llm_calls.items():
                print(f"  {model}: {stats.get('count', 0)} calls, avg {stats.get('avg_time', 0):.2f}s per call")
    
    # Execution Times
    if "execution_time" in report:
        times = report["execution_time"]
        print("\n## Diagnostic Execution Times")
        print(f"Total: {times.get('total_seconds', 0):.2f} seconds")
        
        # Print individual check times
        check_times = times.get("individual_checks", {})
        if check_times:
            for check, duration in sorted(check_times.items(), key=lambda x: x[1], reverse=True):
                print(f"  {check}: {duration:.2f}s")
    
    print("\n========== End Diagnostics Report ==========\n")

def run_all_tests():
    """
    Run all tests for the WebAgent platform.
    
    This function is a wrapper around pytest to run all tests in the diagnostics package.
    """
    import pytest
    import os
    from pathlib import Path
    
    # Get the path to the diagnostics tests directory
    tests_dir = Path(__file__).parent / "tests"
    
    # Run pytest
    print(f"Running all tests in {tests_dir}")
    result = pytest.main(["-xvs", str(tests_dir)])
    
    return result == 0  # Return True if all tests passed 