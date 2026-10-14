#!/usr/bin/env python
"""
Test suite for the diagnostics utility module.

This test suite verifies the functionality of the diagnostics module, including:
1. System information checking
2. Dependency checking
3. Environment configuration validation
4. Agent status reporting
5. Workflow verification
6. Performance metrics
7. Network connectivity checking
8. LLM availability checking

Run with: pytest -xvs backend/tests/test_diagnostics.py
"""
import os
import sys
import pytest
import logging
import json
import tempfile
from unittest.mock import patch, MagicMock, PropertyMock
from io import StringIO

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # Add current directory too

# Import diagnostics module
try:
    # Try to import from app.utils (previous location)
    from app.utils.diagnostics import get_diagnostics, print_diagnostics_report, SystemDiagnostics
except ImportError:
    # Fall back to the current location
    from backend.diagnostics.core import get_diagnostics, print_diagnostics_report, SystemDiagnostics
from mock_utils import setup_test_environment

class TestDiagnosticsUtility:
    """Tests for the diagnostics utility module."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test environment before each test."""
        self.restore_env = setup_test_environment()
        
        # Create a diagnostics instance
        self.diagnostics = get_diagnostics()
        
        yield
        
        # Clean up
        self.restore_env()
    
    def test_system_info_check(self):
        """Test the system information check functionality."""
        # Run the system info check
        system_info = self.diagnostics.check_system_info()
        
        # Verify system info contains key fields
        assert "os" in system_info, "System info missing OS information"
        assert "python_version" in system_info, "System info missing Python version"
        assert "cpu_count" in system_info, "System info missing CPU count"
        assert "hostname" in system_info, "System info missing hostname"
        assert "working_directory" in system_info, "System info missing working directory"
        
        # Verify values are reasonable
        assert len(system_info["python_version"]) > 0, "Python version is empty"
        assert system_info["cpu_count"] > 0, "CPU count should be positive"
    
    def test_dependencies_check(self):
        """Test the dependencies check functionality."""
        # Run the dependencies check
        dependencies = self.diagnostics.check_dependencies()
        
        # Verify key packages are checked
        assert "langchain" in dependencies, "Dependencies should include langchain"
        assert "fastapi" in dependencies, "Dependencies should include fastapi"
        assert "pydantic" in dependencies, "Dependencies should include pydantic"
        
        # Verify at least some packages are installed (in test environment)
        installed_count = sum(1 for dep, info in dependencies.items() 
                              if info.get("status") == "installed")
        assert installed_count > 0, "No installed packages found"
    
    def test_environment_check(self):
        """Test the environment configuration check."""
        # Set some test environment variables
        os.environ["OPENAI_API_KEY"] = "sk-test-key-for-testing-only"
        os.environ["LANGSMITH_API_KEY"] = "ls-test-key-for-testing-only"
        
        # Run the environment check
        env_config = self.diagnostics.check_environment()
        
        # Verify environment config contains key fields
        assert "webagent_env" in env_config, "Missing environment type"
        assert "environment_variables" in env_config, "Missing environment variables check"
        
        # Check API key validation
        env_vars = env_config["environment_variables"]
        assert env_vars["OPENAI_API_KEY"]["set"], "OPENAI_API_KEY should be detected as set"
        assert env_vars["LANGSMITH_API_KEY"]["set"], "LANGSMITH_API_KEY should be detected as set"
    
    @patch('app.agents.supervisor.get_supervisor_agent')
    def test_agents_check(self, mock_get_supervisor):
        """Test the agents status check functionality."""
        # Create a mock supervisor agent
        mock_supervisor = MagicMock()
        mock_supervisor.get_agent_status.return_value = {
            "supervisor": {"status": "active", "initialized": True},
            "web_research": {"status": "active", "initialized": True},
            "internal_research": {"status": "active", "initialized": True},
            "document_extraction": {"status": "active", "initialized": True}
        }
        mock_get_supervisor.return_value = mock_supervisor
        
        # Run the agents check
        agent_status = self.diagnostics.check_agents()
        
        # Verify agent status contains key agents
        assert "supervisor" in agent_status, "Agent status missing supervisor"
        assert "web_research" in agent_status, "Agent status missing web_research"
        assert "internal_research" in agent_status, "Agent status missing internal_research"
        assert "document_extraction" in agent_status, "Agent status missing document_extraction"
        
        # Verify status information
        for agent, status in agent_status.items():
            assert isinstance(status, dict), f"Status for {agent} should be a dictionary"
            assert "status" in status, f"Status for {agent} missing 'status' field"
            assert status["status"] == "active", f"Status for {agent} should be 'active'"
    
    @patch('app.graph.workflows.get_agent_workflow')
    def test_workflow_check(self, mock_get_workflow):
        """Test the workflow check functionality."""
        # Configure mock
        mock_workflow = MagicMock()
        mock_get_workflow.return_value = mock_workflow
        
        # Run the check
        workflow_info = self.diagnostics.check_workflow()
        
        # Verify results
        assert workflow_info is not None, "Workflow check should return data"
        assert "nodes" in workflow_info, "Workflow check missing nodes"
        assert "compiled" in workflow_info, "Workflow check missing compiled status"
        assert "node_count" in workflow_info, "Workflow check missing node count"
        assert "edge_count" in workflow_info, "Workflow check missing edge count"
        
        # Verify workflow was checked
        mock_get_workflow.assert_called_once()
    
    def test_workflow_logs_check(self):
        """Test the workflow logs check functionality."""
        # Check if the workflow_logs check method exists
        if not hasattr(self.diagnostics, 'check_workflow_logs'):
            pytest.skip("check_workflow_logs method not available in this version")
            return
            
        # Method exists, run it directly
        logs_info = self.diagnostics.check_workflow_logs()
            
        # Verify results
        assert logs_info is not None, "Workflow logs check should return data"
        assert "status" in logs_info, "Workflow logs check missing status"
        assert "logs_implementation" in logs_info, "Workflow logs check missing logs_implementation"
        
        # Verify specific fields
        logs_impl = logs_info["logs_implementation"]
        assert "logs_added" in logs_impl, "Missing logs_added field"
        assert "logs_in_report_when_agentic" in logs_impl, "Missing logs_in_report_when_agentic field"
        assert "logs_excluded_when_non_agentic" in logs_impl, "Missing logs_excluded_when_non_agentic field"
        
        # All checks should be true
        assert logs_impl["logs_added"] is True, "Should be able to add logs"
        assert logs_impl["logs_in_report_when_agentic"] is True, "Logs should be in report when agentic is on"
        assert logs_impl["logs_excluded_when_non_agentic"] is True, "Logs should be excluded when agentic is off"
    
    @patch('psutil.Process')
    def test_performance_metrics(self, mock_process):
        """Test the performance metrics collection functionality."""
        # Configure mock process
        mock_proc = MagicMock()
        mock_memory_info = MagicMock()
        mock_memory_info.rss = 150 * 1024 * 1024  # 150MB
        mock_memory_info.vms = 500 * 1024 * 1024  # 500MB
        mock_proc.memory_info.return_value = mock_memory_info
        mock_proc.cpu_percent.return_value = 5.5
        mock_proc.threads.return_value = [1, 2, 3, 4]  # 4 threads
        mock_proc.open_files.return_value = []
        mock_proc.connections.return_value = []
        mock_proc.create_time.return_value = 0  # Start time
        mock_process.return_value = mock_proc
        
        # Mock psutil.virtual_memory
        with patch('psutil.virtual_memory') as mock_vm, \
             patch('psutil.cpu_percent') as mock_cpu:
            
            # Configure virtual memory mock
            mock_vm_obj = MagicMock()
            mock_vm_obj.total = 16 * 1024 * 1024 * 1024  # 16GB
            mock_vm_obj.available = 8 * 1024 * 1024 * 1024  # 8GB
            mock_vm_obj.percent = 50.0
            mock_vm.return_value = mock_vm_obj
            
            # Configure CPU percent mock
            mock_cpu.return_value = 10.0
            
            # Run the performance metrics check
            metrics = self.diagnostics.check_performance_metrics()
            
            # Verify metrics contains key sections
            assert "process" in metrics, "Metrics missing process information"
            assert "system" in metrics, "Metrics missing system information"
            
            # Verify process metrics
            process = metrics["process"]
            assert process["memory_rss_mb"] > 0, "RSS memory should be positive"
            assert process["memory_vms_mb"] > 0, "VMS memory should be positive"
            assert process["cpu_percent"] >= 0, "CPU percent should be non-negative"
            assert process["threads"] > 0, "Thread count should be positive"
            
            # Verify system metrics
            system = metrics["system"]
            assert system["total_memory_gb"] > 0, "Total memory should be positive"
            assert system["available_memory_gb"] > 0, "Available memory should be positive"
            assert 0 <= system["memory_percent"] <= 100, "Memory percent should be between 0 and 100"
            assert system["cpu_count"] > 0, "CPU count should be positive"
    
    @patch('httpx.Client')
    def test_network_connectivity(self, mock_client):
        """Test the network connectivity checking functionality."""
        # Configure mock client responses
        mock_response_openai = MagicMock()
        mock_response_openai.status_code = 200
        
        mock_response_anthropic = MagicMock()
        mock_response_anthropic.status_code = 200
        
        mock_response_langsmith = MagicMock()
        mock_response_langsmith.status_code = 401  # Auth required but reachable
        
        mock_client_instance = MagicMock()
        mock_client_instance.head.side_effect = lambda url: {
            "https://api.openai.com/v1/models": mock_response_openai,
            "https://api.anthropic.com/v1/messages": mock_response_anthropic,
            "https://api.smith.langchain.com": mock_response_langsmith
        }[url]
        
        mock_client.return_value = mock_client_instance
        
        # Run the network connectivity check
        connectivity = self.diagnostics.check_network_connectivity()
        
        # Verify connectivity contains all services
        assert "openai" in connectivity, "Connectivity missing OpenAI check"
        assert "anthropic" in connectivity, "Connectivity missing Anthropic check"
        assert "langsmith" in connectivity, "Connectivity missing LangSmith check"
        
        # Verify status information
        assert connectivity["openai"]["status"] == "available", "OpenAI should be available"
        assert connectivity["anthropic"]["status"] == "available", "Anthropic should be available"
        assert connectivity["langsmith"]["status"] == "available", "LangSmith should be available"
        assert "latency_ms" in connectivity["openai"], "OpenAI check missing latency info"
    
    def test_llm_availability(self):
        """Test the LLM availability checking functionality."""
        llm_status = self.diagnostics.check_language_model_availability()

        assert "service_status" in llm_status, "LLM status missing service_status"
        service_status = llm_status["service_status"]
        assert "provider" in service_status
        assert "openai" in service_status
        assert "together" in service_status
        assert "api_key_set" in service_status["openai"]
    
    @patch('app.core.security.get_security_status', create=True)
    def test_security_status(self, mock_get_security):
        """Test the security status checking functionality."""
        # Configure mock security status
        mock_get_security.return_value = {
            "middleware": {
                "rate_limiting": True,
                "size_limiting": True,
                "security_headers": True
            },
            "authentication": {
                "enabled": True,
                "api_key_enabled": True,
            }
        }
        
        # Run the security status check
        security_info = self.diagnostics.check_security_status()
        
        # Verify security info contains key sections
        assert "middleware" in security_info, "Security info missing middleware section"
        assert "authentication" in security_info, "Security info missing authentication section"
        assert "https" in security_info, "Security info missing HTTPS section"
        
        # Verify middleware settings
        middleware = security_info["middleware"]
        assert middleware["rate_limiting"], "Rate limiting should be enabled"
        assert middleware["size_limiting"], "Size limiting should be enabled"
        assert middleware["security_headers"], "Security headers should be enabled"
        
        # Verify authentication settings
        auth = security_info["authentication"]
        assert auth["enabled"], "Authentication should be enabled"
        assert auth["api_key_enabled"], "API key authentication should be enabled"
    
    def test_run_all_checks(self):
        """Test running all diagnostic checks together."""
        # Create a minimal mock implementation
        def mock_check_system_info(self):
            result = {"os": "test"}
            self.check_results["system_info"] = result
            return result
            
        def mock_check_dependencies(self):
            result = {"package": {"status": "installed"}}
            self.check_results["dependencies"] = result
            return result
            
        def mock_check_environment(self):
            result = {"webagent_env": "test"}
            self.check_results["environment"] = result
            return result
            
        def mock_check_agents(self):
            result = {"agent": {"status": "active"}}
            self.check_results["agents"] = result
            return result
            
        def mock_check_workflow(self):
            result = {"nodes": ["test"]}
            self.check_results["workflow"] = result
            return result
            
        def mock_check_performance_metrics(self):
            result = {"process": {"memory_rss": 100}}
            self.check_results["metrics"] = result
            return result
            
        def mock_check_network_connectivity(self):
            result = {"openai": {"status": "available"}}
            self.check_results["network_connectivity"] = result
            return result
            
        def mock_check_language_model_availability(self):
            result = {"status": "ok"}
            self.check_results["llm_availability"] = result
            return result
            
        def mock_check_security_status(self):
            result = {"middleware": {"rate_limiting": True}}
            self.check_results["security"] = result
            return result
        
        with patch.multiple(
            SystemDiagnostics,
            check_system_info=mock_check_system_info,
            check_dependencies=mock_check_dependencies,
            check_environment=mock_check_environment,
            check_agents=mock_check_agents,
            check_workflow=mock_check_workflow,
            check_performance_metrics=mock_check_performance_metrics,
            check_network_connectivity=mock_check_network_connectivity,
            check_language_model_availability=mock_check_language_model_availability,
            check_security_status=mock_check_security_status
        ):
            # Run all checks
            results = self.diagnostics.run_all_checks()
            
            # Verify results contain all sections
            assert "system_info" in results, "Missing system info in results"
            assert "dependencies" in results, "Missing dependencies in results"
            assert "environment" in results, "Missing environment in results"
            assert "agents" in results, "Missing agents in results"
            assert "workflow" in results, "Missing workflow in results"
            assert "metrics" in results, "Missing metrics in results"
            assert "network_connectivity" in results, "Missing network connectivity in results"
            assert "llm_availability" in results, "Missing LLM availability in results"
            assert "security" in results, "Missing security in results"
            assert "timestamp" in results, "Missing timestamp in results"
            assert "execution_time" in results, "Missing execution time in results"
    
    def test_singleton_pattern(self):
        """Test that the diagnostics utility follows the singleton pattern."""
        # Get two instances of the diagnostics utility
        diagnostics1 = get_diagnostics()
        diagnostics2 = get_diagnostics()
        
        # Verify they are the same instance
        assert diagnostics1 is diagnostics2, "Diagnostics instances should be the same object"
    
    def test_print_diagnostics_report(self):
        """Test the diagnostic report printing functionality."""
        # Create a mock diagnostics instance
        with patch('app.utils.diagnostics.get_diagnostics') as mock_get_diagnostics:
            mock_diagnostics = MagicMock()
            mock_diagnostics.check_results = {
                "timestamp": "2025-03-14T12:34:56.789012",
                "system_info": {
                    "os": "TestOS",
                    "os_version": "1.0",
                    "python_version": "3.10.0 (Test)",
                    "cpu_count": 4,
                    "platform": "TestPlatform",
                    "working_directory": "/test/dir"
                },
                "environment": {
                    "webagent_env": "test",
                    "debug_mode": True,
                    "llm_provider": "test-provider",
                    "environment_variables": {
                        "TEST_API_KEY": {"set": True, "value_length": 10}
                    }
                },
                "agents": {
                    "test-agent": {"status": "active", "average_execution_time": 1.5}
                },
                "workflow": {
                    "nodes": ["start", "middle", "end"],
                    "compiled": True,
                    "node_count": 3,
                    "edge_count": 2,
                    "missing_critical_nodes": []
                },
                "network_connectivity": {
                    "test-service": {"status": "available", "latency_ms": 100.5}
                },
                "llm_availability": {
                    "service_status": {
                        "provider": "test-provider",
                        "default_model": "test-model"
                    },
                    "api_check": {"status": "check_disabled"}
                },
                "metrics": {
                    "process": {
                        "memory_rss_mb": 100.5,
                        "cpu_percent": 5.0,
                        "threads": 10,
                        "uptime_seconds": 300
                    },
                    "system": {
                        "available_memory_gb": 8.0,
                        "total_memory_gb": 16.0,
                        "cpu_percent": 10.0
                    },
                    "llm_calls": {
                        "test-model": {"count": 100, "avg_time": 0.5}
                    }
                },
                "execution_time": {
                    "total_seconds": 0.5,
                    "individual_checks": {
                        "system_info": 0.1,
                        "agents": 0.2,
                        "metrics": 0.2
                    }
                }
            }
            mock_diagnostics.run_all_checks.return_value = mock_diagnostics.check_results
            mock_get_diagnostics.return_value = mock_diagnostics
            
            # Capture stdout
            captured_output = StringIO()
            import sys
            original_stdout = sys.stdout
            sys.stdout = captured_output
            
            try:
                # Print the diagnostics report
                print_diagnostics_report()
                
                # Get the captured output
                output = captured_output.getvalue()
                
                # Verify key sections are in the output
                assert "WebAgent Diagnostics Report" in output, "Missing report header"
                assert "System Information" in output, "Missing system information section"
                assert "Environment Configuration" in output, "Missing environment section"
                assert "Agent Status" in output, "Missing agent status section"
                assert "Workflow" in output, "Missing workflow section"
                assert "Network Connectivity" in output, "Missing network connectivity section"
                assert "Language Model Status" in output, "Missing language model section"
                assert "Performance Metrics" in output, "Missing performance metrics section"
                assert "Diagnostic Execution Times" in output, "Missing execution times section"
                assert "End Diagnostics Report" in output, "Missing report footer"
                
                # Verify key values are in the output
                assert "TestOS" in output, "Missing OS info in output"
                assert "test-agent: active" in output, "Missing agent status in output"
                assert "Memory (RSS): 100.50 MB" in output, "Missing memory info in output"
                assert "test-model: 100 calls" in output, "Missing LLM call stats in output"
                
            finally:
                # Restore stdout
                sys.stdout = original_stdout


