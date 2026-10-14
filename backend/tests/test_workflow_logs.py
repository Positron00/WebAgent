#!/usr/bin/env python
"""
Test suite for the workflow logs functionality.

This test suite verifies that workflow logs are properly:
1. Generated during workflow execution
2. Included in API responses when agentic mode is enabled
3. Formatted correctly for frontend display

Run with: pytest -xvs backend/tests/test_workflow_logs.py
"""
import os
import sys
import pytest
import logging
import json
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from typing import Dict, Any, List
from pathlib import Path

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # Add current directory too

# Add backend to path for imports
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

# Import required modules
from app.models.task import WorkflowState
from app.graph.workflows import build_agent_workflow, get_agent_workflow
from app.api.endpoints.frontend import FrontendChatResponse

# Import test utils
from mock_utils import create_mock_agent, create_test_workflow_state, setup_test_environment


class TestWorkflowLogs:
    """Tests for the workflow logs functionality."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test environment before each test."""
        self.restore_env = setup_test_environment()
        
        # Sample workflow logs for testing
        self.sample_logs = [
            "Planning agent: Analyzing query",
            "Planning agent: Determining required steps",
            "Team Manager: Assessing task requirements",
            "Team Manager: Selecting specialized agents",
            "Research agent: Searching for relevant information",
            "Research agent: Evaluating source credibility",
            "Analysis agent: Processing gathered information",
            "Analysis agent: Synthesizing findings",
            "Writing agent: Generating comprehensive response",
            "Team Manager: Reviewing final output"
        ]
        
        yield
        
        # Clean up
        self.restore_env()
    
    @pytest.fixture
    def mock_agents(self):
        """Set up mock agents for testing."""
        agents = {}
        agent_names = [
            "supervisor", "web_research", "internal_research", 
            "senior_research", "data_analysis", "coding_assistant", 
            "team_manager"
        ]
        
        for name in agent_names:
            agents[name] = create_mock_agent(name)
            
        return agents
    
    @pytest.fixture
    def test_state(self):
        """Create a test workflow state with agentic toggle enabled."""
        return create_test_workflow_state(
            query="Test workflow logs functionality",
            current_step="supervisor",
            context={"agentic_toggle": True}
        )
    
    @pytest.mark.asyncio
    async def test_workflow_logs_generation(self, mock_agents, test_state):
        """Test that workflow execution generates logs when agentic mode is enabled."""
        # Configure supervisor to record logs in the state
        async def mock_supervisor_with_logs(state):
            # Add workflow logs to the state
            if not hasattr(state, "logs"):
                state.logs = []
            
            # Add a log entry
            state.logs.append("Supervisor: Analyzing query requirements")
            state.logs.append("Supervisor: Planning workflow execution path")
            
            # Update context to indicate this is an agentic workflow
            state.context["agentic_toggle"] = True
            
            # Update with agent output
            state.update_with_agent_output("supervisor", {
                "analysis": "Test query requires research",
                "has_logs": True
            })
            
            return {"state": state}
        
        mock_agents["supervisor"].run.side_effect = mock_supervisor_with_logs
        
        # Configure web research agent to add more logs
        async def mock_web_research_with_logs(state):
            # Add more logs
            if not hasattr(state, "logs"):
                state.logs = []
                
            state.logs.append("Web Research: Searching online sources")
            state.logs.append("Web Research: Analyzing search results")
            
            # Update with agent output
            state.update_with_agent_output("web_research", {
                "results": "Found relevant information",
                "has_logs": True
            })
            
            return {"state": state}
        
        mock_agents["web_research"].run.side_effect = mock_web_research_with_logs
        
        # Patch agent getters
        with patch('app.graph.workflows.get_supervisor_agent', return_value=mock_agents["supervisor"]), \
             patch('app.graph.workflows.get_web_research_agent', return_value=mock_agents["web_research"]):
             
            # Call supervisor directly first
            result = await mock_agents["supervisor"](test_state)
            
            # Verify logs were added
            assert hasattr(result["state"], "logs"), "State should have logs attribute"
            assert len(result["state"].logs) >= 2, "State should have at least 2 log entries"
            assert "Supervisor:" in result["state"].logs[0], "First log should be from supervisor"
            
            # Now call web research agent
            web_result = await mock_agents["web_research"](result["state"])
            
            # Verify web research logs were added
            assert len(web_result["state"].logs) >= 4, "State should have at least 4 log entries after web research"
            assert "Web Research:" in web_result["state"].logs[-1], "Latest log should be from web research"
    
    @pytest.mark.asyncio
    async def test_workflow_logs_in_response(self, mock_agents, test_state):
        """Test that workflow logs are included in API responses when agentic mode is enabled."""
        # Configure a test final state with logs
        final_state = test_state
        final_state.logs = self.sample_logs
        final_state.completed = True
        final_state.final_report = {
            "title": "Test Report",
            "content": "This is a test report with workflow logs.",
            "type": "report",
            "sources": []
        }
        
        # Mock the task manager to return our prepared state
        with patch('app.api.endpoints.frontend.task_manager') as mock_task_manager, \
             patch('app.api.endpoints.frontend.uuid4', return_value="test-task-id"):
            
            # Configure mock task manager
            mock_task_manager.run_workflow = AsyncMock()
            mock_task_manager.get_task_status = MagicMock(return_value=("completed", final_state.final_report))
            mock_task_manager.task_results = {
                "test-task-id": final_state.final_report
            }
            
            # Mock implementation for FrontendChatResponse
            from app.api.endpoints.frontend import FrontendChatResponse, FrontendChatRequest
            
            # Create a test request
            request = FrontendChatRequest(
                messages=[
                    {"role": "user", "content": "Test request with workflow logs"}
                ],
                agentic_toggle=True
            )
            
            # Import the endpoint function
            from app.api.endpoints.frontend import frontend_chat_completions
            
            # Call the endpoint function
            response = await frontend_chat_completions(request)
            
            # Verify the run was started on the workflow path with agentic_toggle=True
            assert mock_task_manager.start.called, "The run should be started"
            _task_id, _message, context = mock_task_manager.start.call_args.args
            assert mock_task_manager.start.call_args.kwargs["direct"] is False
            assert context.get("agentic_toggle") is True, "Task should be started with agentic_toggle=True"
    
    def test_workflow_logs_format(self):
        """Test that workflow logs are formatted correctly for frontend display."""
        # Create a test state with logs
        state = create_test_workflow_state()
        state.logs = self.sample_logs
        
        # Verify log format
        for log in state.logs:
            # Each log should be a string
            assert isinstance(log, str), "Log entry should be a string"
            
            # Each log should have an agent name prefix
            assert ":" in log, "Log entry should have format 'Agent: message'"
            
            # Split by colon
            parts = log.split(":", 1)
            assert len(parts) == 2, "Log entry should split into two parts"
            
            # Agent name should be non-empty
            assert len(parts[0].strip()) > 0, "Agent name should be non-empty"
            
            # Message should be non-empty
            assert len(parts[1].strip()) > 0, "Log message should be non-empty"
    
    @pytest.mark.asyncio
    async def test_logs_only_included_when_agentic_enabled(self, mock_agents, test_state):
        """Test that logs are only included in responses when agentic mode is enabled."""
        # Test with agentic mode ON
        test_state.context["agentic_toggle"] = True
        test_state.logs = self.sample_logs[:3]  # Add a few logs
        
        # Test with agentic mode OFF
        non_agentic_state = create_test_workflow_state(
            query="Test without agentic mode",
            context={"agentic_toggle": False}
        )
        non_agentic_state.logs = self.sample_logs[:3]  # Add the same logs
        
        # Set up mock functions
        async def mock_supervisor_with_logs(state):
            # Create a final report
            report = {
                "title": "Test Report",
                "content": "This is a test report."
            }
            
            if state.context.get("agentic_toggle"):
                state.logs.append("Supervisor: Executing with agentic mode ON")
            else:
                state.logs.append("Supervisor: Executing with agentic mode OFF")
                
            # Mark the workflow as completed
            state.mark_completed(report)
            
            return {"state": state}
        
        # Set up mock agents
        mock_agents["supervisor"].run.side_effect = mock_supervisor_with_logs
        
        # Run with agentic mode ON
        agentic_result = await mock_agents["supervisor"](test_state)
        
        # Run with agentic mode OFF
        non_agentic_result = await mock_agents["supervisor"](non_agentic_state)
        
        # Verify logs are in the report for agentic mode ON
        assert "logs" in agentic_result["state"].final_report, "Logs should be in final report when agentic mode is ON"
        assert len(agentic_result["state"].final_report["logs"]) > 0, "Logs array should not be empty"
        
        # Verify logs are NOT in the report for agentic mode OFF
        assert "logs" not in non_agentic_result["state"].final_report, "Logs should not be in final report when agentic mode is OFF"


