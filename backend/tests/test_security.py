#!/usr/bin/env python
"""
Security tests for the LangGraph workflow.

This test suite focuses on security aspects of the LangGraph implementation:
1. Input validation and sanitization
2. Authorization in agent transitions
3. Data leak prevention
4. Injection prevention

Run with: pytest -xvs backend/tests/test_security.py
"""
import os
import sys
import pytest
import logging
import json
from typing import Dict, Any
from unittest.mock import patch, MagicMock

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import required modules
from app.models.task import WorkflowState
from app.graph.workflows import build_agent_workflow, get_agent_workflow

# Import test utilities
from tests.mock_utils import create_mock_agent, create_test_workflow_state, setup_test_environment


class TestLangGraphSecurity:
    """Security tests for the LangGraph workflow."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test environment before each test."""
        self.restore_env = setup_test_environment()
        yield
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
    
    @pytest.mark.asyncio
    async def test_input_validation(self, mock_agents):
        """Test that the workflow properly handles potentially dangerous inputs."""
        dangerous_inputs = [
            "DROP TABLE users;",  # SQL injection
            "<script>alert('XSS')</script>",  # XSS attempt
            "$(rm -rf /)",  # Command injection
            "file:///etc/passwd",  # Path traversal
            "data:text/html;base64,PHNjcmlwdD5hbGVydCgieHNzIik8L3NjcmlwdD4=",  # Data URI XSS
            "query=%3Cscript%3Ealert(%27xss%27)%3C/script%3E"  # URL encoded XSS
        ]
        
        # Configure supervisor to log the inputs
        received_inputs = []
        async def inspect_input(state):
            received_inputs.append(state.query)
            return state
            
        mock_agents["supervisor"].run.side_effect = inspect_input
        
        # Patch all the get_*_agent functions
        with patch('app.graph.workflows.get_supervisor_agent', return_value=mock_agents["supervisor"]), \
             patch('app.graph.workflows.get_web_research_agent', return_value=mock_agents["web_research"]), \
             patch('app.graph.workflows.get_internal_research_agent', return_value=mock_agents["internal_research"]), \
             patch('app.graph.workflows.get_senior_research_agent', return_value=mock_agents["senior_research"]), \
             patch('app.graph.workflows.get_data_analysis_agent', return_value=mock_agents["data_analysis"]), \
             patch('app.graph.workflows.get_coding_assistant_agent', return_value=mock_agents["coding_assistant"]), \
             patch('app.graph.workflows.get_team_manager_agent', return_value=mock_agents["team_manager"]):
            
            # Build the workflow
            workflow = build_agent_workflow()
            
            # Test each dangerous input
            for dangerous_input in dangerous_inputs:
                test_state = create_test_workflow_state(query=dangerous_input)
                
                # Run the workflow with supervisor only
                with patch('langgraph.graph.StateGraph.get_next_node', side_effect=["supervisor", None]):
                    await workflow.ainvoke(test_state)
            
            # Verify all inputs were passed through (expecting validation at agent level)
            assert len(received_inputs) == len(dangerous_inputs)
            for i, input_val in enumerate(dangerous_inputs):
                assert received_inputs[i] == input_val
    
    @pytest.mark.asyncio
    async def test_unauthorized_transitions(self, mock_agents):
        """Test that the workflow prevents unauthorized transitions between states."""
        # Set up a state that attempts to skip directly to team_manager
        test_state = create_test_workflow_state(
            query="Skip the process",
            current_step="team_manager"  # Trying to start at the final step
        )
        
        # Track which agents are called
        called_agents = set()
        
        # Configure all agents to record being called
        for name, agent in mock_agents.items():
            async def record_call(state, agent_name=name):
                called_agents.add(agent_name)
                return state
                
            agent.run.side_effect = record_call
        
        # Patch all the get_*_agent functions
        with patch('app.graph.workflows.get_supervisor_agent', return_value=mock_agents["supervisor"]), \
             patch('app.graph.workflows.get_web_research_agent', return_value=mock_agents["web_research"]), \
             patch('app.graph.workflows.get_internal_research_agent', return_value=mock_agents["internal_research"]), \
             patch('app.graph.workflows.get_senior_research_agent', return_value=mock_agents["senior_research"]), \
             patch('app.graph.workflows.get_data_analysis_agent', return_value=mock_agents["data_analysis"]), \
             patch('app.graph.workflows.get_coding_assistant_agent', return_value=mock_agents["coding_assistant"]), \
             patch('app.graph.workflows.get_team_manager_agent', return_value=mock_agents["team_manager"]):
            
            # Build the workflow
            workflow = build_agent_workflow()
            
            # Run the workflow
            await workflow.ainvoke(test_state)
            
            # Verify the workflow enforces the correct sequence
            # The LangGraph should always start with supervisor regardless of current_step
            assert "supervisor" in called_agents
            
    @pytest.mark.asyncio
    async def test_state_isolation(self, mock_agents):
        """Test that each workflow state is properly isolated to prevent data leaks between runs."""
        # Configure supervisor to add a secret to the state
        async def add_secret(state):
            state.context["secret_key"] = "very-secret-api-key"
            return state
            
        mock_agents["supervisor"].run.side_effect = add_secret
        
        # Configure team_manager to check for data persistence
        found_secrets = []
        async def check_secret(state):
            if "secret_key" in state.context:
                found_secrets.append(state.context["secret_key"])
            return state
            
        mock_agents["team_manager"].run.side_effect = check_secret
        
        # Patch all the get_*_agent functions
        with patch('app.graph.workflows.get_supervisor_agent', return_value=mock_agents["supervisor"]), \
             patch('app.graph.workflows.get_web_research_agent', return_value=mock_agents["web_research"]), \
             patch('app.graph.workflows.get_internal_research_agent', return_value=mock_agents["internal_research"]), \
             patch('app.graph.workflows.get_senior_research_agent', return_value=mock_agents["senior_research"]), \
             patch('app.graph.workflows.get_data_analysis_agent', return_value=mock_agents["data_analysis"]), \
             patch('app.graph.workflows.get_coding_assistant_agent', return_value=mock_agents["coding_assistant"]), \
             patch('app.graph.workflows.get_team_manager_agent', return_value=mock_agents["team_manager"]):
            
            # Build the workflow
            workflow = build_agent_workflow()
            
            # Run first workflow to add the secret
            first_state = create_test_workflow_state(query="First query")
            
            # Configure minimal pathing through the workflow
            route_mapping = {
                "supervisor": "senior_research",
                "senior_research": "data_analysis",
                "data_analysis": "team_manager",
                "team_manager": None
            }
            
            def route_next(state):
                return route_mapping.get(state.current_step)
                
            with patch('langgraph.graph.StateGraph.get_next_node', side_effect=route_next):
                await workflow.ainvoke(first_state)
            
            # Run second workflow to check if secret persists
            second_state = create_test_workflow_state(query="Second query")
            
            with patch('langgraph.graph.StateGraph.get_next_node', side_effect=route_next):
                await workflow.ainvoke(second_state)
            
            # Verify the secret doesn't persist between workflow runs
            assert len(found_secrets) <= 1, "Secret key should not persist between workflow runs"
            
    @pytest.mark.asyncio
    async def test_error_message_sanitization(self, mock_agents):
        """Test that error messages are properly sanitized before being returned."""
        # Configure supervisor to trigger an error with sensitive info
        async def trigger_sensitive_error(state):
            error_msg = "Error processing API key: sk_live_123456789abcdef. Check credentials."
            state.mark_error(error_msg)
            return state
            
        mock_agents["supervisor"].run.side_effect = trigger_sensitive_error
        
        # Patch all the get_*_agent functions
        with patch('app.graph.workflows.get_supervisor_agent', return_value=mock_agents["supervisor"]), \
             patch('app.graph.workflows.get_web_research_agent', return_value=mock_agents["web_research"]), \
             patch('app.graph.workflows.get_internal_research_agent', return_value=mock_agents["internal_research"]), \
             patch('app.graph.workflows.get_senior_research_agent', return_value=mock_agents["senior_research"]), \
             patch('app.graph.workflows.get_data_analysis_agent', return_value=mock_agents["data_analysis"]), \
             patch('app.graph.workflows.get_coding_assistant_agent', return_value=mock_agents["coding_assistant"]), \
             patch('app.graph.workflows.get_team_manager_agent', return_value=mock_agents["team_manager"]):
            
            # Build the workflow
            workflow = build_agent_workflow()
            
            # Run the workflow
            test_state = create_test_workflow_state(query="Test query")
            final_state = await workflow.ainvoke(test_state)
            
            # Check if error message contains sensitive info
            assert final_state.error is not None
            assert "sk_live_" not in final_state.error, "Error message should not contain API key"

if __name__ == "__main__":
    pytest.main(["-xvs", __file__]) 