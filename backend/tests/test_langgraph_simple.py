#!/usr/bin/env python
"""
Simple tests for the LangGraph framework implementation.

This test suite focuses on testing the LangGraph workflow construction.
"""
import os
import sys
import pytest
import logging
from unittest.mock import patch, MagicMock

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the modules to test
from app.models.task import WorkflowState
from app.graph.workflows import build_agent_workflow, get_agent_workflow
from langgraph.graph import StateGraph, END

# Import mock utils
from backend.tests.mock_utils import create_mock_agent

class TestLangGraphWorkflow:
    """Tests for the LangGraph workflow implementation."""
    
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
    async def test_workflow_construction(self, mock_agents):
        """Test that the workflow can be constructed properly."""
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
            
            # Check that workflow was created
            assert workflow is not None, "Workflow should be created"
    
    @pytest.mark.asyncio
    async def test_workflow_singleton(self, mock_agents):
        """Test the singleton pattern for get_agent_workflow."""
        # Patch all the get_*_agent functions for both calls
        with patch('app.graph.workflows.get_supervisor_agent', return_value=mock_agents["supervisor"]), \
             patch('app.graph.workflows.get_web_research_agent', return_value=mock_agents["web_research"]), \
             patch('app.graph.workflows.get_internal_research_agent', return_value=mock_agents["internal_research"]), \
             patch('app.graph.workflows.get_senior_research_agent', return_value=mock_agents["senior_research"]), \
             patch('app.graph.workflows.get_data_analysis_agent', return_value=mock_agents["data_analysis"]), \
             patch('app.graph.workflows.get_coding_assistant_agent', return_value=mock_agents["coding_assistant"]), \
             patch('app.graph.workflows.get_team_manager_agent', return_value=mock_agents["team_manager"]):
            
            # Get the workflow twice
            workflow1 = get_agent_workflow()
            workflow2 = get_agent_workflow()
            
            # Check that we got the same instance
            assert workflow1 is workflow2, "get_agent_workflow should return the same instance"

if __name__ == "__main__":
    pytest.main(["-xvs", __file__]) 