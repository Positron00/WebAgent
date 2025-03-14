#!/usr/bin/env python
"""
Tests for the TeamManagerAgent component.

This test suite focuses on testing the TeamManagerAgent:
1. Proper output formatting
2. Correct interaction with workflow state
3. UI output generation
4. Final report formatting

Run with: pytest -xvs backend/diagnostics/tests/test_team_manager.py
"""
import os
import sys
import json
import pytest
import logging
import asyncio
from typing import Dict, Any, Optional
from unittest.mock import patch, MagicMock, AsyncMock

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import the modules to test
from backend.app.models.task import WorkflowState
from backend.app.agents.team_manager import TeamManagerAgent, get_team_manager_agent

class TestTeamManagerAgent:
    """Tests for the TeamManagerAgent implementation."""
    
    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM that returns a valid JSON response."""
        # Create a mock LLM with a simple invoke method
        mock = MagicMock()
        
        # The LLM should return a properly formatted JSON string
        json_response = {
            "summary": "This is a summary of the findings",
            "recommendations": [
                "First recommendation",
                "Second recommendation",
                "Third recommendation with details"
            ],
            "analysis": "Detailed analysis of the query and research results...",
            "next_steps": "Suggested next steps for the user",
            "sources": [
                {"url": "https://example.com/1", "title": "Example source 1"},
                {"url": "https://example.com/2", "title": "Example source 2"}
            ]
        }
        
        # Create a side effect function for the mock
        async def mock_invoke(inputs, **kwargs):
            return json.dumps(json_response)
        
        # Patch the original method with our mock
        mock.invoke = AsyncMock(side_effect=mock_invoke)
        
        return mock
    
    @pytest.fixture
    def test_state(self):
        """Create a test workflow state with research results."""
        state = WorkflowState(
            query="What are the latest advancements in LangGraph?",
            current_step="team_manager",
            reports={
                "supervisor": {
                    "analysis": "Query requires web research into LangGraph."
                },
                "web_research": {
                    "search_results": [
                        {"url": "https://python.langchain.com/docs/langgraph", 
                         "title": "LangGraph Documentation",
                         "snippet": "LangGraph is a library for building stateful..."}
                    ],
                    "extracted_info": "LangGraph is a library for building stateful, multi-actor applications with LLMs."
                },
                "senior_research": {
                    "summary": "LangGraph is a powerful framework for orchestrating multi-agent systems.",
                    "key_points": ["Directed graphs", "State management", "Multi-agent orchestration"]
                }
            },
            context={
                "research_completed": True,
                "research_sources": [
                    {"url": "https://python.langchain.com/docs/langgraph", "title": "LangGraph Documentation"}
                ],
                "verified_findings": "LangGraph is a framework for building stateful multi-agent systems.",
                "data_insights": "LangGraph usage has increased significantly in 2023.",
                "visualization_code": "# Sample code for visualization\nimport matplotlib.pyplot as plt\n"
            }
        )
        return state
    
    @pytest.mark.asyncio
    async def test_agent_initialization(self, mock_llm):
        """Test that the agent initializes correctly."""
        with patch('backend.app.agents.team_manager.get_llm', return_value=mock_llm):
            agent = TeamManagerAgent()
            
            # Check basic properties
            assert agent.name == "team_manager", "Agent name should be 'team_manager'"
            assert agent.prompt is not None, "Agent should have a prompt template"
    
    @pytest.mark.asyncio
    async def test_agent_run(self, mock_llm, test_state):
        """Test that the agent processes the state correctly."""
        with patch('backend.app.agents.team_manager.get_llm', return_value=mock_llm):
            agent = TeamManagerAgent()
            
            # Mock the chain's ainvoke method
            agent.chain = MagicMock()
            agent.chain.ainvoke = AsyncMock(return_value=json.dumps({
                "summary": "Test summary",
                "recommendations": ["Test recommendation"],
                "analysis": "Test analysis",
                "next_steps": "Test next steps",
                "sources": [{"url": "https://example.com", "title": "Example"}]
            }))
            
            # Run the agent
            result_state = await agent.run(test_state)
            
            # Verify the mock chain was called
            agent.chain.ainvoke.assert_called_once()
            
            # Verify the state was updated with the agent's output
            assert "team_manager" in result_state.reports
            assert "summary" in result_state.reports["team_manager"]
            assert "recommendations" in result_state.reports["team_manager"]
            assert "analysis" in result_state.reports["team_manager"]
            
            # Verify the final report was created
            assert result_state.final_report is not None
            assert "content" in result_state.final_report
    
    @pytest.mark.asyncio
    async def test_final_report_formatting(self, mock_llm, test_state):
        """Test that the final report is properly formatted for the UI."""
        with patch('backend.app.agents.team_manager.get_llm', return_value=mock_llm):
            agent = TeamManagerAgent()
            
            # Mock the chain's ainvoke method
            agent.chain = MagicMock()
            agent.chain.ainvoke = AsyncMock(return_value=json.dumps({
                "summary": "Test summary for UI",
                "recommendations": ["Test recommendation for UI"],
                "analysis": "Test analysis for UI",
                "next_steps": "Test next steps",
                "sources": [{"url": "https://example.com", "title": "Example"}]
            }))
            
            # Run the agent
            result_state = await agent.run(test_state)
            
            # Verify final report structure for UI rendering
            assert "title" in result_state.final_report
            assert "content" in result_state.final_report
            
            # Check that the content includes the summary
            assert result_state.reports["team_manager"]["summary"] in result_state.final_report["content"]
    
    @pytest.mark.asyncio
    async def test_sources_handling(self, mock_llm, test_state):
        """Test that research sources are properly included in the output."""
        with patch('backend.app.agents.team_manager.get_llm', return_value=mock_llm):
            agent = TeamManagerAgent()
            
            # Mock the chain's ainvoke method
            agent.chain = MagicMock()
            agent.chain.ainvoke = AsyncMock(return_value=json.dumps({
                "summary": "Test summary",
                "recommendations": ["Test recommendation"],
                "analysis": "Test analysis",
                "next_steps": "Test next steps",
                "sources": [
                    {"url": "https://python.langchain.com/docs/langgraph", "title": "LangGraph Documentation"},
                    {"url": "https://example.com/2", "title": "Example source 2"}
                ]
            }))
            
            # Run the agent
            result_state = await agent.run(test_state)
            
            # Check that sources are included
            assert "sources" in result_state.reports["team_manager"]
            
            # The sources should include at least one item
            sources = result_state.reports["team_manager"]["sources"]
            assert len(sources) > 0, "No sources found in report"
    
    @pytest.mark.asyncio
    async def test_singleton_pattern(self, mock_llm):
        """Test the singleton pattern for get_team_manager_agent."""
        with patch('backend.app.agents.team_manager.get_llm', return_value=mock_llm):
            # Get the agent twice
            agent1 = get_team_manager_agent()
            agent2 = get_team_manager_agent()
            
            # Check that we got the same instance
            assert agent1 is agent2, "get_team_manager_agent should return the same instance"
    
    @pytest.mark.asyncio
    async def test_invalid_llm_response(self):
        """Test handling of invalid LLM responses."""
        agent = TeamManagerAgent()
        
        # Mock the chain with an error response
        agent.chain = MagicMock()
        agent.chain.ainvoke = AsyncMock(side_effect=Exception("Invalid JSON response"))
        
        test_state = WorkflowState(query="Test query")
        
        # Run the agent - it should handle the exception gracefully
        result_state = await agent.run(test_state)
        
        # Verify the error handling worked
        assert "team_manager" in result_state.reports
        assert result_state.reports["team_manager"]["summary"] == "Error in report generation"
        assert len(result_state.reports["team_manager"]["recommendations"]) > 0
        assert result_state.error is not None

if __name__ == "__main__":
    pytest.main(["-xvs", __file__]) 