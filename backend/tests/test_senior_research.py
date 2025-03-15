"""
Test the Senior Research Agent and research loop functionality.
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime
import json

from app.agents.senior_research import SeniorResearchAgent, ResearchEvaluation
from app.models.task import WorkflowState

# Import test utilities
from backend.tests.mock_utils import create_test_workflow_state

# Mock responses for testing
EVALUATION_RESPONSE = """```evaluation
score: 6
missing_information: ["detailed statistics on market growth", "competitor analysis"]
```

Based on the research conducted, here is a synthesized report addressing the query about AI assistant market trends:

# AI Assistant Market Trends Report

## Overview
The AI assistant market has shown significant growth over the past several years, with increasing adoption across various sectors.

## Key Findings
- Market is expanding rapidly
- Voice-based assistants dominate the consumer segment
- Enterprise adoption is accelerating

## Conclusion
While the research provides a good overview, more detailed statistics and competitor analysis would strengthen the report.
"""

COMPLETE_RESPONSE = """```evaluation
score: 9
missing_information: []
requires_additional_research: no
research_questions: []
```

Based on the comprehensive research conducted, here is a synthesized report addressing the query about AI assistant market trends:

# AI Assistant Market Trends Report

## Overview
The AI assistant market has shown remarkable growth, with a CAGR of 28.5% from 2021-2025, and is projected to reach $45.5 billion by 2026.

## Key Findings
- Market is expanding rapidly (28.5% CAGR)
- Voice-based assistants account for 65% of consumer usage
- Enterprise adoption increased by 35% year-over-year
- Top competitors include Google Assistant (28% market share), Amazon Alexa (25%), Apple Siri (18%), and Microsoft Cortana (10%)

## Conclusion
The AI assistant market continues to evolve with significant growth potential in healthcare, automotive, and enterprise sectors.
"""


@pytest.fixture
def senior_research_agent():
    """Create a Senior Research Agent for testing."""
    agent = SeniorResearchAgent()
    agent.chain = AsyncMock()
    return agent


@pytest.mark.asyncio
async def test_parse_evaluation():
    """Test parsing evaluation data from LLM output."""
    agent = SeniorResearchAgent()
    
    # Test with valid evaluation block
    evaluation = agent._parse_evaluation(EVALUATION_RESPONSE)
    assert evaluation["score"] == 6
    assert len(evaluation["missing_information"]) == 2
    assert evaluation["requires_additional_research"] is True
    assert len(evaluation["research_questions"]) == 2
    
    # Test with no evaluation block
    evaluation = agent._parse_evaluation("Just a regular text with no evaluation block")
    assert "score" in evaluation
    assert "missing_information" in evaluation
    assert "requires_additional_research" in evaluation
    assert "research_questions" in evaluation
    assert evaluation["requires_additional_research"] is False


@pytest.mark.asyncio
async def test_get_report_content():
    """Test getting report content from LLM output."""
    agent = SeniorResearchAgent()
    
    # Test with evaluation block
    content = agent._get_report_content(EVALUATION_RESPONSE)
    assert "AI Assistant Market Trends Report" in content
    assert "## Overview" in content
    assert "## Key Findings" in content
    
    # Test with no evaluation block
    content = agent._get_report_content("Just a regular text with no evaluation block")
    assert content == "Just a regular text with no evaluation block"


@pytest.mark.asyncio
async def test_requesting_additional_research():
    """Test requesting additional research after evaluation."""
    agent = SeniorResearchAgent()
    # Replace the whole chain instead of setting ainvoke directly
    original_chain = agent.chain
    agent.chain = AsyncMock()
    agent.chain.ainvoke = AsyncMock(return_value=EVALUATION_RESPONSE)
    
    state = create_test_workflow_state(
        query="What is the current state of AI assistants?",
        context={
            "web_research": "Some basic information about AI assistants",
            "internal_research": "Some knowledge base data about AI"
        }
    )
    
    try:
        result = await agent.run(state)
        
        # Verify that additional research was requested
        assert result.context.get("research_feedback") is not None
        assert result.context.get("continue_research") is True
        assert result.context.get("research_feedback").get("status") == "needs_more_research"
        assert result.context.get("research_feedback").get("score") == 6
        assert len(result.context.get("research_feedback").get("missing_information", [])) > 0
        assert len(result.context.get("research_feedback").get("research_questions", [])) > 0
        assert result.context.get("research_iteration", 0) > 1
    finally:
        # Restore original chain
        agent.chain = original_chain


@pytest.mark.asyncio
async def test_finalizing_research_report():
    """Test finalizing research after a complete evaluation."""
    agent = SeniorResearchAgent()
    # Replace the whole chain instead of setting ainvoke directly
    original_chain = agent.chain
    agent.chain = AsyncMock()
    agent.chain.ainvoke = AsyncMock(return_value=COMPLETE_RESPONSE)
    
    state = create_test_workflow_state(
        query="What is the current state of AI assistants?",
        context={
            "web_research": "Comprehensive information about AI assistants",
            "internal_research": "Detailed knowledge base data about AI",
            "research_iteration": 1
        }
    )
    
    try:
        result = await agent.run(state)
        
        # Verify that research was finalized
        assert result.context.get("continue_research") is False
        assert "verified_findings" in result.context
        assert result.final_report is not None
        assert len(result.final_report.get("content", "")) > 100
    finally:
        # Restore original chain
        agent.chain = original_chain


@pytest.mark.asyncio
async def test_max_iterations_reached():
    """Test finalizing research after reaching max iterations even if not complete."""
    agent = SeniorResearchAgent()
    # Replace the whole chain instead of setting ainvoke directly
    original_chain = agent.chain
    agent.chain = AsyncMock()
    agent.chain.ainvoke = AsyncMock(return_value=EVALUATION_RESPONSE)  # Still needs more research
    
    # Set max_iterations to 2 for this test
    original_max_iterations = agent.max_iterations
    agent.max_iterations = 2
    
    state = create_test_workflow_state(
        query="What is the current state of AI assistants?",
        context={
            "web_research": "Some information about AI assistants",
            "internal_research": "Some knowledge base data about AI",
            "research_iteration": 2  # Already at max iterations
        }
    )
    
    try:
        result = await agent.run(state)
        
        # Verify that research was finalized despite needing more
        assert "verified_findings" in result.context
        assert result.final_report is not None
        assert result.context.get("continue_research") is False
    finally:
        # Restore original values
        agent.chain = original_chain
        agent.max_iterations = original_max_iterations


@pytest.mark.asyncio
async def test_error_handling():
    """Test error handling during senior research agent execution."""
    agent = SeniorResearchAgent()
    # Replace the whole chain instead of setting ainvoke directly
    original_chain = agent.chain
    agent.chain = AsyncMock()
    agent.chain.ainvoke = AsyncMock(side_effect=Exception("Test error"))
    
    state = create_test_workflow_state(
        query="What is the current state of AI assistants?",
        context={
            "web_research": "Some information about AI assistants",
            "internal_research": "Some knowledge base data about AI"
        }
    )
    
    try:
        result = await agent.run(state)
        
        # Verify error handling
        assert result.error is not None
        assert "Test error" in result.error
    finally:
        # Restore original chain
        agent.chain = original_chain 