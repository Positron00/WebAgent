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
from .mock_utils import create_test_workflow_state

# Mock responses for testing
EVALUATION_RESPONSE = """```evaluation
score: 6
missing_information: ["detailed statistics on market growth", "competitor analysis"]
requires_additional_research: yes
research_questions: ["What are the detailed market growth statistics for AI assistants?", "Who are the top competitors in this space?"]
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
    """Test parsing the evaluation from the LLM output."""
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
    """Test extracting the report content from the LLM output."""
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
    agent.chain.ainvoke = AsyncMock(return_value=EVALUATION_RESPONSE)
    
    # Create a state with initial research
    state = create_test_workflow_state(
        query="What are the current market trends for AI assistants?",
        context={
            "research_plan": {
                "analysis": "Analyze current market trends for AI assistants",
                "requires_web_search": True,
                "requires_internal_knowledge": True
            },
            "research_iteration": 1
        }
    )
    
    # Add initial research reports
    state.reports["web_research"] = {
        "findings": "Some initial web findings about AI assistants"
    }
    state.reports["internal_research"] = {
        "findings": "Some initial internal findings about AI assistants"
    }
    
    # Run the agent
    updated_state = await agent.run(state)
    
    # Verify that additional research was requested
    assert updated_state.context.get("continue_research") is True
    assert "research_feedback" in updated_state.context
    assert updated_state.context["research_iteration"] == 2
    assert "next_research_agents" in updated_state.context
    assert "web_research" in updated_state.context["next_research_agents"]
    assert "internal_research" in updated_state.context["next_research_agents"]
    
    # Verify the research feedback
    feedback = updated_state.context["research_feedback"]
    assert feedback["status"] == "needs_more_research"
    assert feedback["score"] == 6
    assert len(feedback["missing_information"]) == 2
    assert len(feedback["research_questions"]) == 2


@pytest.mark.asyncio
async def test_finalizing_research_report():
    """Test finalizing research after a complete evaluation."""
    agent = SeniorResearchAgent()
    agent.chain.ainvoke = AsyncMock(return_value=COMPLETE_RESPONSE)
    
    # Create a state with final research iteration
    state = create_test_workflow_state(
        query="What are the current market trends for AI assistants?",
        context={
            "research_plan": {
                "analysis": "Analyze current market trends for AI assistants",
                "requires_web_search": True,
                "requires_internal_knowledge": True
            },
            "research_iteration": 3  # Final iteration
        }
    )
    
    # Add research reports from previous iterations
    state.reports["web_research"] = {
        "findings": "Detailed web findings about AI assistant market trends",
        "iteration": 3
    }
    state.reports["internal_research"] = {
        "findings": "Comprehensive internal findings about AI assistant market trends",
        "iteration": 3
    }
    
    # Run the agent
    updated_state = await agent.run(state)
    
    # Verify that research is complete
    assert updated_state.context.get("continue_research") is False
    assert "verified_findings" in updated_state.context
    
    # Verify the final report
    assert updated_state.final_report is not None
    assert "title" in updated_state.final_report
    assert "content" in updated_state.final_report
    assert updated_state.reports["senior_research"]["status"] == "completed"
    assert updated_state.reports["senior_research"]["score"] == 9


@pytest.mark.asyncio
async def test_max_iterations_reached():
    """Test finalizing research after reaching max iterations even if not complete."""
    agent = SeniorResearchAgent()
    agent.chain.ainvoke = AsyncMock(return_value=EVALUATION_RESPONSE)  # Still needs more research
    
    # Create a state at max iterations
    state = create_test_workflow_state(
        query="What are the current market trends for AI assistants?",
        context={
            "research_plan": {
                "analysis": "Analyze current market trends for AI assistants",
                "requires_web_search": True,
                "requires_internal_knowledge": True
            },
            "research_iteration": 3  # Max iteration
        }
    )
    
    # Add research reports
    state.reports["web_research"] = {
        "findings": "Some web findings about AI assistants",
        "iteration": 3
    }
    state.reports["internal_research"] = {
        "findings": "Some internal findings about AI assistants",
        "iteration": 3
    }
    
    # Run the agent
    updated_state = await agent.run(state)
    
    # Verify that research is complete despite evaluation suggesting more research
    assert updated_state.context.get("continue_research") is False
    assert "verified_findings" in updated_state.context
    assert updated_state.final_report is not None


@pytest.mark.asyncio
async def test_error_handling():
    """Test error handling during senior research agent execution."""
    agent = SeniorResearchAgent()
    agent.chain.ainvoke = AsyncMock(side_effect=Exception("Test error"))
    
    # Create a basic state
    state = create_test_workflow_state(
        query="What are the current market trends for AI assistants?"
    )
    
    # Run the agent
    updated_state = await agent.run(state)
    
    # Verify error is captured
    assert updated_state.error is not None
    assert "Senior Research Agent error" in updated_state.error 