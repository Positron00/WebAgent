#!/usr/bin/env python
"""
Mock utilities for testing.

This module provides utilities to create mock objects for testing:
- Mock agents with standardized interfaces
- Test workflow states
- Other test fixtures
"""
import os
import logging
from typing import Dict, Any, Optional
from unittest.mock import MagicMock, AsyncMock

from app.models.task import WorkflowState

logger = logging.getLogger(__name__)

def create_mock_agent(name: str) -> MagicMock:
    """
    Create a mock agent with standard methods and attributes.
    
    Args:
        name: Name of the agent
        
    Returns:
        Mock agent with .run() method as AsyncMock
    """
    mock_agent = MagicMock()
    mock_agent.name = name
    mock_agent.agent_id = f"mock-{name}-12345678"
    
    # Create an AsyncMock for the run method
    mock_agent.run = AsyncMock()
    
    # Configure the default behavior to echo back the state
    async def default_run(state):
        state.update_with_agent_output(name, {"status": "completed"})
        return state
    
    mock_agent.run.side_effect = default_run
    
    # Add other standard methods and attributes
    mock_agent.get_status = MagicMock(return_value={
        "agent_id": mock_agent.agent_id,
        "name": name,
        "status": "active"
    })
    
    mock_agent.log_activity = MagicMock()
    
    return mock_agent

def create_test_workflow_state(
    query: str = "Test query",
    current_step: str = "planning",
    context: Optional[Dict[str, Any]] = None
) -> WorkflowState:
    """
    Create a test workflow state for testing.
    
    Args:
        query: The user query string
        current_step: The current workflow step
        context: Optional context dictionary
        
    Returns:
        A WorkflowState instance
    """
    return WorkflowState(
        query=query,
        context=context or {},
        current_step=current_step
    )

def setup_test_environment():
    """
    Set up the test environment with necessary environment variables.
    """
    # Set testing mode
    os.environ["TESTING"] = "True"
    
    # Set minimal required environment variables
    os.environ.setdefault("WEBAGENT_ENV", "test")
    
    logger.info("Test environment configured")
    
    # Return a function to restore the environment
    original_env = dict(os.environ)
    
    def restore_environment():
        """Restore the original environment variables."""
        os.environ.clear()
        os.environ.update(original_env)
        logger.info("Original environment restored")
        
    return restore_environment 