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
import time
import asyncio
from datetime import datetime
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
    logger.debug(f"Creating mock agent: {name}")
    start_time = time.time()
    
    # Create an AsyncMock for the agent
    mock_agent = AsyncMock()
    mock_agent.name = name
    mock_agent.agent_id = f"mock-{name}-12345678"
    
    # Create an AsyncMock for the run method
    mock_agent.run = AsyncMock()
    
    # Configure the default behavior to echo back the state
    async def default_run(state):
        logger.debug(f"Mock {name} agent running with query: {state.query}")
        exec_start = time.time()
        
        # Simulate some processing time
        await asyncio.sleep(0.01)
        
        # Add mock output to the state
        state.update_with_agent_output(name, {
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
            "execution_time": time.time() - exec_start
        })
        
        logger.debug(f"Mock {name} agent completed in {time.time() - exec_start:.3f}s")
        return {"state": state}
    
    mock_agent.run.side_effect = default_run
    
    # For LangGraph 0.3.5 compatibility - configure the agent itself as an async callable
    async def async_call(state_dict):
        # In LangGraph 0.3.5, the state might be passed directly
        from app.models.task import WorkflowState
        
        # Check if state_dict is already a WorkflowState
        if isinstance(state_dict, WorkflowState):
            state = state_dict
        else:
            # Extract the state from the input dictionary
            state = state_dict.get("state")
            
        logger.debug(f"Mock {name} agent called with state type: {type(state).__name__}")
            
        # Process the state with the agent's run method
        result = await mock_agent.run(state)
        # Return the updated state in the same format
        return result
    
    # Set the side_effect directly on the AsyncMock
    mock_agent.side_effect = async_call
    
    # Add other standard methods and attributes
    mock_agent.get_status = MagicMock(return_value={
        "agent_id": mock_agent.agent_id,
        "name": name,
        "status": "active",
        "created_at": datetime.now().isoformat()
    })
    
    mock_agent.log_activity = MagicMock()
    
    logger.debug(f"Mock agent '{name}' created in {time.time() - start_time:.3f}s")
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
    logger.debug(f"Creating test workflow state with query: '{query}', step: '{current_step}'")
    
    state = WorkflowState(
        query=query,
        context=context or {},
        current_step=current_step
    )
    
    logger.debug(f"Test workflow state created: {state}")
    return state

def setup_test_environment():
    """
    Set up the test environment with necessary environment variables.
    """
    # Set testing mode
    os.environ["TESTING"] = "True"
    
    # Set minimal required environment variables
    os.environ.setdefault("WEBAGENT_ENV", "test")
    os.environ.setdefault("LOG_LEVEL", "DEBUG") 
    
    logger.info("Test environment configured with TESTING=True, WEBAGENT_ENV=test")
    
    # Return a function to restore the environment
    original_env = dict(os.environ)
    
    def restore_environment():
        """Restore the original environment variables."""
        os.environ.clear()
        os.environ.update(original_env)
        logger.info("Original environment restored")
        
    return restore_environment 