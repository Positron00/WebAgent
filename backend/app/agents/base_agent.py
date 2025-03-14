"""
Base Agent
==========

This module defines the BaseAgent class which is the foundation for all specialized agents
in the WebAgent platform. It provides common functionality and interface requirements.
"""

import logging
import time
from typing import Dict, Any, Optional
import json
import uuid
import os
import traceback
from abc import ABC, abstractmethod

# Import local modules
from app.core.config import settings

from langchain.schema.runnable import RunnableSequence
from langchain.schema.runnable import RunnableConfig

from app.core.setup_langsmith import get_langchain_tracer, create_langsmith_tags
from app.core.loadEnvYAML import get_langsmith_config
from app.models.task import WorkflowState

# Initialize logging
logger = logging.getLogger(__name__)

class BaseAgent:
    """
    Base class for all agents in the WebAgent platform.
    
    This class provides common functionality and defines
    the interface that all specialized agents must implement.
    """
    
    def __init__(self, name: str, config: Dict = None):
        """
        Initialize a base agent.
        
        Args:
            name: Unique agent name
            config: Configuration dictionary (falls back to settings if None)
        """
        self.name = name
        self.agent_id = f"{name}-{uuid.uuid4().hex[:8]}"
        self.config = config or getattr(settings, f"{name}_config", {})
        self.start_time = time.time()
        self.last_activity = self.start_time
        
        logger.info(f"Initialized {self.name} agent with ID: {self.agent_id}")
        
        self.chain = None
        self._setup_tracing()
        
    def _setup_tracing(self):
        """Set up LangSmith tracing for the agent if enabled."""
        self.langsmith_config = get_langsmith_config()
        self.tracer = get_langchain_tracer() if self.langsmith_config.tracing_enabled else None
        
    def get_trace_config(self) -> Optional[Dict[str, Any]]:
        """
        Get the trace configuration for LangChain runnables.
        
        Returns:
            A dictionary with trace configuration if tracing is enabled, otherwise None
        """
        if not self.langsmith_config.tracing_enabled or not self.tracer:
            return None
            
        # Create tags as simple strings in format "key=value"
        tags_dict = create_langsmith_tags(agent_name=self.name)
        tags = [f"{k}={v}" for k, v in tags_dict.items()]
            
        return {
            "tags": tags,
            "callbacks": [self.tracer]
        }
    
    def trace_chain(self, chain: RunnableSequence) -> RunnableSequence:
        """
        Add tracing to a LangChain runnable chain.
        
        Args:
            chain: The LangChain runnable to trace
            
        Returns:
            The same chain with tracing configured if enabled
        """
        if self.langsmith_config.tracing_enabled and self.tracer:
            logger.debug(f"Adding LangSmith tracing to {self.name} chain")
            return chain.with_config(self.get_trace_config())
        return chain
    
    async def run(self, state: WorkflowState) -> WorkflowState:
        """
        Run the agent on the current workflow state.
        
        This method must be implemented by all agent subclasses.
        
        Args:
            state: The current workflow state
            
        Returns:
            Updated workflow state after agent processing
        """
        raise NotImplementedError("Subclasses must implement run().")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the agent.
        
        Returns:
            Status information
        """
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "uptime": time.time() - self.start_time,
            "last_activity": self.last_activity,
            "status": "active"
        }
    
    def log_activity(self, activity_type: str, details: Dict[str, Any] = None) -> None:
        """
        Log an agent activity.
        
        Args:
            activity_type: Type of activity
            details: Additional activity details
        """
        self.last_activity = time.time()
        
        log_entry = {
            "agent_id": self.agent_id,
            "name": self.name,
            "activity_type": activity_type,
            "timestamp": self.last_activity
        }
        
        if details:
            log_entry["details"] = details
        
        logger.info(f"Agent activity: {json.dumps(log_entry)}")
    
    def validate_config(self) -> bool:
        """
        Validate the agent configuration.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        # Base implementation assumes config is valid
        # Specialized agents should override this
        return True 