"""
Base Agent class for the WebAgent backend.

This module provides a base class for all agents in the system, handling common functionality
like LangSmith tracing and logging.
"""
from typing import Dict, Any, Optional
import logging

from langchain.schema.runnable import RunnableSequence
from langchain.schema.runnable import RunnableConfig

from app.core.setup_langsmith import get_langchain_tracer, create_langsmith_tags
from app.core.loadEnvYAML import get_langsmith_config
from app.models.task import WorkflowState

logger = logging.getLogger(__name__)

class BaseAgent:
    """
    Base class for all agents in the WebAgent system.
    Provides common functionality for tracing, logging, and state management.
    """
    
    def __init__(self, agent_name: str):
        """
        Initialize the base agent.
        
        Args:
            agent_name: The name of the agent for tracing and logging
        """
        self.agent_name = agent_name
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
        tags_dict = create_langsmith_tags(agent_name=self.agent_name)
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
            logger.debug(f"Adding LangSmith tracing to {self.agent_name} chain")
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