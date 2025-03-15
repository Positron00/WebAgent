"""
Supervisor Agent
===============

The Supervisor Agent is responsible for:
1. Analyzing the user query
2. Breaking down complex queries into manageable tasks
3. Planning the research workflow
4. Assigning tasks to specialized agents
5. Orchestrating the workflow between specialized agents
6. Processing document extraction requests
7. Routing outputs to appropriate downstream agents

Recent Updates:
- Combined supervisor implementations for LangGraph integration
- Added direct integration with Document Extraction Agent
- Added ability to route document extraction results directly to Team Manager Agent
- Improved handling of different request types
"""
from typing import Dict, List, Any, Optional
import logging
import time
import json
import os
from datetime import datetime
from pathlib import Path

from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

# Import local modules
from app.core.config import settings
from app.agents.base_agent import BaseAgent
from app.agents.document_extraction_agent import DocumentExtractionAgent
from app.services.llm import get_llm
from app.utils.metrics import timing_decorator, log_memory_usage
from app.models.task import WorkflowState

# Initialize logging
logger = logging.getLogger(__name__)

# Prompt for the Supervisor Agent research planning
SUPERVISOR_PROMPT = """You are the Supervisor Agent in a multi-agent research system.
Your job is to analyze the user's query, break it down into research tasks, and create a research plan.

USER QUERY: {query}

Your task:
1. Analyze the query to understand what the user is asking
2. Determine if web search is needed or if internal knowledge is sufficient
3. Create a research plan with specific questions to investigate
4. Decide which specialized agents should handle each part of the research
5. Determine if document extraction is needed for any referenced documents

Output your analysis and plan in a clear, structured format.
Include whether document extraction is needed under the "requires_document_extraction" key.
"""

class SupervisorAgent(BaseAgent):
    """
    Supervisor Agent that plans and orchestrates the research workflow.
    This agent serves both as a LangGraph node and as a direct service for document extraction.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the Supervisor Agent.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(name="supervisor", config=config)
        
        # Set up LLM and chain for research planning
        self.prompt = ChatPromptTemplate.from_template(SUPERVISOR_PROMPT)
        self.llm = get_llm()
        self.parser = StrOutputParser()
        self.chain = self.prompt | self.llm | self.parser
        self.chain = self.trace_chain(self.chain)
        
        # Initialize specialized agents
        self.document_extraction_agent = None  # Lazy-loaded when needed
        
        logger.info("Supervisor Agent initialized with both LangGraph and direct service capabilities")
    
    # Make the agent callable for LangGraph compatibility
    def __call__(self, state: WorkflowState) -> WorkflowState:
        """
        Make the agent compatible with LangGraph by providing a __call__ method.
        
        Args:
            state: The current workflow state
            
        Returns:
            Updated workflow state
        """
        # This is a synchronous wrapper around the async run method
        import asyncio
        try:
            # Use a new event loop if needed
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Run the async method and return the result
            return loop.run_until_complete(self.run(state))
        except Exception as e:
            logger.error(f"Error in SupervisorAgent.__call__: {e}")
            # Handle the error appropriately and return updated state
            state.error = str(e)
            state.status = "error"
            return state
    
    async def run(self, state: WorkflowState) -> WorkflowState:
        """
        Execute the Supervisor Agent to analyze the query and create a research plan.
        This method is used by LangGraph for workflow orchestration.
        
        Args:
            state: The current workflow state
            
        Returns:
            Updated workflow state with research plan
        """
        logger.info(f"Supervisor Agent analyzing query: {state.query}")
        state.current_step = "supervisor"
        
        try:
            # Run the LLM chain to analyze the query
            result = await self.chain.ainvoke({"query": state.query})
            
            # Extract key information from the analysis
            requires_web_search = "web" in result.lower() or "online" in result.lower() or "internet" in result.lower()
            requires_internal_knowledge = "internal" in result.lower() or "existing" in result.lower() or "database" in result.lower()
            requires_document_extraction = "document" in result.lower() or "extract" in result.lower() or "pdf" in result.lower()
            
            # Create a research plan
            research_plan = {
                "analysis": result,
                "requires_web_search": requires_web_search,
                "requires_internal_knowledge": requires_internal_knowledge,
                "requires_document_extraction": requires_document_extraction,
                "timestamp": datetime.now().isoformat()
            }
            
            # Update the state with the research plan
            state.update_with_agent_output("supervisor", research_plan)
            
            # Add the analysis to the context for other agents to use
            state.context["research_plan"] = research_plan
            
            # Handle document extraction directly if needed
            if requires_document_extraction and "document_path" in state.context:
                logger.info("Document extraction required - initiating extraction")
                document_result = self.handle_document_extraction({
                    "document_path": state.context["document_path"],
                    "extraction_method": "auto",
                    "route_to_team_manager": True
                })
                state.context["document_extraction_result"] = document_result
            
            logger.info(f"Supervisor completed analysis. Web search: {requires_web_search}, " 
                      f"Internal knowledge: {requires_internal_knowledge}, "
                      f"Document extraction: {requires_document_extraction}")
            return state
            
        except Exception as e:
            logger.error(f"Error in Supervisor Agent: {str(e)}")
            state.mark_error(f"Supervisor Agent failed: {str(e)}")
            return state
    
    @timing_decorator
    @log_memory_usage
    def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an incoming request and route to appropriate specialized agents.
        This method is used for direct service invocation outside of LangGraph.
        
        Args:
            request: Request data
            
        Returns:
            Response with processed data
        """
        start_time = time.time()
        request_type = request.get("request_type", "unknown")
        
        logger.info(f"Processing {request_type} request")
        
        try:
            # Route request based on type
            if request_type == "document_extraction":
                return self.handle_document_extraction(request)
            elif request_type == "team_management":
                return self.handle_team_management(request)
            elif request_type == "research":
                return self.handle_research_request(request)
            # Add other request types as needed
            else:
                return {
                    "success": False,
                    "error": f"Unknown request type: {request_type}",
                    "processing_time": time.time() - start_time
                }
        except Exception as e:
            logger.error(f"Error processing request: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": f"Request processing error: {str(e)}",
                "processing_time": time.time() - start_time
            }
    
    def handle_document_extraction(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle document extraction requests.
        
        Args:
            request: Document extraction request
            
        Returns:
            Extraction results
        """
        start_time = time.time()
        
        # Validate request
        if "document_path" not in request:
            return {
                "success": False,
                "error": "Missing required field: document_path",
                "processing_time": time.time() - start_time
            }
        
        document_path = request.get("document_path")
        extraction_method = request.get("extraction_method", "auto")
        extraction_params = request.get("extraction_params", {})
        route_to_team_manager = request.get("route_to_team_manager", False)
        
        # Initialize document extraction agent if not already done
        if self.document_extraction_agent is None:
            logger.info("Initializing Document Extraction Agent")
            self.document_extraction_agent = DocumentExtractionAgent(config=self.config)
        
        # Process the document
        result = self.document_extraction_agent.process_document(
            document_path=document_path,
            method=extraction_method,
            extraction_params=extraction_params,
            route_to_team_manager=route_to_team_manager
        )
        
        # Add supervisor metadata
        result["supervisor_metadata"] = {
            "request_time": start_time,
            "processing_time": time.time() - start_time,
            "routed_to_team_manager": route_to_team_manager
        }
        
        return result
    
    def handle_team_management(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle team management requests.
        
        Args:
            request: Team management request
            
        Returns:
            Team management results
        """
        # This would be implemented to handle team management requests
        # For now, just return a placeholder
        start_time = time.time()
        return {
            "success": False,
            "error": "Team management not implemented yet",
            "processing_time": time.time() - start_time
        }
    
    def handle_research_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle research requests outside of LangGraph.
        
        Args:
            request: Research request
            
        Returns:
            Research results
        """
        start_time = time.time()
        
        # Validate request
        if "query" not in request:
            return {
                "success": False,
                "error": "Missing required field: query",
                "processing_time": time.time() - start_time
            }
        
        # In a real implementation, this would call the LangGraph workflow
        # For now, just return a placeholder
        query = request.get("query")
        return {
            "success": True,
            "query": query,
            "message": "Research request acknowledged but must be processed through LangGraph workflow",
            "processing_time": time.time() - start_time
        }
    
    @timing_decorator
    def get_agent_status(self) -> Dict[str, Any]:
        """
        Get the status of all managed agents.
        
        Returns:
            Status information for all agents
        """
        status = {
            "supervisor": {
                "status": "active",
                "timestamp": time.time()
            },
            "document_extraction": {
                "status": "initialized" if self.document_extraction_agent else "not_initialized",
                "timestamp": time.time()
            }
            # Add other agents as they are implemented
        }
        
        return status


def get_supervisor_agent() -> SupervisorAgent:
    """
    Get an instance of the Supervisor Agent.
    
    Returns:
        A SupervisorAgent instance
    """
    return SupervisorAgent() 