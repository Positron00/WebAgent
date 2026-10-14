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
from app.core.log_context import describe_text
from app.agents.base_agent import BaseAgent
from app.agents.document_extraction_agent import DocumentExtractionAgent
from app.services.llm import get_agent_llm
from app.utils.metrics import timing_decorator, log_memory_usage
from app.models.task import WorkflowState
from app.models.research_plan import normalize_research_plan, without_missing_document
from app.utils.json_utils import extract_json_object
from app.utils.reasoning import strip_reasoning
from app.security.guarded_llm import register_template
from app.security.hop_guard import acheck_hop, task_items

# Initialize logging
logger = logging.getLogger(__name__)

# Prompt for the Supervisor Agent research planning
SUPERVISOR_PROMPT = """You are the Supervisor Agent in a multi-agent research system.
Your job is to analyze the user's query, break it down into research tasks, and create a research plan.

USER QUERY: {query}

Your task:
1. Analyze the query to understand what the user is asking
2. Determine the task type:
   - "document_extraction" for simple information extraction from documents
   - "complex_research" for queries requiring detailed analysis, web research, or multiple steps
3. Create specific requirements for completing the query
4. If complex research is needed, create a detailed research plan with specific questions to investigate
5. Determine which specialized agents should handle each part of the research

Output your analysis in JSON format with these fields:
{{
  "task_type": "document_extraction" or "complex_research",
  "requirements": [list of specific requirements needed to complete the task],
  "research_plan": {{
    "requires_web_search": true/false,
    "requires_internal_knowledge": true/false,
    "requires_document_extraction": true/false,
    "questions": [list of specific questions to research],
    "standalone_query": "the current query rewritten so it makes sense without the conversation (copy it unchanged if it already does)"
  }}
}}
"""

# Direct answers (no research). The model is a text-completion model, so the prompt ends
# where the answer should start; without that it continues the question instead.
DIRECT_ANSWER_PROMPT = """You are a helpful assistant. Answer the user's question directly and concisely.
Use the conversation for context. Do not show your reasoning or notes.

{conversation}

Answer:"""

# Static prompt text is never scored by the safety net (SEC-032).
register_template("supervisor", SUPERVISOR_PROMPT)
register_template("supervisor", DIRECT_ANSWER_PROMPT)

IMAGE_NOT_AVAILABLE_NOTE = (
    "Note: the user attached an image, but you cannot see it. Do not describe or guess its "
    "content; if the question depends on the image, say you can't view images here."
)



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
        self.llm = get_agent_llm("supervisor")
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
    
    async def answer_directly(
        self, query: str, previous_messages: Optional[list] = None, image_attached: bool = False
    ) -> str:
        """Text-only direct answer (no LangGraph), using conversation history when present."""
        state = WorkflowState(
            query=query,
            context={"previous_messages": previous_messages or []},
        )
        conversation = self._with_history(state)
        if image_attached:
            # The model never receives the image; without this note it describes one anyway.
            conversation += f"\n\n{IMAGE_NOT_AVAILABLE_NOTE}"
        prompt = DIRECT_ANSWER_PROMPT.format(conversation=conversation)
        await acheck_hop("supervisor", task_items(state), state)
        direct = await self.llm.ainvoke(prompt)
        text = direct if isinstance(direct, str) else getattr(direct, "content", str(direct))
        # The text model continues the prompt and may think out loud before answering.
        return strip_reasoning(text)

    @staticmethod
    def _with_history(state: WorkflowState) -> str:
        """Prefix the query with recent chat turns so follow-up questions keep their meaning."""
        previous = state.context.get("previous_messages", [])
        if not previous:
            return state.query
        history = "\n".join(
            f"{m.get('role', 'user')}: {m.get('content', '')}" for m in previous[-8:]
        )
        return f"Conversation so far:\n{history}\n\nCurrent query: {state.query}"

    async def run(self, state: WorkflowState) -> WorkflowState:
        """
        Execute the Supervisor Agent to analyze the query and create a research plan.
        This method is used by LangGraph for workflow orchestration.
        
        Args:
            state: The current workflow state
            
        Returns:
            Updated workflow state with research plan
        """
        logger.info(f"Supervisor Agent analyzing query ({describe_text(state.query)})")
        if settings.WEBAGENT_ENV == "dev":
            logger.debug("Query: %s", state.query)
        state.current_step = "supervisor"

        if not state.context.get("agentic_toggle", False):
            try:
                content = await self.answer_directly(
                    state.query, state.context.get("previous_messages", [])
                )
                state.mark_completed(
                    {
                        "title": "Response",
                        "content": content,
                        "type": "direct_response",
                    }
                )
                return state
            except Exception as e:
                logger.error(f"Direct response failed: {e}")
                state.mark_error(f"Supervisor Agent failed: {str(e)}")
                return state
        
        try:
            state.add_log("Supervisor: Analyzing query and planning research")
            await acheck_hop("supervisor", task_items(state), state)
            result = await self.chain.ainvoke({"query": self._with_history(state)})
            
            plan_data = result if isinstance(result, dict) else extract_json_object(str(result))
            if plan_data is None:
                logger.warning("Could not parse supervisor output as JSON; inferring the plan from keywords")
                logger.debug(f"Raw supervisor output: {result}")
                text = str(result).lower()
                plan_data = {
                    "task_type": "document_extraction" if "document" in text else "complex_research",
                    "requirements": ["Extract relevant information from the query"],
                    "research_plan": {
                        "needs_web_research": "web" in text or "online" in text,
                        "needs_internal_research": "internal" in text or "existing" in text,
                        "questions": ["What information is needed to answer the query?"],
                    },
                }
            
            # Update the state with the supervisor's analysis
            state.update_with_agent_output("supervisor", plan_data)
            
            plan = normalize_research_plan(plan_data.get("research_plan", {}))
            has_document = bool(state.context.get("document_id"))
            if plan.requires_document_extraction and not has_document:
                logger.info("Plan asked for document extraction, but no document was attached; researching instead")
            plan = without_missing_document(plan, has_document)
            state.context["research_plan"] = plan.model_dump()
            state.add_log(
                "Supervisor: Plan ready - "
                f"web search: {plan.requires_web_search}, "
                f"internal knowledge: {plan.requires_internal_knowledge}, "
                f"document extraction: {plan.requires_document_extraction}"
            )
            
            logger.info(f"Supervisor completed analysis. Task type: {plan_data.get('task_type', 'undefined')}")
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


def get_supervisor_agent():
    """
    Get an instance of the Supervisor Agent.
    
    Returns:
        A callable function that takes a WorkflowState and calls the agent's run method.
        This makes it compatible with LangGraph which expects nodes to be callable.
    """
    agent = SupervisorAgent()
    
    # Return a callable function instead of the agent directly
    async def supervisor_function(state: WorkflowState) -> WorkflowState:
        """Callable wrapper for the supervisor agent"""
        return await agent.run(state)
    
    return supervisor_function 