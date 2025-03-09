"""
Supervisor Agent for the WebAgent backend.

The Supervisor Agent is responsible for:
1. Analyzing the user query
2. Breaking down complex queries into manageable tasks
3. Planning the research workflow
4. Assigning tasks to specialized agents
5. Synthesizing final results
"""
from typing import Dict, List, Any
import logging
from datetime import datetime

from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

from app.models.task import WorkflowState
from app.services.llm import get_llm
from app.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

# Prompt for the Supervisor Agent
SUPERVISOR_PROMPT = """You are the Supervisor Agent in a multi-agent research system.
Your job is to analyze the user's query, break it down into research tasks, and create a research plan.

USER QUERY: {query}

Your task:
1. Analyze the query to understand what the user is asking
2. Determine if web search is needed or if internal knowledge is sufficient
3. Create a research plan with specific questions to investigate
4. Decide which specialized agents should handle each part of the research

Output your analysis and plan in a clear, structured format.
"""

class SupervisorAgent(BaseAgent):
    """
    Supervisor Agent that plans and orchestrates the research workflow.
    """
    
    def __init__(self):
        """Initialize the Supervisor Agent."""
        super().__init__(agent_name="supervisor")
        self.prompt = ChatPromptTemplate.from_template(SUPERVISOR_PROMPT)
        self.llm = get_llm("gpt-4-turbo")
        self.parser = StrOutputParser()
        self.chain = self.prompt | self.llm | self.parser
        self.chain = self.trace_chain(self.chain)
        
    async def run(self, state: WorkflowState) -> WorkflowState:
        """
        Execute the Supervisor Agent to analyze the query and create a research plan.
        
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
            
            # Create a research plan
            research_plan = {
                "analysis": result,
                "requires_web_search": requires_web_search,
                "requires_internal_knowledge": requires_internal_knowledge,
                "timestamp": datetime.now().isoformat()
            }
            
            # Update the state with the research plan
            state.update_with_agent_output("supervisor", research_plan)
            
            # Add the analysis to the context for other agents to use
            state.context["research_plan"] = research_plan
            
            logger.info(f"Supervisor completed analysis. Web search: {requires_web_search}, Internal knowledge: {requires_internal_knowledge}")
            return state
            
        except Exception as e:
            logger.error(f"Error in Supervisor Agent: {str(e)}")
            state.mark_error(f"Supervisor Agent failed: {str(e)}")
            return state


def get_supervisor_agent() -> SupervisorAgent:
    """
    Get an instance of the Supervisor Agent.
    
    Returns:
        A SupervisorAgent instance
    """
    return SupervisorAgent() 