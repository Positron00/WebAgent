"""
Web Research Agent for the WebAgent backend.

This agent performs web searches and extracts relevant information from web pages.
"""
from typing import Dict, List, Any
import logging
import json

from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

from app.models.task import WorkflowState
from app.services.llm import get_llm
from app.services.web_search import perform_web_search
from app.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

# Prompt for the Web Research Agent
WEB_RESEARCH_PROMPT = """You are a Web Research Agent specializing in finding information online.

USER QUERY: {query}
RESEARCH PLAN: {research_plan}

You have been provided with the following search results:
{search_results}

Your task:
1. Analyze the search results and extract relevant information for the query
2. Organize the information in a clear, structured format
3. Cite your sources with URLs
4. Only include information that is directly related to the query
5. Highlight key facts, statistics, and quotes
6. Format your findings as markdown

IMPORTANT: Be thorough, accurate, and objective. Cite all sources.
"""

class WebResearchAgent(BaseAgent):
    """
    Web Research Agent that searches the internet for information.
    """
    
    def __init__(self):
        """Initialize the Web Research Agent."""
        super().__init__(agent_name="web_research")
        self.prompt = ChatPromptTemplate.from_template(WEB_RESEARCH_PROMPT)
        self.llm = get_llm("gpt-4-turbo")
        self.parser = StrOutputParser()
        self.chain = self.prompt | self.llm | self.parser
        self.chain = self.trace_chain(self.chain)
    
    def _format_search_results(self, results: List[Dict[str, Any]]) -> str:
        """Format search results for the prompt."""
        if not results:
            return "No relevant search results found."
            
        formatted_results = ""
        for i, result in enumerate(results, 1):
            formatted_results += f"SOURCE {i}:\n"
            formatted_results += f"Title: {result.get('title', 'No title')}\n"
            formatted_results += f"URL: {result.get('url', 'No URL')}\n"
            formatted_results += f"Content: {result.get('content', 'No content')}\n\n"
            
        return formatted_results
        
    async def run(self, state: WorkflowState) -> WorkflowState:
        """
        Execute the Web Research Agent to search and analyze information.
        
        Args:
            state: The current workflow state
            
        Returns:
            Updated workflow state with web research results
        """
        logger.info(f"Web Research Agent running for query: {state.query}")
        state.current_step = "web_research"
        
        try:
            # Get the research plan
            research_plan = state.context.get("research_plan", {})
            
            # If web search is not required, skip this step
            if not research_plan.get("requires_web_search", True):
                logger.info("Web search not required by research plan. Skipping.")
                state.update_with_agent_output("web_research", {
                    "status": "skipped",
                    "reason": "Web search not required by research plan",
                    "timestamp": datetime.now().isoformat()
                })
                return state
            
            # Perform the web search
            logger.info("Performing web search")
            search_results = await perform_web_search(
                query=state.query,
                search_depth="comprehensive",
                max_results=7
            )
            
            if not search_results:
                logger.warning("No search results found.")
                state.update_with_agent_output("web_research", {
                    "status": "completed",
                    "results": [],
                    "findings": "No relevant information found through web search.",
                    "timestamp": datetime.now().isoformat()
                })
                return state
            
            # Format search results for the prompt
            formatted_results = self._format_search_results(search_results)
            
            # Run the LLM chain to analyze the search results
            analysis = await self.chain.ainvoke({
                "query": state.query,
                "research_plan": research_plan.get("analysis", "No research plan provided."),
                "search_results": formatted_results
            })
            
            # Create the research report
            web_research_report = {
                "status": "completed",
                "results": search_results,
                "findings": analysis,
                "timestamp": datetime.now().isoformat()
            }
            
            # Update the state with the web research report
            state.update_with_agent_output("web_research", web_research_report)
            
            # Add the findings to the context for other agents to use
            state.context["web_research_findings"] = analysis
            
            logger.info(f"Web Research completed with {len(search_results)} results.")
            return state
            
        except Exception as e:
            logger.error(f"Error in Web Research Agent: {str(e)}")
            state.mark_error(f"Web Research Agent failed: {str(e)}")
            return state


def get_web_research_agent() -> WebResearchAgent:
    """
    Get an instance of the Web Research Agent.
    
    Returns:
        A WebResearchAgent instance
    """
    return WebResearchAgent() 