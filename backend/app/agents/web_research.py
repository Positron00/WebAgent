"""
Web Research Agent for the WebAgent backend.

This agent performs web searches and extracts relevant information from web pages.
"""
from typing import Dict, List, Any
import logging
import json
from datetime import datetime

from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

from app.models.task import WorkflowState
from app.services.llm import get_llm
from app.services.web_search import perform_web_search
from app.agents.base_agent import BaseAgent
from app.core.config import settings
logger = logging.getLogger(__name__)

# Prompt for the Web Research Agent
WEB_RESEARCH_PROMPT = """You are a Web Research Agent specializing in finding information online.

USER QUERY: {query}
RESEARCH PLAN: {research_plan}

You have been provided with the following search results:
{search_results}

{feedback_section}

Your task:
1. Analyze the search results and extract relevant information for the query
2. Organize the information in a clear, structured format
3. Cite your sources with URLs
4. Only include information that is directly related to the query
5. Highlight key facts, statistics, and quotes
6. Format your findings as markdown
7. If additional research was requested, focus on addressing the specific questions and missing information

IMPORTANT: Be thorough, accurate, and objective. Cite all sources.
"""

# Prompt for follow-up research
FOLLOWUP_RESEARCH_PROMPT = """
ADDITIONAL RESEARCH REQUESTED - ITERATION {iteration}

Previous research was rated {score}/10.

Missing information:
{missing_information}

Specific research questions to address:
{research_questions}

Please focus on finding this information in your analysis.
"""

class WebResearchAgent(BaseAgent):
    """
    Web Research Agent that searches the internet for information.
    """
    
    def __init__(self):
        """Initialize the Web Research Agent."""
        super().__init__(name="web_research")
        self.prompt = WEB_RESEARCH_PROMPT
        self.llm = get_llm(settings.LLM_DEFAULT_MODEL)
        self.output_parser = StrOutputParser()
        self.llm_chain = self.llm
    
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
    
    def _format_feedback(self, feedback: Dict) -> str:
        """Format feedback from Senior Research Agent if available."""
        # Check if there's feedback in the context
        if not feedback:
            return ""
            
        iteration = feedback.get("iteration", 1)
        if iteration <= 1:
            return ""
            
        # Format the feedback section
        score = feedback.get("score", 0)
        missing_info = feedback.get("missing_information", [])
        questions = feedback.get("research_questions", [])
        
        # Format missing information as bullet points
        missing_info_str = "\n".join([f"- {item}" for item in missing_info])
        
        # Format research questions as bullet points
        questions_str = "\n".join([f"- {q}" for q in questions])
        
        return FOLLOWUP_RESEARCH_PROMPT.format(
            iteration=iteration,
            score=score,
            missing_information=missing_info_str,
            research_questions=questions_str
        )
        
    async def run(self, state: WorkflowState) -> WorkflowState:
        """Execute the Web Research Agent to search and analyze information."""
        query = state.context.get("query", "")
        logger.info(f"Processing query: {query}")

        # Update workflow state with current step
        state.current_step = "web_research"

        # Get the research plan from the state context
        research_plan = state.context.get("research_plan", {})
        
        # Get the research iteration from the context
        iteration = state.context.get("research_iteration", 1)
        is_followup = iteration > 1
        
        # Check if the research plan doesn't require a web search
        if not research_plan.get("requires_web_search", True):
            logger.info("Research plan does not require a web search")
            state.context["web_search_results"] = []
            state.context["web_research_results"] = {
                "skipped": True,
                "reasoning": "The research question doesn't require a web search."
            }
            return state
        
        # Use a different search query for follow-up research
        search_query = query
        if is_followup:
            feedback = state.context.get("research_feedback", {})
            
            # Use a specific question from feedback if available
            questions = feedback.get("research_questions", [])
            if questions and isinstance(questions, list) and len(questions) > 0:
                search_query = questions[0]
                logger.info(f"Using feedback question for search: {search_query}")
            else:
                search_query = feedback.get("search_query", query)
                logger.info(f"Using feedback search query: {search_query}")
                
        # Perform the web search
        logger.info(f"Performing web search (iteration {iteration})")
        try:
            # Map the search depth to valid values
            depth = "basic"  # Default to basic for reliability
            if research_plan.get("search_depth") == "deep":
                depth = "comprehensive"
            
            logger.debug(f"Using search depth: {depth}")
            
            search_results = await perform_web_search(
                query=search_query,
                search_depth=depth,
                max_results=5  # Reduced from 7 to avoid overloading
            )
            
            # If we couldn't get results, try a simplified query
            if not search_results:
                logger.warning(f"No results for query: '{search_query}'. Trying simplified query.")
                # Extract just main keywords for a simpler query
                simplified_query = " ".join(search_query.split()[:6])
                logger.info(f"Simplified query: {simplified_query}")
                
                search_results = await perform_web_search(
                    query=simplified_query,
                    search_depth="basic",  # Always use basic for fallback
                    max_results=3
                )
        except Exception as search_error:
            logger.error(f"Error during web search: {str(search_error)}")
            # Create empty results to continue with
            search_results = []
        
        # If follow-up research, get additional results for other questions
        if is_followup and search_results:  # Only try additional if first search worked
            feedback = state.context.get("research_feedback", {})
            questions = feedback.get("research_questions", [])[1:2]  # Get next question if there are more
            
            for question in questions:
                logger.info(f"Performing additional search for: {question}")
                try:
                    additional_results = await perform_web_search(
                        query=question,
                        search_depth="basic",  # Changed from "focused" to valid "basic"
                        max_results=3
                    )
                    search_results.extend(additional_results)
                except Exception as additional_error:
                    logger.error(f"Error during additional search: {str(additional_error)}")
        
        # Handle the case where no search results are found
        if not search_results:
            logger.warning("No relevant search results found")
            state.context["web_search_results"] = []
            state.context["web_research_results"] = {
                "analysis": "No relevant information found on the web for this query.",
                "sources": []
            }
            return state
        
        # Format the search results for the prompt
        formatted_results = self._format_search_results(search_results)
        
        # Prepare the prompt with the research plan and results
        messages = [
            {"role": "system", "content": self.prompt},
            {"role": "user", "content": f"""
# Research Question
{query}

# Research Plan
{research_plan.get("plan", "No specific research plan provided.")}

# Web Search Results
{formatted_results}

{self._format_feedback(state.context.get("research_feedback", {}))}
"""}
        ]
        
        # Call the language model to analyze the results
        try:
            # Call the LLM directly
            response = self.llm.invoke(messages)
            assistant_message = response if isinstance(response, str) else response.get("content", "")
            
            # Log response for debugging
            logger.debug(f"LLM response type: {type(response)}")
            if not assistant_message:
                logger.warning("Empty response from LLM")
                assistant_message = "No analysis was generated. Please review the sources directly."
            
            # Parse the response
            parsed_response = {
                "analysis": assistant_message,
                "sources": []
            }
            
            # Add the sources from the search results
            sources = []
            for result in search_results:
                sources.append({
                    "title": result.get("title", "Unknown title"),
                    "url": result.get("url", ""),
                    "content_snippet": result.get("content", "")[:150] + "..." if result.get("content") else ""
                })
            
            parsed_response["sources"] = sources
            
            # Log the final response for debugging
            logger.debug(f"Final response structure: {list(parsed_response.keys())}")
            logger.debug(f"Number of sources: {len(sources)}")
            
            # Update the workflow state with the research results
            state.context["web_search_results"] = search_results
            state.context["web_research_results"] = parsed_response
            return state
            
        except Exception as e:
            logger.error(f"Error in web research analysis: {str(e)}")
            # Provide a basic fallback response with the sources in case of error
            sources = []
            for result in search_results:
                sources.append({
                    "title": result.get("title", "Unknown title"),
                    "url": result.get("url", ""),
                    "content_snippet": result.get("content", "")[:150] + "..." if result.get("content") else ""
                })
                
            state.context["web_search_results"] = search_results
            state.context["web_research_results"] = {
                "analysis": "Error occurred during analysis. Please review the sources directly.",
                "sources": sources
            }
            return state


def get_web_research_agent():
    """
    Get an instance of the Web Research Agent.
    
    Returns:
        A callable function that takes a WorkflowState and calls the agent's run method.
        This makes it compatible with LangGraph which expects nodes to be callable.
    """
    agent = WebResearchAgent()
    
    # Return a callable function instead of the agent directly
    async def web_research_function(state: WorkflowState) -> WorkflowState:
        """Callable wrapper for the web research agent"""
        return await agent.run(state)
    
    return web_research_function 