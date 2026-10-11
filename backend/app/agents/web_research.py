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
from app.models.research_plan import research_query
from app.services.llm import get_agent_llm
from app.services.web_search import perform_web_search
from app.agents.base_agent import BaseAgent
from app.core.config import settings
from app.utils.reasoning import strip_reasoning
logger = logging.getLogger(__name__)

# Prompt for the Web Research Agent. Filled with str.format() and sent as one completion
# prompt; it ends with "ANALYSIS:" so the model writes the answer there and stops.
WEB_RESEARCH_PROMPT = """You are a Web Research Agent. Answer the research question using only the search results below.

RESEARCH QUESTION: {query}

RESEARCH PLAN:
{research_plan}

SEARCH RESULTS (numbered; cite them as [1], [2], ... using these numbers):
{search_results}
{feedback_section}
Instructions:
1. Write ONE comprehensive analysis in markdown that directly answers the research question.
2. Use only facts found in the search results above. Do not invent sources, URLs, names, dates or numbers.
3. Cite sources inline with their numbers in square brackets, e.g. [1] or [2][3].
4. Use clear headings and bullet points. Do not repeat information.
5. If additional research was requested, answer those questions first.
6. Do not show your reasoning, plans or notes, and do not list the sources at the end; they are added automatically.

Write the analysis now and stop when it is complete.

ANALYSIS:
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
        self.llm = get_agent_llm("web_research")
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
    
    def _format_feedback(self, feedback: Dict, iteration: int) -> str:
        """Format feedback from Senior Research Agent if available."""
        if not feedback or iteration <= 1:
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
        # A follow-up like "what about Europe?" is searched as the supervisor's standalone rewrite.
        query = research_query(state)
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
            skipped = {
                "skipped": True,
                "reasoning": "The research question doesn't require a web search."
            }
            state.context["web_search_results"] = []
            state.context["web_research_results"] = skipped
            state.update_with_agent_output("web_research", skipped)
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
                depth = "advanced"
            elif research_plan.get("search_depth") == "standard":
                depth = "basic"  # Map standard to basic
            elif research_plan.get("search_depth") == "quick":
                depth = "basic"  # Map quick to basic as well
            
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
            empty = {
                "analysis": "No relevant information found on the web for this query.",
                "sources": []
            }
            state.context["web_search_results"] = []
            state.context["web_research_results"] = empty
            state.update_with_agent_output("web_research", empty)
            return state
        
        # Format the search results for the prompt
        formatted_results = self._format_search_results(search_results)
        
        # Prepare the prompt with the research plan and results
        prompt_text = self.prompt.format(
            query=query,
            research_plan="\n".join(f"- {q}" for q in research_plan.get("questions", []))
            or "No specific research plan provided.",
            search_results=formatted_results,
            feedback_section=self._format_feedback(state.context.get("research_feedback", {}), iteration),
        )
        
        # Call the language model to analyze the results
        try:
            # Call the LLM directly
            response = await self.llm.ainvoke(prompt_text)
            assistant_message = (
                response if isinstance(response, str) else getattr(response, "content", "")
            )
            
            # Log response for debugging
            logger.debug(f"LLM response type: {type(response)}")
            if not assistant_message:
                logger.warning("Empty response from LLM")
                assistant_message = "No analysis was generated. Please review the sources directly."
            
            # Clean up the response to remove thinking process and duplications
            assistant_message = self._clean_analysis(assistant_message)
            
            # Parse the response
            parsed_response = {
                "analysis": assistant_message,
                "sources": []
            }
            
            # Add the sources from the search results
            sources = []
            seen_urls = set()  # Track URLs we've already processed
            
            for result in search_results:
                url = result.get("url", "")
                
                # Skip this source if we've already seen this URL
                if url and url in seen_urls:
                    continue
                    
                # Add URL to our tracking set
                if url:
                    seen_urls.add(url)
                
                sources.append({
                    "title": result.get("title", "Unknown title"),
                    "url": url,
                    "content_snippet": result.get("content", "")[:150] + "..." if result.get("content") else ""
                })
            
            parsed_response["sources"] = sources
            
            # Log the final response for debugging
            logger.debug(f"Final response structure: {list(parsed_response.keys())}")
            logger.debug(f"Number of sources: {len(sources)}")
            
            state.context["web_search_results"] = search_results
            state.context["web_research_results"] = parsed_response
            state.update_with_agent_output("web_research", parsed_response)
            return state
            
        except Exception as e:
            # Log detailed error information
            logger.error(f"Error during LLM analysis: {str(e)}")
            
            # Create a more helpful fallback analysis based on source titles
            fallback_analysis = self._create_fallback_analysis(search_results, query)
            
            # Process sources with deduplication
            sources = []
            seen_urls = set()  # Track URLs we've already processed
            
            for result in search_results:
                url = result.get("url", "")
                
                # Skip this source if we've already seen this URL
                if url and url in seen_urls:
                    continue
                    
                # Add URL to our tracking set
                if url:
                    seen_urls.add(url)
                
                sources.append({
                    "title": result.get("title", "Unknown title"),
                    "url": url,
                    "content_snippet": result.get("content", "")[:150] + "..." if result.get("content") else ""
                })
            
            # Create response with fallback analysis
            parsed_response = {
                "analysis": fallback_analysis,
                "sources": sources
            }
            
            state.context["web_search_results"] = search_results
            state.context["web_research_results"] = parsed_response
            state.update_with_agent_output("web_research", parsed_response)
            return state

    def _clean_analysis(self, analysis: str) -> str:
        """Clean up the analysis to remove thinking steps and repetitions."""
        # Some models write their reasoning first and then the answer after a channel marker.
        analysis = strip_reasoning(analysis)
        # Remove lines that indicate step-by-step thinking
        lines = analysis.split('\n')
        cleaned_lines = []
        skip_line = False
        in_sources_section = False
        
        for line in lines:
            # Check if we're entering a sources section
            if line.strip().lower().startswith(("sources:", "# sources", "## sources")):
                in_sources_section = True
                continue
            
            # Skip source listings as they'll be automatically added
            if in_sources_section:
                continue
            
            # Skip lines showing step-by-step thinking
            if line.strip().startswith(("Step ", "step ", "- Step ", "STEP ")) or \
               "final answer" in line.lower() or \
               "feedback section" in line.lower():
                skip_line = True
                continue
            
            # Start capturing again when we hit a meaningful header
            if line.strip().startswith('#') or line.strip().startswith('##'):
                skip_line = False
            
            if not skip_line:
                cleaned_lines.append(line)
        
        # Join back into a single string
        cleaned = '\n'.join(cleaned_lines)
        
        # Further process to remove duplicate paragraphs
        paragraphs = cleaned.split('\n\n')
        unique_paragraphs = []
        paragraph_content = set()
        
        for paragraph in paragraphs:
            # Normalize the paragraph for comparison (lowercase, remove whitespace)
            normalized = ' '.join(paragraph.lower().split())
            if normalized and normalized not in paragraph_content and len(normalized) > 10:
                paragraph_content.add(normalized)
                unique_paragraphs.append(paragraph)
        
        # Join back into a single string
        return '\n\n'.join(unique_paragraphs)

    def _create_fallback_analysis(self, search_results, query):
        """Create a fallback analysis when LLM fails, based on search result titles."""
        if not search_results:
            return "No search results were found for this query. Please try a different query or check your internet connection."
        
        # Extract main topics from the titles
        titles = [result.get("title", "").strip() for result in search_results if result.get("title")]
        
        # Filter out empty titles and limit to the first 5
        titles = [t for t in titles if t][:5]
        
        if not titles:
            return "Search results were found but couldn't be analyzed. Please review the sources below for information on this topic."
        
        # Create a simple analysis based on source titles
        fallback_text = f"### Key Information on {query}\n\n"
        fallback_text += "Based on the search results, here are some key topics related to your query:\n\n"
        
        # Add bullet points for each title
        for i, title in enumerate(titles, 1):
            fallback_text += f"- {title} [Source {i}]\n"
        
        fallback_text += "\n### Note\n\n"
        fallback_text += "This is an automated summary generated from search result titles. For more detailed information, please review the full sources listed below."
        
        return fallback_text

_web_research_function = None


def get_web_research_agent():
    """
    Get the shared Web Research Agent (created once, reused on every research pass).

    Returns:
        A callable function that takes a WorkflowState and calls the agent's run method.
        This makes it compatible with LangGraph which expects nodes to be callable.
    """
    global _web_research_function
    if _web_research_function is None:
        agent = WebResearchAgent()

        async def web_research_function(state: WorkflowState) -> WorkflowState:
            """Callable wrapper for the web research agent"""
            return await agent.run(state)

        _web_research_function = web_research_function
    return _web_research_function