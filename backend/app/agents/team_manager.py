"""
Team Manager Agent for the WebAgent backend.

This agent coordinates outputs from all specialized agents and creates a comprehensive final report.
"""
from typing import Dict, List, Any
import logging
from datetime import datetime
import json

from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.output import Generation

from app.models.task import WorkflowState
from app.services.llm import get_llm
from app.agents.base_agent import BaseAgent
from app.core.metrics import timing_decorator, track_task_completion

logger = logging.getLogger(__name__)

# Prompt for the Team Manager Agent
TEAM_MANAGER_PROMPT = """You are a Team Manager Agent responsible for compiling comprehensive research reports that integrate findings from multiple specialized agents.

USER QUERY: {query}

You have received the following reports:

RESEARCH SYNTHESIS (Senior Research Agent):
{research_synthesis}

DATA ANALYSIS (Data Analysis Agent):
{data_analysis}

VISUALIZATIONS (Coding Assistant Agent):
{visualization_code}

Your task:
1. Create a comprehensive final report that integrates all the information
2. Structure the report with clear sections:
   - Executive Summary
   - Research Findings (from Senior Research)
   - Data Analysis & Insights
   - Visualizations (include the code and explain what the visualizations show)
   - Conclusions & Recommendations
3. Ensure the report addresses the original user query comprehensively
4. Maintain all citations and sources from the original research
5. Format the report as clean, professional markdown with appropriate headings
6. Add a "References" section at the end with all cited sources

Respond with a JSON object structured as follows (example structure):
{{
  "summary": "brief executive summary",
  "recommendations": ["rec1", "rec2", "rec3"],
  "analysis": "detailed analysis text",
  "next_steps": "suggested next steps",
  "sources": [
    {{"url": "source_url", "title": "source_title"}}
  ]
}}

IMPORTANT: Your goal is to create a unified, cohesive report that reads as if it was written by a single expert rather than assembled from multiple sources. Ensure the report flows logically and maintains a consistent tone throughout.
"""

class TeamManagerAgent(BaseAgent):
    """
    Team Manager Agent that compiles final reports from specialized agents.
    """
    
    def __init__(self):
        """Initialize the Team Manager Agent."""
        super().__init__(name="team_manager")
        self.prompt = ChatPromptTemplate.from_template(TEAM_MANAGER_PROMPT)
        self.llm = get_llm("gpt-4-turbo")
        self.parser = StrOutputParser()
        self.chain = self.prompt | self.llm | self.parser
        self.chain = self.trace_chain(self.chain)
        
    @timing_decorator
    async def run(self, state: WorkflowState) -> WorkflowState:
        """
        Run the Team Manager Agent on the current workflow state.
        
        Args:
            state: The current workflow state
            
        Returns:
            Updated workflow state with the final compiled report
        """
        start_time = datetime.now()
        task_id = f"team_manager_{start_time.timestamp()}"
        
        try:
            logger.info(f"Team Manager Agent processing query: {state.query}")
            
            # Get reports from each specialized agent
            research_synthesis = state.context.get("verified_findings", "No research synthesis available.")
            data_analysis = state.context.get("data_insights", "No data analysis available.")
            visualization_code = state.context.get("visualization_code", "No visualization code available.")
            
            # Log sources availability
            logger.debug(f"Research synthesis available: {len(research_synthesis) > 50}")
            logger.debug(f"Data analysis available: {len(data_analysis) > 50}")
            logger.debug(f"Visualization code available: {len(visualization_code) > 50}")
            
            logger.debug("Compiling final report from all agent outputs")
            
            # Run the LLM chain to compile the final report
            final_report_content = await self.chain.ainvoke({
                "query": state.query,
                "research_synthesis": research_synthesis,
                "data_analysis": data_analysis,
                "visualization_code": visualization_code
            })
            
            # Parse the report content as JSON
            try:
                report_json = json.loads(final_report_content)
            except json.JSONDecodeError:
                # If JSON parsing fails, create a fallback structure
                logger.warning("Failed to parse LLM output as JSON. Using fallback structure.")
                report_json = {
                    "summary": "Error formatting report",
                    "recommendations": ["Check system output"],
                    "analysis": final_report_content[:500] + "...",
                    "next_steps": "Contact support",
                    "sources": []
                }
            
            # Log report size for monitoring
            logger.info(f"Generated final report with {len(final_report_content)} characters")
            
            # Create the team manager report
            team_manager_report = {
                "status": "completed",
                "report": final_report_content,
                "timestamp": datetime.now().isoformat(),
                **report_json  # Include the parsed JSON structure
            }
            
            # Add the report to the state
            state.reports["team_manager"] = team_manager_report
            
            # Prepare the final report with more metadata
            final_report = {
                "title": f"Comprehensive Analysis: {state.query}",
                "content": final_report_content,
                "type": "comprehensive_report",
                "contains_visualizations": "visualization_code" in state.context,
                "sources": report_json.get("sources", []),
                "timestamp": datetime.now().isoformat()
            }
            
            # Update the final report in the state
            state.final_report = final_report
            
            # Mark the task as completed
            state.completed = True
            
            # Log successful completion
            logger.info("Team Manager Agent completed final report successfully")
            
            # Execution time tracking for metrics
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            track_task_completion(task_id, duration, "completed")
            
            return state
            
        except Exception as e:
            logger.error(f"Error in Team Manager Agent: {str(e)}", exc_info=True)
            state.error = f"Team Manager Agent error: {str(e)}"
            
            # For testing, handle the error case by ensuring reports exist
            if "team_manager" not in state.reports:
                state.reports["team_manager"] = {
                    "status": "error",
                    "error": str(e),
                    "summary": "Error in report generation",
                    "recommendations": ["Check system logs"],
                    "analysis": "An error occurred",
                    "sources": []
                }
            
            # Track error in metrics
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            track_task_completion(task_id, duration, "error")
            
            return state

# Singleton instance
_team_manager_instance = None

def get_team_manager_agent() -> TeamManagerAgent:
    """
    Get an instance of the Team Manager Agent (singleton pattern).
    
    Returns:
        A TeamManagerAgent instance
    """
    global _team_manager_instance
    if _team_manager_instance is None:
        _team_manager_instance = TeamManagerAgent()
    return _team_manager_instance 