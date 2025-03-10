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

from app.models.task import WorkflowState
from app.services.llm import get_llm
from app.agents.base_agent import BaseAgent

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

IMPORTANT: Your goal is to create a unified, cohesive report that reads as if it was written by a single expert rather than assembled from multiple sources. Ensure the report flows logically and maintains a consistent tone throughout.
"""

class TeamManagerAgent(BaseAgent):
    """
    Team Manager Agent that compiles final reports from specialized agents.
    """
    
    def __init__(self):
        """Initialize the Team Manager Agent."""
        super().__init__(agent_name="team_manager")
        self.prompt = ChatPromptTemplate.from_template(TEAM_MANAGER_PROMPT)
        self.llm = get_llm("gpt-4-turbo")
        self.parser = StrOutputParser()
        self.chain = self.prompt | self.llm | self.parser
        self.chain = self.trace_chain(self.chain)
        
    async def run(self, state: WorkflowState) -> WorkflowState:
        """
        Run the Team Manager Agent on the current workflow state.
        
        Args:
            state: The current workflow state
            
        Returns:
            Updated workflow state with the final compiled report
        """
        try:
            logger.info(f"Team Manager Agent processing query: {state.query}")
            
            # Get reports from each specialized agent
            research_synthesis = state.context.get("verified_findings", "No research synthesis available.")
            data_analysis = state.context.get("data_insights", "No data analysis available.")
            visualization_code = state.context.get("visualization_code", "No visualization code available.")
            
            logger.debug("Compiling final report from all agent outputs")
            
            # Run the LLM chain to compile the final report
            final_report_content = await self.chain.ainvoke({
                "query": state.query,
                "research_synthesis": research_synthesis,
                "data_analysis": data_analysis,
                "visualization_code": visualization_code
            })
            
            # Create the team manager report
            team_manager_report = {
                "status": "completed",
                "report": final_report_content,
                "timestamp": datetime.now().isoformat()
            }
            
            # Add the report to the state
            state.reports["team_manager"] = team_manager_report
            
            # Prepare the final report with more metadata
            final_report = {
                "title": f"Comprehensive Analysis: {state.query}",
                "content": final_report_content,
                "type": "comprehensive_report",
                "contains_visualizations": "visualization_code" in state.context,
                "sources": {
                    "research": "senior_research" in state.reports,
                    "data_analysis": "data_analysis" in state.reports,
                    "coding": "coding_assistant" in state.reports
                },
                "timestamp": datetime.now().isoformat()
            }
            
            # Update the final report in the state
            state.final_report = final_report
            
            logger.info("Team Manager Agent completed final report successfully")
            return state
            
        except Exception as e:
            logger.error(f"Error in Team Manager Agent: {str(e)}", exc_info=True)
            state.error = f"Team Manager Agent error: {str(e)}"
            return state


def get_team_manager_agent() -> TeamManagerAgent:
    """
    Get an instance of the Team Manager Agent.
    
    Returns:
        A TeamManagerAgent instance
    """
    return TeamManagerAgent() 