"""
Senior Research Agent for the WebAgent backend.

This agent synthesizes findings from other research agents and produces a comprehensive report.
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

# Prompt for the Senior Research Agent
SENIOR_RESEARCH_PROMPT = """You are a Senior Research Agent specializing in synthesizing information from multiple sources.

USER QUERY: {query}
RESEARCH PLAN: {research_plan}

You have been provided with the following research reports:

WEB RESEARCH:
{web_research}

INTERNAL RESEARCH:
{internal_research}

Your task:
1. Synthesize the information from all research sources
2. Create a comprehensive, well-structured report that answers the user's query
3. Highlight key findings, insights, and conclusions
4. Identify any contradictions or gaps in the research
5. Maintain all source citations from the original reports
6. Format your report as markdown with clear sections and headings

IMPORTANT: Be thorough, accurate, and objective. Ensure your report is comprehensive and addresses all aspects of the query.
"""

class SeniorResearchAgent(BaseAgent):
    """
    Senior Research Agent that synthesizes findings and creates the final report.
    """
    
    def __init__(self):
        """Initialize the Senior Research Agent."""
        super().__init__(agent_name="senior_research")
        self.prompt = ChatPromptTemplate.from_template(SENIOR_RESEARCH_PROMPT)
        self.llm = get_llm("gpt-4-turbo")
        self.parser = StrOutputParser()
        self.chain = self.prompt | self.llm | self.parser
        self.chain = self.trace_chain(self.chain)
        
    async def run(self, state: WorkflowState) -> WorkflowState:
        """
        Run the Senior Research Agent on the current workflow state.
        
        Args:
            state: The current workflow state
            
        Returns:
            Updated workflow state with the synthesized research report
        """
        try:
            logger.info(f"Senior Research Agent processing query: {state.query}")
            
            # Get the research plan from the context
            research_plan = state.context.get("research_plan", {})
            
            # Get the findings from both research agents
            web_findings = state.reports.get("web_research", "No web research findings available.")
            internal_findings = state.reports.get("internal_research", "No internal research findings available.")
            
            logger.debug("Synthesizing findings from web and internal research")
            
            # Run the LLM chain to synthesize the findings
            synthesis = await self.chain.ainvoke({
                "query": state.query,
                "research_plan": research_plan.get("analysis", "No research plan provided."),
                "web_research": web_findings,
                "internal_research": internal_findings
            })
            
            # Create the senior research report
            senior_research_report = {
                "status": "completed",
                "synthesis": synthesis,
                "timestamp": datetime.now().isoformat()
            }
            
            # Add the report to the state
            state.reports["senior_research"] = senior_research_report
            
            # Add the synthesis to the context for other agents to use
            state.context["verified_findings"] = synthesis
            
            # Prepare the final report
            final_report = {
                "title": f"Research Report: {state.query}",
                "content": synthesis,
                "type": "final_report"
            }
            
            # Add the final report to the state
            state.final_report = final_report
            
            logger.info("Senior Research Agent completed synthesis successfully")
            return state
            
        except Exception as e:
            logger.error(f"Error in Senior Research Agent: {str(e)}", exc_info=True)
            state.error = f"Senior Research Agent error: {str(e)}"
            return state


def get_senior_research_agent() -> SeniorResearchAgent:
    """
    Get an instance of the Senior Research Agent.
    
    Returns:
        A SeniorResearchAgent instance
    """
    return SeniorResearchAgent() 