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
from app.services.llm import get_agent_llm
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

ANALYZE REQUIREMENTS:
Based on the research synthesis, determine if additional analysis is needed:
- If the query involves data that needs statistical analysis, indicate "needs_analysis": true
- If the query would benefit from code examples or visualizations, indicate "needs_coding": true

Respond with a JSON object structured as follows:
{{
  "needs_analysis": true/false,
  "needs_coding": true/false,
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

# Global variable to store the singleton instance
_team_manager_instance = None

class TeamManagerAgent(BaseAgent):
    """
    Team Manager Agent that compiles final reports from specialized agents.
    """
    
    def __init__(self):
        """Initialize the Team Manager Agent."""
        super().__init__(name="team_manager")
        self.prompt = ChatPromptTemplate.from_template(TEAM_MANAGER_PROMPT)
        self.llm = get_agent_llm("team_manager")
        self.parser = StrOutputParser()
        self.chain = self.prompt | self.llm | self.parser
        self.chain = self.trace_chain(self.chain)

    @staticmethod
    def _should_finalize(needs_analysis: bool, needs_coding: bool, state: WorkflowState) -> bool:
        if needs_analysis and "data_analysis" not in state.reports:
            return False
        if needs_coding and "coding_review" not in state.reports:
            return False
        return True

    @staticmethod
    def _build_report_markdown(
        query: str,
        report_json: Dict[str, Any],
        code_block: str,
        review_explanation: str = "",
    ) -> str:
        recommendations = report_json.get("recommendations") or []
        rec_lines = "\n".join(f"- {r}" for r in recommendations)
        sources = report_json.get("sources") or []
        source_lines = "\n".join(
            f"- [{s.get('title', 'Source')}]({s.get('url', '')})" for s in sources if isinstance(s, dict)
        )
        sections = [
            f"# Comprehensive Analysis: {query}",
            "## Executive Summary",
            report_json.get("summary", ""),
            "## Analysis",
            report_json.get("analysis", ""),
            "## Recommendations",
            rec_lines or "_None_",
            "## Next Steps",
            report_json.get("next_steps", ""),
        ]
        if code_block and code_block != "No visualization code available.":
            code = code_block.strip()
            if not code.startswith("```"):
                code = f"```python\n{code}\n```"
            sections.extend(["## Code / Visualizations", code])
            if review_explanation:
                sections.extend(["### Code Review Notes", review_explanation])
        if source_lines:
            sections.extend(["## References", source_lines])
        return "\n\n".join(sections)
        
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
            
            # Get reports from specialized agents
            senior_research_output = state.reports.get("senior_research", {})
            research_synthesis = senior_research_output.get("synthesis", "No research synthesis available.")
            data_analysis = state.reports.get("data_analysis", {}).get("analysis", "No data analysis available.")
            coding_review = state.reports.get("coding_review", {})
            fixed_run = coding_review.get("fixed_execution") or {}
            review_explanation = coding_review.get("improvement_explanation", "")
            if coding_review.get("fixed_code") and fixed_run.get("success"):
                visualization_code = coding_review["fixed_code"]
            else:
                visualization_code = state.reports.get("coding_assistant", {}).get(
                    "code", "No visualization code available."
                )
            
            # Log sources availability
            logger.debug(f"Research synthesis available: {len(research_synthesis) > 50}")
            logger.debug(f"Data analysis available: {len(data_analysis) > 50}")
            logger.debug(f"Visualization code available: {len(visualization_code) > 50}")
            
            logger.debug("Compiling report and checking if analysis or coding is needed")
            
            # Run the LLM chain to compile the report and check for needs
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
                # Look for JSON structure in the output
                import re
                json_match = re.search(r'(\{.*\})', final_report_content, re.DOTALL)
                if json_match:
                    try:
                        report_json = json.loads(json_match.group(1))
                    except:
                        report_json = {
                            "needs_analysis": False,
                            "needs_coding": False,
                            "summary": "Error formatting report",
                            "recommendations": ["Check system output"],
                            "analysis": final_report_content[:500] + "...",
                            "next_steps": "Contact support",
                            "sources": []
                        }
                else:
                    report_json = {
                        "needs_analysis": False,
                        "needs_coding": False,
                        "summary": "Error formatting report",
                        "recommendations": ["Check system output"],
                        "analysis": final_report_content[:500] + "...",
                        "next_steps": "Contact support",
                        "sources": []
                    }
            
            # Check if additional analysis or coding is needed
            needs_analysis = report_json.get("needs_analysis", False)
            needs_coding = report_json.get("needs_coding", False)
            
            # Create the team manager report
            team_manager_report = {
                "status": "in_progress",
                "needs_analysis": needs_analysis,
                "needs_coding": needs_coding,
                "report": final_report_content,
                "timestamp": datetime.now().isoformat(),
                **report_json  # Include the parsed JSON structure
            }
            
            # Add the report to the state
            state.update_with_agent_output("team_manager", team_manager_report)
            
            if self._should_finalize(needs_analysis, needs_coding, state):
                state.add_log("Team Manager: Compiling final report")
                markdown = self._build_report_markdown(
                    state.query, report_json, visualization_code, review_explanation
                )
                state.mark_completed(
                    {
                        "title": f"Comprehensive Analysis: {state.query}",
                        "content": markdown,
                        "type": "comprehensive_report",
                        "contains_visualizations": visualization_code
                        != "No visualization code available.",
                        "sources": report_json.get("sources", []),
                        "timestamp": datetime.now().isoformat(),
                    }
                )
                logger.info("Team Manager Agent completed final report successfully")
            else:
                if needs_analysis and "data_analysis" not in state.reports:
                    state.add_log("Team Manager: Requesting data analysis")
                    logger.info("Team Manager Agent requesting data analysis")
                elif needs_coding and "coding_review" not in state.reports:
                    state.add_log("Team Manager: Requesting code and visualizations")
                    logger.info("Team Manager Agent requesting code/visualization assistance")

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            track_task_completion(task_id, duration, "completed")

            return state
            
        except Exception as e:
            logger.error(f"Error in Team Manager Agent: {str(e)}", exc_info=True)
            state.mark_error(f"Team Manager Agent error: {str(e)}")
            
            # For testing, handle the error case by ensuring reports exist
            if "team_manager" not in state.reports:
                state.update_with_agent_output("team_manager", {
                    "status": "error",
                    "error": str(e),
                    "summary": "Error in report generation",
                    "recommendations": ["Check system logs"],
                    "analysis": "An error occurred",
                    "sources": []
                })
            
            # Track error in metrics
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            track_task_completion(task_id, duration, "error")
            
            return state

def get_team_manager_agent():
    """
    Get an instance of the Team Manager Agent.
    
    Returns:
        A callable function that takes a WorkflowState and calls the agent's run method.
        This makes it compatible with LangGraph which expects nodes to be callable.
    """
    global _team_manager_instance
    
    # If we don't have an instance yet, create one
    if _team_manager_instance is None:
        agent = TeamManagerAgent()
        
        # Create the callable function
        async def team_manager_function(state: WorkflowState) -> WorkflowState:
            """Callable wrapper for the team manager agent"""
            return await agent.run(state)
        
        # Store the instance
        _team_manager_instance = team_manager_function
    
    return _team_manager_instance 