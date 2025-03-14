"""
Senior Research Agent for the WebAgent backend.

This agent synthesizes findings from other research agents and produces a comprehensive report.
"""
from typing import Dict, List, Any
import logging
from datetime import datetime

from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from app.models.task import WorkflowState
from app.services.llm import get_llm
from app.agents.base_agent import BaseAgent
from app.core.config import settings

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

CURRENT RESEARCH ITERATION: {research_iteration} (Maximum: 3)

Your task:
1. Synthesize the information from all research sources
2. Create a comprehensive, well-structured report that answers the user's query
3. Highlight key findings, insights, and conclusions
4. Identify any contradictions or gaps in the research
5. Maintain all source citations from the original reports
6. Format your report as markdown with clear sections and headings

MOST IMPORTANTLY: You must evaluate if the research satisfies all requirements from the research plan.
If the research is insufficient, unclear, or missing key information, you should identify what additional information is needed.

Your response must include an evaluation section with:
- A score from 1-10 on how well the research answers the query
- A list of any missing or insufficient information
- Whether additional research is needed (yes/no)
- If additional research is needed, provide specific questions that need to be addressed

IMPORTANT: If research iteration has already reached 3, you MUST complete the report with available information and not request additional research.

Format your evaluation using the following structure at the start of your output:
```evaluation
score: [1-10]
missing_information: ["item 1", "item 2", ...]
requires_additional_research: [yes/no]
research_questions: ["question 1", "question 2", ...]
```

After the evaluation, provide your complete synthesized report.
"""

# Define the evaluation output structure
class ResearchEvaluation(BaseModel):
    """Structured evaluation of research quality and completeness."""
    score: int = Field(description="Score from 1-10 on how well the research answers the query")
    missing_information: List[str] = Field(description="List of missing or insufficient information")
    requires_additional_research: bool = Field(description="Whether additional research is needed")
    research_questions: List[str] = Field(description="Specific questions that need to be addressed in additional research")

class SeniorResearchAgent(BaseAgent):
    """
    Senior Research Agent that synthesizes findings and creates the final report.
    """
    
    def __init__(self):
        """Initialize the Senior Research Agent."""
        super().__init__(name="senior_research")
        self.prompt = ChatPromptTemplate.from_template(SENIOR_RESEARCH_PROMPT)
        self.llm = get_llm("gpt-4-turbo")
        self.parser = StrOutputParser()
        self.chain = self.prompt | self.llm | self.parser
        self.chain = self.trace_chain(self.chain)
        
        # Set up configuration
        self.config = getattr(settings, "senior_research_config", {})
        self.max_iterations = self.config.get("max_follow_up_questions", 3)
        
    def _parse_evaluation(self, text: str) -> Dict[str, Any]:
        """
        Parse the evaluation section from the output.
        
        Args:
            text: The full text output from the LLM
            
        Returns:
            Parsed evaluation as a dictionary
        """
        try:
            # Extract the evaluation block between ```evaluation and ```
            if "```evaluation" in text and "```" in text.split("```evaluation", 1)[1]:
                eval_block = text.split("```evaluation", 1)[1].split("```", 1)[0].strip()
                
                # Parse the YAML-like format
                eval_dict = {}
                lines = eval_block.split('\n')
                for line in lines:
                    if ':' in line:
                        key, value = line.split(':', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        # Handle lists
                        if value.startswith('[') and value.endswith(']'):
                            value = value[1:-1]  # Remove brackets
                            if value:
                                items = [item.strip().strip('"\'') for item in value.split(',')]
                                eval_dict[key] = items
                            else:
                                eval_dict[key] = []
                        # Handle boolean
                        elif value.lower() in ['yes', 'true']:
                            eval_dict[key] = True
                        elif value.lower() in ['no', 'false']:
                            eval_dict[key] = False
                        # Handle integer
                        elif key == 'score' and value.isdigit():
                            eval_dict[key] = int(value)
                        else:
                            eval_dict[key] = value
                
                return eval_dict
            else:
                logger.warning("Evaluation block not found in output")
                return {
                    "score": 5, 
                    "missing_information": [], 
                    "requires_additional_research": False,
                    "research_questions": []
                }
        except Exception as e:
            logger.error(f"Error parsing evaluation: {str(e)}")
            return {
                "score": 5, 
                "missing_information": [], 
                "requires_additional_research": False,
                "research_questions": []
            }
            
    def _get_report_content(self, text: str) -> str:
        """
        Extract the report content after the evaluation section.
        
        Args:
            text: The full text output from the LLM
            
        Returns:
            Report content as a string
        """
        try:
            if "```evaluation" in text and "```" in text.split("```evaluation", 1)[1]:
                # Get everything after the evaluation block
                return text.split("```evaluation", 1)[1].split("```", 1)[1].strip()
            else:
                # If no evaluation block found, return the full text
                return text
        except Exception as e:
            logger.error(f"Error extracting report content: {str(e)}")
            return text
        
    async def run(self, state: WorkflowState) -> WorkflowState:
        """
        Run the Senior Research Agent on the current workflow state.
        
        Args:
            state: The current workflow state
            
        Returns:
            Updated workflow state with the synthesized research report or
            additional research requests
        """
        try:
            logger.info(f"Senior Research Agent processing query: {state.query}")
            
            # Get the research plan from the context
            research_plan = state.context.get("research_plan", {})
            
            # Track research iterations
            current_iteration = state.context.get("research_iteration", 1)
            state.context["research_iteration"] = current_iteration
            
            # Get the findings from both research agents
            web_findings = state.reports.get("web_research", "No web research findings available.")
            internal_findings = state.reports.get("internal_research", "No internal research findings available.")
            
            logger.debug(f"Synthesizing findings from web and internal research (iteration {current_iteration})")
            
            # Run the LLM chain to synthesize the findings
            full_output = await self.chain.ainvoke({
                "query": state.query,
                "research_plan": research_plan.get("analysis", "No research plan provided."),
                "web_research": web_findings,
                "internal_research": internal_findings,
                "research_iteration": current_iteration
            })
            
            # Parse the evaluation from the output
            evaluation = self._parse_evaluation(full_output)
            
            # Extract the report content
            synthesis = self._get_report_content(full_output)
            
            # Decide whether additional research is needed
            requires_additional_research = (
                evaluation.get("requires_additional_research", False) and
                current_iteration < self.max_iterations
            )
            
            if requires_additional_research:
                logger.info(f"Senior Research Agent requesting additional research (iteration {current_iteration})")
                
                # Create a research request to give back to research agents
                research_request = {
                    "status": "needs_more_research",
                    "score": evaluation.get("score", 0),
                    "missing_information": evaluation.get("missing_information", []),
                    "research_questions": evaluation.get("research_questions", []),
                    "current_iteration": current_iteration,
                    "timestamp": datetime.now().isoformat()
                }
                
                # Add the research request to the state
                state.update_with_agent_output("senior_research", research_request)
                
                # Add feedback to the context for other agents to use
                state.context["research_feedback"] = research_request
                state.context["research_iteration"] = current_iteration + 1
                
                # Set routing to research agents
                requires_web = research_plan.get("requires_web_search", False)
                requires_internal = research_plan.get("requires_internal_knowledge", False)
                
                # Indicate next steps in context
                state.context["next_research_agents"] = []
                if requires_web:
                    state.context["next_research_agents"].append("web_research")
                if requires_internal:
                    state.context["next_research_agents"].append("internal_research")
                
                # Set that research is continuing
                state.context["continue_research"] = True
                
                logger.info(f"Requesting additional research from: {state.context['next_research_agents']}")
                return state
            else:
                # If no additional research needed or max iterations reached, create the final report
                logger.info(f"Senior Research Agent completing synthesis (iteration {current_iteration})")
                
                # Create the senior research report
                senior_research_report = {
                    "status": "completed",
                    "score": evaluation.get("score", 5),
                    "synthesis": synthesis,
                    "timestamp": datetime.now().isoformat()
                }
                
                # Add the report to the state
                state.update_with_agent_output("senior_research", senior_research_report)
                
                # Add the synthesis to the context for other agents to use
                state.context["verified_findings"] = synthesis
                
                # Set that research is complete
                state.context["continue_research"] = False
                
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