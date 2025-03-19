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
SUPERVISOR REQUIREMENTS: {requirements}
RESEARCH ITERATION: {research_iteration} (Maximum: 3)

You have been provided with the following research reports:

WEB RESEARCH:
{web_research}

INTERNAL RESEARCH:
{internal_research}

Your task:
1. Carefully verify that all requirements from the Supervisor have been met
2. Synthesize the information from all research sources
3. Create a comprehensive, well-structured report that answers the user's query
4. Highlight key findings, insights, and conclusions
5. Identify any contradictions or gaps in the research
6. Maintain all source citations from the original reports
7. Format your report as markdown with clear sections and headings

MOST IMPORTANTLY: You must evaluate if the research satisfies ALL requirements specified by the Supervisor.
If the research is insufficient, unclear, or missing key information, you should identify what additional information is needed.

Your response must include an evaluation section with:
- A verification of each requirement (met/not met)
- A list of any missing or insufficient information
- Whether additional research is needed (yes/no)
- If additional research is needed, provide specific questions that need to be addressed

IMPORTANT: If research iteration has already reached 3, you MUST complete the report with available information and not request additional research.

Format your evaluation using the following structure at the start of your output:
```evaluation
research_status: ["needs_more_research", "needs_analysis", "complete"]
requirements_met: ["requirement 1: met", "requirement 2: not met", ...]
missing_information: ["item 1", "item 2", ...]
research_questions: ["question 1", "question 2", ...]
```

After the evaluation, provide your complete synthesized report.
"""

# Define the evaluation output structure
class ResearchEvaluation(BaseModel):
    """Structured evaluation of research quality and completeness."""
    research_status: str = Field(description="Research status")
    requirements_met: List[str] = Field(description="List of requirements met")
    missing_information: List[str] = Field(description="List of missing or insufficient information")
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
                    "research_status": "complete",
                    "requirements_met": [],
                    "missing_information": [],
                    "research_questions": []
                }
        except Exception as e:
            logger.error(f"Error parsing evaluation: {str(e)}")
            return {
                "research_status": "complete",
                "requirements_met": [],
                "missing_information": [],
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
            
            # Get the supervisor's requirements
            supervisor_output = state.reports.get("supervisor", {})
            requirements = supervisor_output.get("requirements", ["No specific requirements provided"])
            
            # Track research iterations
            current_iteration = state.context.get("research_loops", 0) + 1
            state.context["research_loops"] = current_iteration
            
            # Get the findings from both research agents
            web_findings = state.reports.get("web_research", "No web research findings available.")
            internal_findings = state.reports.get("internal_research", "No internal research findings available.")
            
            logger.debug(f"Synthesizing findings from web and internal research (iteration {current_iteration}/3)")
            
            # Run the LLM chain to synthesize the findings
            full_output = await self.chain.ainvoke({
                "query": state.query,
                "requirements": requirements,
                "web_research": web_findings,
                "internal_research": internal_findings,
                "research_iteration": current_iteration
            })
            
            # Parse the evaluation from the output
            evaluation = self._parse_evaluation(full_output)
            
            # Extract the report content
            synthesis = self._get_report_content(full_output)
            
            # Determine research status
            research_status = evaluation.get("research_status", "complete")
            
            # Prepare senior research output
            senior_research_output = {
                "research_status": research_status,
                "requirements_met": evaluation.get("requirements_met", []),
                "missing_information": evaluation.get("missing_information", []),
                "research_questions": evaluation.get("research_questions", []),
                "current_iteration": current_iteration,
                "synthesis": synthesis,
                "timestamp": datetime.now().isoformat()
            }
            
            # Update the state with the senior research output
            state.update_with_agent_output("senior_research", senior_research_output)
            
            # If we need more research and haven't reached the maximum iterations
            if research_status == "needs_more_research" and current_iteration < self.max_iterations:
                logger.info(f"Senior Research Agent requesting additional research (iteration {current_iteration}/3)")
                
                # Add feedback to the context for other agents to use
                state.context["research_feedback"] = senior_research_output
            elif research_status == "needs_analysis":
                logger.info("Senior Research Agent indicating analysis is needed")
                # Mark that analysis is needed in the context
                state.context["needs_analysis"] = True
            else:
                logger.info("Senior Research Agent completed synthesis, proceeding to Team Manager")
                # If research is complete, mark it in the context
                state.context["research_complete"] = True
            
            return state
            
        except Exception as e:
            logger.error(f"Error in Senior Research Agent: {str(e)}")
            state.mark_error(f"Senior Research Agent failed: {str(e)}")
            return state


def get_senior_research_agent():
    """
    Get an instance of the Senior Research Agent.
    
    Returns:
        A callable function that takes a WorkflowState and calls the agent's run method.
        This makes it compatible with LangGraph which expects nodes to be callable.
    """
    agent = SeniorResearchAgent()
    
    # Return a callable function instead of the agent directly
    async def senior_research_function(state: WorkflowState) -> WorkflowState:
        """Callable wrapper for the senior research agent"""
        return await agent.run(state)
    
    return senior_research_function 