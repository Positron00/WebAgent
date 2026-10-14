"""
Senior Research Agent for the WebAgent backend.

This agent synthesizes findings from other research agents and produces a comprehensive report.
"""
from typing import Dict, List, Any
import ast
import csv
import json
import logging
import re
from datetime import datetime

from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from app.models.task import WorkflowState
from app.services.llm import get_agent_llm
from app.security.hop_guard import acheck_hop, kept_text, task_items
from app.security.types import ContextItem
from app.security.guarded_llm import register_template
from app.agents.base_agent import BaseAgent
from app.core.config import settings
from app.core.loadEnvYAML import get_agents_config
from app.core.log_context import describe_text

logger = logging.getLogger(__name__)

# A reply whose evaluation can't be read counts as "needs more research" (IMP-031).
UNREADABLE_EVALUATION = {
    "research_status": "unknown",
    "score": None,
    "requirements_met": [],
    "missing_information": [],
    "research_questions": [],
    "requires_additional_research": True,
}

# Prompt for the Senior Research Agent
SENIOR_RESEARCH_PROMPT = """You are a Senior Research Agent specializing in synthesizing information from multiple sources.

USER QUERY: {query}
SUPERVISOR REQUIREMENTS: {requirements}
RESEARCH ITERATION: {research_iteration} (Maximum: {max_iterations})

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

IMPORTANT: If research iteration has already reached {max_iterations}, you MUST complete the report with available information and not request additional research.

Format your evaluation using the following structure at the start of your output:
```evaluation
research_status: ["needs_more_research", "needs_analysis", "complete"]
requirements_met: ["requirement 1: met", "requirement 2: not met", ...]
missing_information: ["item 1", "item 2", ...]
research_questions: ["question 1", "question 2", ...]
```

After the evaluation, provide your complete synthesized report.
"""

# Static prompt text is never scored by the safety net (SEC-032).
register_template("senior_research", SENIOR_RESEARCH_PROMPT)

# Define the evaluation output structure
class ResearchEvaluation(BaseModel):
    """Structured evaluation of research quality and completeness."""
    research_status: str = Field(description="Research status")
    requirements_met: List[str] = Field(description="List of requirements met")
    missing_information: List[str] = Field(description="List of missing or insufficient information")
    research_questions: List[str] = Field(description="Specific questions that need to be addressed in additional research")

# The prompt's own example values; models sometimes copy them back.
_PLACEHOLDER = re.compile(r"^(question|item|requirement)\s*\d+(:.*)?$|^\.{2,}$|^…$", re.IGNORECASE)


def _parse_list(value: str) -> List[str]:
    """Parse `["a, b", "c"]` from the evaluation block.

    Splitting on every comma cut questions that contain commas into fragments ("b", "middle"),
    which were then searched for; quoted items are kept whole.
    """
    for parse in (json.loads, ast.literal_eval):
        try:
            items = parse(value)
            if isinstance(items, list):
                # literal_eval reads a trailing `...` as Ellipsis
                return [str(item).strip() for item in items if item is not Ellipsis and str(item).strip()]
        except (ValueError, SyntaxError, TypeError):
            continue
    inner = value[1:-1]
    row = next(csv.reader([inner], skipinitialspace=True), [])
    return [item.strip().strip('"\'') for item in row if item.strip().strip('"\'')]


def _is_placeholder(item: Any) -> bool:
    text = str(item).strip()
    return len(text) < 3 or bool(_PLACEHOLDER.match(text))


class SeniorResearchAgent(BaseAgent):
    """
    Senior Research Agent that synthesizes findings and creates the final report.
    """
    
    def __init__(self):
        """Initialize the Senior Research Agent."""
        super().__init__(name="senior_research")
        self.max_iterations = get_agents_config().senior_research.max_follow_up_questions
        self.prompt = ChatPromptTemplate.from_template(SENIOR_RESEARCH_PROMPT).partial(
            max_iterations=str(self.max_iterations)
        )
        self.llm = get_agent_llm("senior_research")
        self.parser = StrOutputParser()
        self.chain = self.prompt | self.llm | self.parser
        self.chain = self.trace_chain(self.chain)

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
                            eval_dict[key] = _parse_list(value)
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
                
                # Models sometimes write `research_status: [complete]`; keep it a plain string
                status = eval_dict.get("research_status")
                if isinstance(status, list):
                    eval_dict["research_status"] = str(status[0]) if status else ""
                elif isinstance(status, bool):
                    eval_dict["research_status"] = "complete" if status else "needs_more_research"

                # Add requires_additional_research key based on score or research_status
                if "requires_additional_research" not in eval_dict:
                    if "score" in eval_dict and isinstance(eval_dict["score"], int):
                        # Lower scores indicate more research needed
                        eval_dict["requires_additional_research"] = eval_dict["score"] < 8
                    elif "research_status" in eval_dict:
                        # Check if research is complete based on status
                        eval_dict["requires_additional_research"] = eval_dict["research_status"].lower() != "complete"
                    else:
                        # Default to requiring more research if we have missing information
                        eval_dict["requires_additional_research"] = bool(eval_dict.get("missing_information", []))
                
                for list_key in ("research_questions", "missing_information"):
                    if isinstance(eval_dict.get(list_key), list):
                        eval_dict[list_key] = [q for q in eval_dict[list_key] if not _is_placeholder(q)]

                # Ensure research_questions exists - needed for tests
                if "research_questions" not in eval_dict:
                    # If not specified, extract from missing_information
                    eval_dict["research_questions"] = [f"Please provide more information about {item}" 
                                                     for item in eval_dict.get("missing_information", [])]
                
                # Set research status if missing
                if "research_status" not in eval_dict:
                    if eval_dict.get("requires_additional_research", False):
                        eval_dict["research_status"] = "needs_more_research"
                    else:
                        eval_dict["research_status"] = "complete"
                
                return eval_dict
            else:
                logger.warning("Evaluation block not found in output; treating research as unknown")
                return dict(UNREADABLE_EVALUATION)
        except Exception as e:
            logger.warning(f"Could not parse evaluation ({type(e).__name__}: {e}); treating research as unknown")
            return dict(UNREADABLE_EVALUATION)
            
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
            logger.info(f"Senior Research Agent processing query ({describe_text(state.query)})")
            if settings.WEBAGENT_ENV == "dev":
                logger.debug("Query: %s", state.query)
            
            # Get the supervisor's requirements
            supervisor_output = state.reports.get("supervisor", {})
            requirements = supervisor_output.get("requirements", ["No specific requirements provided"])
            
            # The research node sets research_iteration before each research pass.
            current_iteration = state.context.get("research_iteration", 1)
            
            # Get the findings from both research agents
            web_findings = state.reports.get("web_research", "No web research findings available.")
            internal_findings = state.reports.get("internal_research", "No internal research findings available.")
            
            logger.debug(f"Synthesizing findings from web and internal research (iteration {current_iteration}/{self.max_iterations})")
            
            hop = await acheck_hop(
                "senior_research",
                task_items(state)
                + [
                    ContextItem("agent_output", str(web_findings), ref="web_research"),
                    ContextItem("agent_output", str(internal_findings), ref="internal_research"),
                ],
                state,
            )
            web_findings = kept_text(hop, "web_research", str(web_findings))
            internal_findings = kept_text(hop, "internal_research", str(internal_findings))
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
            requires_additional_research = evaluation.get("requires_additional_research", False)
            
            # Prepare senior research output
            senior_research_output = {
                "research_status": research_status,
                "requirements_met": evaluation.get("requirements_met", []),
                "missing_information": evaluation.get("missing_information", []),
                "research_questions": evaluation.get("research_questions", []),
                "research_iteration": current_iteration,
                "synthesis": synthesis,
                "timestamp": datetime.now().isoformat()
            }
            
            # Update the state with the senior research output
            state.update_with_agent_output("senior_research", senior_research_output)
            
            # Check if we've reached max iterations - force research complete
            reached_max_iterations = current_iteration >= self.max_iterations
            
            # If we need more research and haven't reached the maximum iterations
            if (research_status == "needs_more_research" or requires_additional_research) and not reached_max_iterations:
                logger.info(f"Senior Research Agent requesting additional research (iteration {current_iteration}/{self.max_iterations})")
                state.add_log(f"Senior Research: Requesting more research after iteration {current_iteration}")
                
                # Add feedback to the context for other agents to use
                state.context["research_feedback"] = senior_research_output
                state.context["continue_research"] = True
            elif research_status == "needs_analysis":
                logger.info("Senior Research Agent indicating analysis is needed")
                # Mark that analysis is needed in the context
                state.context["needs_analysis"] = True
                state.context["continue_research"] = False
                state.context["research_complete"] = True
            else:
                logger.info("Senior Research Agent completed synthesis, proceeding to Team Manager")
                state.add_log("Senior Research: Findings verified, passing to Team Manager")
                # If research is complete, mark it in the context
                state.context["research_complete"] = True
                state.context["continue_research"] = False
                
                # Add verified findings to context - needed for tests
                state.context["verified_findings"] = {
                    "web_research": web_findings,
                    "internal_research": internal_findings,
                    "synthesis": synthesis,
                    "requirements_met": evaluation.get("requirements_met", []),
                    "score": evaluation.get("score", 10)
                }
                
                # Add research feedback even if complete to satisfy tests
                state.context["research_feedback"] = senior_research_output
            
            # Handle max iterations case
            if reached_max_iterations and (research_status == "needs_more_research" or requires_additional_research):
                logger.info("Senior Research Agent reached maximum iterations, finalizing research")
                state.context["research_complete"] = True
                state.context["continue_research"] = False
                state.context["verified_findings"] = {
                    "web_research": web_findings,
                    "internal_research": internal_findings, 
                    "synthesis": synthesis,
                    "requirements_met": evaluation.get("requirements_met", []),
                    "score": evaluation.get("score", 10),
                    "max_iterations_reached": True
                }
            
            return state
            
        except Exception as e:
            logger.error(f"Error in Senior Research Agent: {str(e)}")
            # Set error information in the state
            state.error = str(e)
            # Mark research as complete in case of error
            state.context["research_complete"] = True
            state.context["continue_research"] = False
            state.context["error"] = str(e)
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