"""
Coding Assistant Agent for the WebAgent backend.

This agent generates Python code for data visualization and analysis based on research findings.
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
from app.core.loadEnvYAML import get_agents_config

logger = logging.getLogger(__name__)

# Prompt for the Coding Assistant Agent
CODING_ASSISTANT_PROMPT = """You are a Coding Assistant Agent specializing in Python code generation for data visualization and analysis.

USER QUERY: {query}
RESEARCH FINDINGS: {research_findings}
DATA INSIGHTS: {data_insights}

Your task:
1. Generate Python code to visualize the key insights from the research
2. Create appropriate visualizations (charts, graphs, plots) that best represent the data
3. Use libraries such as matplotlib, seaborn, pandas, and numpy
4. Include clear comments explaining each section of the code
5. Make sure the code is clean, efficient, and properly structured
6. Format the code nicely with markdown code blocks

ALLOWED LIBRARIES: {allowed_libraries}

Your code should:
- Be self-contained and executable
- Have appropriate error handling
- Use sensible default data when specific data is not available
- Create visually appealing and informative charts
- Include a title and labels for all visualizations

IMPORTANT: Focus on code that directly supports the insights from the research. Make sure your code is clean, well-documented, and follows best practices.

Your output should include both the visualization code and a brief explanation of what each visualization shows, formatted as a JSON object:
{
  "explanation": "A brief explanation of what your visualizations will show and why you chose these specific visualizations",
  "visualizations": [
    {
      "title": "Visualization 1 Title",
      "description": "What this visualization shows",
      "type": "line_chart/bar_chart/etc."
    },
    ...
  ],
  "code": "```python\\n# Your complete python code here\\n```"
}
"""

class CodingAssistantAgent(BaseAgent):
    """
    Coding Assistant Agent that generates code for data visualization.
    """
    
    def __init__(self):
        """Initialize the Coding Assistant Agent."""
        super().__init__(name="coding_assistant")
        self.prompt = ChatPromptTemplate.from_template(CODING_ASSISTANT_PROMPT)
        self.llm = get_llm("gpt-4-turbo")
        self.parser = StrOutputParser()
        
        # Get configuration for allowed modules
        agents_config = get_agents_config()
        self.allowed_libraries = ", ".join(agents_config.coding.allowed_modules)
        self.timeout_seconds = agents_config.coding.timeout_seconds
        
        self.chain = self.prompt | self.llm | self.parser
        self.chain = self.trace_chain(self.chain)
        
    async def run(self, state: WorkflowState) -> WorkflowState:
        """
        Run the Coding Assistant Agent on the current workflow state.
        
        Args:
            state: The current workflow state
            
        Returns:
            Updated workflow state with code generation results
        """
        try:
            logger.info(f"Coding Assistant Agent processing query: {state.query}")
            
            # Get the research findings and data insights
            senior_research_output = state.reports.get("senior_research", {})
            research_findings = senior_research_output.get("synthesis", "No research findings available.")
            
            data_analysis = state.reports.get("data_analysis", {})
            data_insights = data_analysis.get("analysis", "No data insights available.")
            
            logger.debug("Generating visualization code based on research findings and data insights")
            
            # Run the LLM chain to generate code
            result = await self.chain.ainvoke({
                "query": state.query,
                "research_findings": research_findings,
                "data_insights": data_insights,
                "allowed_libraries": self.allowed_libraries
            })
            
            # Parse the result as JSON if possible
            try:
                # Try parsing the JSON output
                if isinstance(result, str):
                    # If the result is a string, it might be a JSON string or have a JSON section
                    if result.strip().startswith("{") and result.strip().endswith("}"):
                        # Looks like JSON - try to parse it
                        code_data = json.loads(result)
                    else:
                        # Try to extract JSON from text
                        import re
                        json_match = re.search(r'(\{.*\})', result, re.DOTALL)
                        if json_match:
                            code_data = json.loads(json_match.group(1))
                        else:
                            # No JSON structure found, create a basic structure
                            logger.warning("Could not parse JSON from LLM output, creating basic structure")
                            code_data = {
                                "explanation": "Visualization code for research findings",
                                "visualizations": [],
                                "code": result
                            }
                else:
                    # Result is already structured data
                    code_data = result
                
            except Exception as json_error:
                logger.error(f"Error parsing coding output as JSON: {str(json_error)}")
                logger.debug(f"Raw coding output: {result}")
                # Create a basic structure
                code_data = {
                    "explanation": "Visualization code for research findings",
                    "visualizations": [],
                    "code": result
                }
            
            # Create the coding assistant report
            coding_report = {
                "status": "completed",
                "code": code_data.get("code", result),
                "explanation": code_data.get("explanation", "Code for data visualization"),
                "visualizations": code_data.get("visualizations", []),
                "timestamp": datetime.now().isoformat()
            }
            
            # Add the report to the state
            state.update_with_agent_output("coding_assistant", coding_report)
            
            logger.info("Coding Assistant Agent completed code generation successfully")
            return state
            
        except Exception as e:
            logger.error(f"Error in Coding Assistant Agent: {str(e)}", exc_info=True)
            state.mark_error(f"Coding Assistant Agent error: {str(e)}")
            return state


def get_coding_assistant_agent():
    """
    Get an instance of the Coding Assistant Agent.
    
    Returns:
        A callable function that takes a WorkflowState and calls the agent's run method.
        This makes it compatible with LangGraph which expects nodes to be callable.
    """
    agent = CodingAssistantAgent()
    
    # Return a callable function instead of the agent directly
    async def coding_assistant_function(state: WorkflowState) -> WorkflowState:
        """Callable wrapper for the coding assistant agent"""
        return await agent.run(state)
    
    return coding_assistant_function 