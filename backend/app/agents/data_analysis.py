"""
Data Analysis Agent for the WebAgent backend.

This agent analyzes structured data, identifies patterns, and extracts insights from research findings.
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

# Prompt for the Data Analysis Agent
DATA_ANALYSIS_PROMPT = """You are a Data Analysis Agent specializing in identifying patterns and extracting insights from research data.

USER QUERY: {query}
RESEARCH FINDINGS: {research_findings}

Your task:
1. Analyze the research findings to identify key patterns, trends, and relationships
2. Extract quantitative insights where possible (numbers, statistics, metrics)
3. Identify qualitative insights (themes, concepts, categories)
4. Organize findings into a structured analysis
5. Prioritize insights based on relevance to the user's query
6. Format your analysis as markdown with sections for different types of insights

Consider:
- Numerical patterns and statistical relationships
- Time-based trends if temporal data is present
- Correlations between different data points
- Anomalies or outliers that might be significant
- Clusters or groupings of related information

IMPORTANT: Be analytical, precise, and data-driven. Focus on extracting meaningful insights rather than just summarizing the research.
"""

class DataAnalysisAgent(BaseAgent):
    """
    Data Analysis Agent that analyzes structured data and extracts insights.
    """
    
    def __init__(self):
        """Initialize the Data Analysis Agent."""
        super().__init__(name="data_analysis")
        self.prompt = ChatPromptTemplate.from_template(DATA_ANALYSIS_PROMPT)
        self.llm = get_llm("gpt-4-turbo")
        self.parser = StrOutputParser()
        self.chain = self.prompt | self.llm | self.parser
        self.chain = self.trace_chain(self.chain)
        
    async def run(self, state: WorkflowState) -> WorkflowState:
        """
        Run the Data Analysis Agent on the current workflow state.
        
        Args:
            state: The current workflow state
            
        Returns:
            Updated workflow state with data analysis results
        """
        try:
            logger.info(f"Data Analysis Agent processing query: {state.query}")
            
            # Get the verified findings from the senior research agent
            research_findings = state.context.get("verified_findings", "No research findings available.")
            
            logger.debug("Analyzing research findings for patterns and insights")
            
            # Run the LLM chain to analyze the data
            analysis = await self.chain.ainvoke({
                "query": state.query,
                "research_findings": research_findings
            })
            
            # Create the data analysis report
            data_analysis_report = {
                "status": "completed",
                "analysis": analysis,
                "timestamp": datetime.now().isoformat()
            }
            
            # Add the report to the state
            state.reports["data_analysis"] = data_analysis_report
            
            # Add the analysis to the context for other agents to use
            state.context["data_insights"] = analysis
            
            logger.info("Data Analysis Agent completed analysis successfully")
            return state
            
        except Exception as e:
            logger.error(f"Error in Data Analysis Agent: {str(e)}", exc_info=True)
            state.error = f"Data Analysis Agent error: {str(e)}"
            return state


def get_data_analysis_agent() -> DataAnalysisAgent:
    """
    Get an instance of the Data Analysis Agent.
    
    Returns:
        A DataAnalysisAgent instance
    """
    return DataAnalysisAgent() 