"""
Internal Research Agent for the WebAgent backend.

This agent searches internal knowledge bases and documents to find relevant information.
"""
from typing import Dict, List, Any
import logging
from datetime import datetime

from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

from app.models.task import WorkflowState
from app.services.llm import get_llm
from app.services.vectordb import retrieve_documents
from app.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

# Prompt for the Internal Research Agent
INTERNAL_RESEARCH_PROMPT = """You are an Internal Research Agent specializing in retrieving information from internal knowledge bases.

USER QUERY: {query}
RESEARCH PLAN: {research_plan}

You have been provided with the following internal documents:
{documents}

Your task:
1. Analyze the documents and extract relevant information for the query
2. Organize the information in a clear, structured format
3. Cite your sources with document references
4. Only include information that is directly related to the query
5. Highlight key facts, statistics, and insights
6. Format your findings as markdown

IMPORTANT: Be thorough, accurate, and objective. Cite all sources.
"""

class InternalResearchAgent(BaseAgent):
    """
    Internal Research Agent that searches knowledge bases and internal documents.
    """
    
    def __init__(self):
        """Initialize the Internal Research Agent."""
        super().__init__(agent_name="internal_research")
        self.prompt = ChatPromptTemplate.from_template(INTERNAL_RESEARCH_PROMPT)
        self.llm = get_llm("gpt-4-turbo")
        self.parser = StrOutputParser()
        self.chain = self.prompt | self.llm | self.parser
        self.chain = self.trace_chain(self.chain)
    
    def _format_internal_documents(self, documents: List[Dict[str, Any]]) -> str:
        """Format internal documents for the prompt."""
        if not documents:
            return "No relevant internal documents found."
            
        formatted_docs = ""
        for i, doc in enumerate(documents, 1):
            formatted_docs += f"DOCUMENT {i}:\n"
            formatted_docs += f"Title: {doc.get('metadata', {}).get('title', 'No title')}\n"
            formatted_docs += f"Source: {doc.get('metadata', {}).get('source', 'Internal knowledge base')}\n"
            formatted_docs += f"Content: {doc.get('page_content', 'No content')}\n\n"
            
        return formatted_docs
        
    async def run(self, state: WorkflowState) -> WorkflowState:
        """
        Execute the Internal Research Agent to search and analyze internal documents.
        
        Args:
            state: The current workflow state
            
        Returns:
            Updated workflow state with internal research results
        """
        logger.info(f"Internal Research Agent running for query: {state.query}")
        state.current_step = "internal_research"
        
        try:
            # Get the research plan
            research_plan = state.context.get("research_plan", {})
            
            # If internal knowledge is not required, skip this step
            if not research_plan.get("requires_internal_knowledge", True):
                logger.info("Internal knowledge not required by research plan. Skipping.")
                state.update_with_agent_output("internal_research", {
                    "status": "skipped",
                    "reason": "Internal knowledge not required by research plan",
                    "timestamp": datetime.now().isoformat()
                })
                return state
            
            # Retrieve relevant documents from the vector database
            logger.info("Retrieving internal documents")
            internal_documents = await retrieve_documents(state.query, k=5)
            
            if not internal_documents:
                logger.warning("No internal documents found.")
                state.update_with_agent_output("internal_research", {
                    "status": "completed",
                    "documents": [],
                    "findings": "No relevant information found in internal knowledge base.",
                    "timestamp": datetime.now().isoformat()
                })
                return state
            
            # Format internal documents for the prompt
            formatted_docs = self._format_internal_documents(internal_documents)
            
            # Run the LLM chain to analyze the internal documents
            analysis = await self.chain.ainvoke({
                "query": state.query,
                "research_plan": research_plan.get("analysis", "No research plan provided."),
                "documents": formatted_docs
            })
            
            # Create the internal research report
            internal_research_report = {
                "status": "completed",
                "documents": internal_documents,
                "findings": analysis,
                "timestamp": datetime.now().isoformat()
            }
            
            # Update the state with the internal research report
            state.update_with_agent_output("internal_research", internal_research_report)
            
            # Add the findings to the context for other agents to use
            state.context["internal_research_findings"] = analysis
            
            logger.info(f"Internal Research completed with {len(internal_documents)} documents.")
            return state
            
        except Exception as e:
            logger.error(f"Error in Internal Research Agent: {str(e)}")
            state.mark_error(f"Internal Research Agent failed: {str(e)}")
            return state


def get_internal_research_agent() -> InternalResearchAgent:
    """
    Get an instance of the Internal Research Agent.
    
    Returns:
        An InternalResearchAgent instance
    """
    return InternalResearchAgent() 