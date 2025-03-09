"""
Chat-related data models for the WebAgent backend.
"""
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    """
    Request model for initiating a chat with the multi-agent system.
    """
    message: str = Field(..., description="The user's message or query")
    context: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="Additional context for the request"
    )
    session_id: Optional[str] = Field(
        default=None, 
        description="Session ID for continuing an existing conversation"
    )

class ChatResponse(BaseModel):
    """
    Response model for chat requests.
    """
    task_id: str = Field(..., description="The task ID for this request")
    status: str = Field(
        ..., 
        description="Status of the task (processing, completed, error)"
    )
    message: str = Field(..., description="Human-readable status message")
    result: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="The result data when status is 'completed'"
    )

class SourceDocument(BaseModel):
    """
    A source document used in research.
    """
    title: Optional[str] = None
    content: str
    url: Optional[str] = None
    relevance: Optional[float] = None
    source_type: str = "web"  # web, internal, etc.
    
class ResearchReport(BaseModel):
    """
    A research report from an agent.
    """
    agent_name: str
    content: str
    sources: List[SourceDocument] = []
    confidence: float = 0.0

class VisualizationData(BaseModel):
    """
    Data for a visualization created by the Coding Assistant.
    """
    type: str  # chart, graph, plot, etc.
    description: str
    image_data: str  # Base64 encoded image
    code: Optional[str] = None  # The code used to generate the visualization
    
class AnalysisReport(BaseModel):
    """
    Analysis report from the Data Analysis Agent.
    """
    agent_name: str
    content: str
    visualization_requirements: List[Dict[str, Any]] = []
    
class VisualizationReport(BaseModel):
    """
    Visualization report from the Coding Assistant Agent.
    """
    agent_name: str
    visualizations: List[VisualizationData] = []
    
class FinalReport(BaseModel):
    """
    Final comprehensive report.
    """
    title: str
    content: str
    visualizations: List[VisualizationData] = []
    sources: List[SourceDocument] = [] 