"""
Task-related data models for the WebAgent backend.
"""
from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class TaskInfo(BaseModel):
    """
    Information about a task.
    """
    task_id: str
    status: str  # processing, completed, error
    created_at: datetime
    completed_at: Optional[datetime] = None
    message: str
    query: str
    current_step: Optional[str] = None
    error: Optional[str] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class TaskList(BaseModel):
    """
    List of tasks.
    """
    tasks: List[TaskInfo]

class WorkflowState(BaseModel):
    """
    State object for the agent workflow.
    """
    query: str
    context: Dict[str, Any] = Field(default_factory=dict)
    history: List[Dict[str, Any]] = Field(default_factory=list)
    reports: Dict[str, Any] = Field(default_factory=dict)
    final_report: Optional[Dict[str, Any]] = None
    current_step: str = "planning"
    completed: bool = False
    error: Optional[str] = None
    
    @property
    def agent_outputs(self) -> Dict[str, Any]:
        """Get agent outputs from reports for compatibility with tests."""
        return self.reports
    
    def update_with_agent_output(self, agent_name: str, output: Any):
        """Update state with an agent's output."""
        self.reports[agent_name] = output
        self.history.append({
            "agent": agent_name,
            "timestamp": datetime.now().isoformat(),
            "output": str(output)
        })
        
    def mark_completed(self, final_report: Dict[str, Any]):
        """Mark the workflow as completed."""
        self.completed = True
        self.final_report = final_report
        
    def mark_error(self, error: str):
        """Mark the workflow as failed with an error."""
        self.error = error 