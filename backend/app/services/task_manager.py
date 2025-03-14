"""
Task manager service for the WebAgent backend.
Manages workflow task execution and results.
"""
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import asyncio
import logging
import traceback

from app.graph.workflows import get_agent_workflow
from app.models.task import TaskInfo, WorkflowState
from app.core.metrics import timing_decorator, track_task_start, track_task_completion

logger = logging.getLogger(__name__)

class TaskManager:
    """
    Manages the execution and results of workflow tasks.
    """
    _instance = None
    
    def __new__(cls):
        """Singleton pattern to ensure only one task manager exists."""
        if cls._instance is None:
            cls._instance = super(TaskManager, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize the task manager."""
        self.active_tasks = {}  # task_id -> task_info
        self.task_results = {}  # task_id -> result
        self.task_errors = {}   # task_id -> error details
        logger.info("Task Manager initialized")
        
    @timing_decorator
    async def run_workflow(self, task_id: str, message: str, context: Optional[Dict[str, Any]] = None):
        """
        Run a workflow task asynchronously.
        
        Args:
            task_id: The unique ID for this task
            message: The user's message/query
            context: Optional additional context
        """
        start_time = datetime.now()
        
        # Create task info
        task_info = TaskInfo(
            task_id=task_id,
            status="processing",
            created_at=datetime.now(),
            message="Task started",
            query=message,
            current_step="initializing"
        )
        
        # Store in active tasks
        self.active_tasks[task_id] = task_info
        
        # Track task start for metrics
        track_task_start(task_id)
        
        try:
            # Get the workflow
            workflow = get_agent_workflow()
            
            # Initialize the state
            initial_state = WorkflowState(
                query=message,
                context=context or {},
                current_step="planning"
            )
            
            # Update task info
            task_info.current_step = "planning"
            self.active_tasks[task_id] = task_info
            
            # Run the workflow
            logger.info(f"Starting workflow for task {task_id}")
            final_state = await workflow.ainvoke(initial_state)
            
            # Calculate duration for metrics
            duration = (datetime.now() - start_time).total_seconds()
            
            # Update task info
            task_info.status = "completed"
            task_info.completed_at = datetime.now()
            task_info.message = "Task completed successfully"
            task_info.current_step = "completed"
            
            # Log completion
            logger.info(f"Task {task_id} completed in {duration:.2f} seconds")
            
            # Store the result
            if final_state.final_report:
                logger.info(f"Storing final report for task {task_id} (content length: {len(final_state.final_report.get('content', ''))})")
                self.task_results[task_id] = final_state.final_report
                
                # Track successful completion for metrics
                track_task_completion(task_id, duration, "completed")
            else:
                logger.warning(f"Workflow completed for task {task_id} but no final report was generated")
                error_msg = "Workflow did not complete successfully - no final report"
                self.task_results[task_id] = {"error": error_msg}
                self.task_errors[task_id] = {
                    "message": error_msg,
                    "workflow_error": final_state.error
                }
                
                # Track completion with warning for metrics
                track_task_completion(task_id, duration, "incomplete")
                
        except Exception as e:
            # Calculate duration for metrics
            duration = (datetime.now() - start_time).total_seconds()
            
            # Log the full exception
            logger.exception(f"Error in workflow for task {task_id}: {str(e)}")
            
            # Update task info
            task_info.status = "error"
            task_info.message = f"Task failed: {str(e)}"
            task_info.error = str(e)
            
            # Store detailed error information
            error_detail = {
                "error": str(e),
                "traceback": traceback.format_exc(),
                "timestamp": datetime.now().isoformat()
            }
            self.task_errors[task_id] = error_detail
            
            # Store the error message in results
            self.task_results[task_id] = {"error": str(e)}
            
            # Track error for metrics
            track_task_completion(task_id, duration, "error")
            
        finally:
            # Always update the active tasks
            if task_id in self.active_tasks:
                self.active_tasks[task_id] = task_info
    
    def get_task_status(self, task_id: str) -> Tuple[str, Optional[Dict[str, Any]]]:
        """
        Get the status and result of a task.
        
        Args:
            task_id: The task ID
            
        Returns:
            Tuple of (status, result)
        """
        if task_id not in self.active_tasks and task_id not in self.task_results:
            return "not_found", None
        
        if task_id in self.active_tasks:
            task_info = self.active_tasks[task_id]
            return task_info.status, self.task_results.get(task_id)
        
        return "completed", self.task_results.get(task_id)
    
    def task_exists(self, task_id: str) -> bool:
        """Check if a task exists."""
        return task_id in self.active_tasks or task_id in self.task_results
    
    def list_tasks(self) -> List[TaskInfo]:
        """List all tasks."""
        return list(self.active_tasks.values())
    
    def get_task_info(self, task_id: str) -> Optional[TaskInfo]:
        """Get task info by ID."""
        return self.active_tasks.get(task_id)
    
    def get_task_error(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed error information for a task."""
        return self.task_errors.get(task_id)
    
    def delete_task(self, task_id: str) -> bool:
        """Delete a task and its results."""
        if task_id in self.active_tasks:
            del self.active_tasks[task_id]
        
        if task_id in self.task_results:
            del self.task_results[task_id]
            
        if task_id in self.task_errors:
            del self.task_errors[task_id]
            
        return True 