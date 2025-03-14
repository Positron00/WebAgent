"""
Task management endpoints for the WebAgent backend.
"""
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
import uuid
import logging

from app.models.task import TaskInfo, TaskList
from app.services.task_manager import TaskManager
from app.core.metrics import timing_decorator

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter()
task_manager = TaskManager()

@router.get("/", response_model=TaskList)
@timing_decorator
async def list_tasks():
    """
    List all active and completed tasks.
    """
    tasks = task_manager.list_tasks()
    return TaskList(tasks=tasks)

@router.get("/{task_id}", response_model=TaskInfo)
@timing_decorator
async def get_task(task_id: str):
    """
    Get details about a specific task.
    """
    if not task_manager.task_exists(task_id):
        raise HTTPException(status_code=404, detail="Task not found")
    
    task_info = task_manager.get_task_info(task_id)
    return task_info

@router.get("/{task_id}/result")
@timing_decorator
async def get_task_result(task_id: str):
    """
    Get the result of a completed task.
    """
    if not task_manager.task_exists(task_id):
        raise HTTPException(status_code=404, detail="Task not found")
    
    status, result = task_manager.get_task_status(task_id)
    
    if status != "completed":
        raise HTTPException(status_code=400, detail=f"Task is not completed. Current status: {status}")
    
    if not result:
        raise HTTPException(status_code=404, detail="Task result not found")
    
    return result

@router.get("/{task_id}/error")
@timing_decorator
async def get_task_error(task_id: str):
    """
    Get detailed error information for a failed task.
    """
    if not task_manager.task_exists(task_id):
        raise HTTPException(status_code=404, detail="Task not found")
    
    task_info = task_manager.get_task_info(task_id)
    if task_info.status != "error":
        raise HTTPException(status_code=400, detail=f"Task has not failed. Current status: {task_info.status}")
    
    error_details = task_manager.get_task_error(task_id)
    if not error_details:
        error_details = {"error": task_info.error or "Unknown error"}
    
    return error_details

@router.post("/create")
@timing_decorator
async def create_task(request: Dict[str, Any], background_tasks: BackgroundTasks):
    """
    Create a new task to be processed asynchronously.
    """
    if "query" not in request:
        raise HTTPException(status_code=400, detail="Query is required")
    
    query = request["query"]
    context = request.get("context", {})
    
    # Generate a task ID
    task_id = str(uuid.uuid4())
    
    # Start processing in the background
    background_tasks.add_task(task_manager.run_workflow, task_id, query, context)
    
    logger.info(f"Created new task {task_id} for query: {query}")
    
    return {
        "task_id": task_id, 
        "status": "processing",
        "message": "Task created and processing started"
    }

@router.delete("/{task_id}")
@timing_decorator
async def delete_task(task_id: str):
    """
    Delete a task and its results.
    """
    if not task_manager.task_exists(task_id):
        raise HTTPException(status_code=404, detail="Task not found")
    
    task_manager.delete_task(task_id)
    logger.info(f"Deleted task {task_id}")
    
    return {"status": "success", "message": f"Task {task_id} deleted"} 