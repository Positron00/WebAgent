"""
Task management endpoints for the WebAgent backend.
"""
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
import uuid
import logging

from app.api.endpoints.task import start_legacy_task
from app.models.task import TaskInfo
from app.services import result_store
from app.services.task_manager import TaskManager
from app.core.log_context import describe_text
from app.core.metrics import timing_decorator

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter()
task_manager = TaskManager()

@router.get("/")
@timing_decorator
async def list_tasks(limit: int = Query(20, ge=1, le=100), offset: int = Query(0, ge=0)):
    """
    List saved tasks from the result files (the same answer from every worker), newest first.
    Each task's `query` is its length ("<n> chars"), never its text.
    """
    return {"tasks": result_store.list_results(limit=limit, offset=offset), "limit": limit, "offset": offset}

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

@router.post("/create", deprecated=True)
@timing_decorator
async def create_task(request: Dict[str, Any]):
    """
    Create a research task (deprecated; use POST /api/task).
    Same path as /api/task in Always mode: intake guard, length limit, run supervisor.
    """
    if "query" not in request:
        raise HTTPException(status_code=400, detail="Query is required")

    started = await start_legacy_task(request["query"], request.get("context"))
    if isinstance(started, JSONResponse):  # the intake guard refused it (403 or 409)
        return started
    logger.info(f"Created new task {started['task_id']} ({describe_text(str(request['query']))})")
    return {
        "task_id": started["task_id"],
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