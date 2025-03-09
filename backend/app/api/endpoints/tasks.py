"""
Task management endpoints for the WebAgent backend.
"""
from typing import Dict, List
from fastapi import APIRouter, HTTPException

from app.models.task import TaskInfo, TaskList
from app.services.task_manager import TaskManager

router = APIRouter()
task_manager = TaskManager()

@router.get("/", response_model=TaskList)
async def list_tasks():
    """
    List all active and completed tasks.
    """
    tasks = task_manager.list_tasks()
    return TaskList(tasks=tasks)

@router.get("/{task_id}", response_model=TaskInfo)
async def get_task(task_id: str):
    """
    Get details about a specific task.
    """
    if not task_manager.task_exists(task_id):
        raise HTTPException(status_code=404, detail="Task not found")
    
    task_info = task_manager.get_task_info(task_id)
    return task_info

@router.delete("/{task_id}")
async def delete_task(task_id: str):
    """
    Delete a task and its results.
    """
    if not task_manager.task_exists(task_id):
        raise HTTPException(status_code=404, detail="Task not found")
    
    task_manager.delete_task(task_id)
    return {"status": "success", "message": f"Task {task_id} deleted"} 