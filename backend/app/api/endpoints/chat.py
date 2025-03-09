"""
Chat endpoints for the WebAgent backend.
"""
from typing import Dict, Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends

from app.models.chat import ChatRequest, ChatResponse
from app.graph.workflows import get_agent_workflow
from app.services.task_manager import TaskManager

router = APIRouter()
task_manager = TaskManager()

@router.post("/", response_model=ChatResponse)
async def create_chat(
    request: ChatRequest, 
    background_tasks: BackgroundTasks
):
    """
    Start a new chat workflow with the multi-agent system.
    Returns a task ID that can be used to check progress and get results.
    """
    try:
        # Generate a unique task ID
        task_id = str(uuid4())
        
        # Start the workflow in the background
        background_tasks.add_task(
            task_manager.run_workflow,
            task_id=task_id,
            message=request.message,
            context=request.context or {}
        )
        
        # Return a response with the task ID
        return ChatResponse(
            task_id=task_id,
            status="processing",
            message="Your request is being processed. Check the status using the task ID."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{task_id}", response_model=ChatResponse)
async def get_chat_status(task_id: str):
    """
    Get the status and results of a chat task by ID.
    """
    # Check if the task exists
    if not task_manager.task_exists(task_id):
        raise HTTPException(status_code=404, detail="Task not found")
    
    # Get the task status
    status, result = task_manager.get_task_status(task_id)
    
    return ChatResponse(
        task_id=task_id,
        status=status,
        message="Task complete" if status == "completed" else "Task in progress",
        result=result
    ) 