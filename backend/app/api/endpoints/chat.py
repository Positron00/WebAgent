"""
Chat endpoints for the WebAgent backend.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from app.models.chat import ChatRequest, ChatResponse
from app.services.task_manager import TaskManager
from app.api.endpoints.task import start_legacy_task

router = APIRouter()
task_manager = TaskManager()

@router.post("/", response_model=ChatResponse, deprecated=True)
async def create_chat(request: ChatRequest):
    """
    Start a research run (deprecated; use POST /api/task).
    Same path as /api/task in Always mode: intake guard, length limit, run supervisor.
    Returns a task ID that can be used to check progress and get results.
    """
    started = await start_legacy_task(request.message, request.context)
    if isinstance(started, JSONResponse):  # the intake guard refused it (403 or 409)
        return started
    return ChatResponse(
        task_id=started["task_id"],
        status="processing",
        message="Your request is being processed. Check the status using the task ID."
    )

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