"""
Frontend integration endpoints for the WebAgent backend.
These endpoints provide compatibility with the frontend API expectations.
"""
from typing import Dict, List, Optional, Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel, Field

from app.models.chat import ChatRequest, ChatResponse
from app.graph.workflows import get_agent_workflow
from app.services.task_manager import TaskManager
from app.services.llm import get_llm_status
from app.core.config import settings

router = APIRouter()
task_manager = TaskManager()

# Models for frontend compatibility
class FrontendMessage(BaseModel):
    """Message format expected by the frontend."""
    role: str
    content: str

class FrontendChatRequest(BaseModel):
    """Chat request format from the frontend."""
    model: str = Field(default=settings.DEFAULT_MODEL)
    messages: List[FrontendMessage]
    max_tokens: int = Field(default=4096)
    temperature: float = Field(default=0.7)
    top_p: float = Field(default=1.0)
    frequency_penalty: float = Field(default=0.0)
    presence_penalty: float = Field(default=0.0)

class FrontendChatResponse(BaseModel):
    """Chat response format expected by the frontend."""
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]

@router.post("/chat/completions", response_model=FrontendChatResponse)
async def frontend_chat_completions(
    request: FrontendChatRequest,
    background_tasks: BackgroundTasks
):
    """
    Handle chat completions in a format compatible with the frontend.
    This endpoint adapts the frontend request format to our backend workflow.
    """
    try:
        # Extract the last user message
        last_message = next((m for m in reversed(request.messages) if m.role == "user"), None)
        if not last_message:
            raise HTTPException(status_code=400, detail="No user message found")
        
        # Create a context with previous messages if needed
        context = {
            "previous_messages": [
                {"role": m.role, "content": m.content} 
                for m in request.messages[:-1]  # Exclude the last message which we'll use as the query
            ]
        }
        
        # Generate a unique task ID
        task_id = str(uuid4())
        
        # Start the workflow in the background
        background_tasks.add_task(
            task_manager.run_workflow,
            task_id=task_id,
            message=last_message.content,
            context=context
        )
        
        # Return a response that mimics the expected format
        # Note: This is a placeholder response. The frontend should poll for the actual result.
        return FrontendChatResponse(
            choices=[
                {
                    "message": {
                        "role": "assistant",
                        "content": f"Your request is being processed (Task ID: {task_id}). Please check back in a moment."
                    },
                    "finish_reason": "processing"
                }
            ],
            usage={
                "prompt_tokens": len(last_message.content.split()),
                "completion_tokens": 0,
                "total_tokens": len(last_message.content.split())
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/chat/status/{task_id}", response_model=Dict[str, Any])
async def get_chat_completion_status(task_id: str):
    """
    Get the status of a chat completion task.
    This endpoint allows the frontend to poll for results.
    """
    # Check if the task exists
    if not task_manager.task_exists(task_id):
        raise HTTPException(status_code=404, detail="Task not found")
    
    # Get the task status
    status, result = task_manager.get_task_status(task_id)
    
    if status == "completed" and result:
        # Format the result in a way the frontend expects
        return {
            "status": "completed",
            "response": {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": result.get("content", "")
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,  # We don't track these yet
                    "completion_tokens": 0,
                    "total_tokens": 0
                }
            }
        }
    elif status == "error":
        return {
            "status": "error",
            "error": result.get("error", "Unknown error")
        }
    else:
        return {
            "status": "processing",
            "message": "Your request is still being processed"
        }

@router.get("/status", response_model=Dict[str, Any])
async def frontend_status():
    """
    Get the status of the backend services.
    This endpoint provides information about the LLM and other services.
    """
    llm_status = await get_llm_status()
    
    return {
        "status": "ok" if llm_status["status"] == "ok" else "degraded",
        "version": settings.VERSION,
        "llm": llm_status,
        "services": {
            "workflow": "ok",
            "database": "ok"
        }
    } 