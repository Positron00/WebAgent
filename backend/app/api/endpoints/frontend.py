"""
Frontend integration endpoints for the WebAgent backend.
These endpoints provide compatibility with the frontend API expectations.
"""
from typing import Dict, List, Literal, Optional, Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.api.intake import GuardRefused, run_intake_guard
from app.models.chat import ChatRequest, ChatResponse
from app.graph.workflows import get_agent_workflow
from app.services.routing import choose_route, resolve_mode
from app.services.task_manager import TaskManager
from app.services.llm import get_llm_status
from app.core.config import settings
from app.api.http_errors import raise_safe_http_exception

router = APIRouter()
task_manager = TaskManager()

# Models for frontend compatibility
class FrontendMessage(BaseModel):
    """Message format expected by the frontend."""
    role: Literal["user", "assistant", "system"]
    content: str = Field(max_length=20000)

class FrontendChatRequest(BaseModel):
    """Chat request format from the frontend."""
    model: str = Field(default=settings.DEFAULT_MODEL)
    # 8 earlier turns and the question, the same as /api/task.
    messages: List[FrontendMessage] = Field(max_length=9)
    max_tokens: int = Field(default=4096)
    temperature: float = Field(default=0.7)
    top_p: float = Field(default=1.0)
    frequency_penalty: float = Field(default=0.0)
    presence_penalty: float = Field(default=0.0)
    agentic_toggle: Optional[bool] = Field(default=None, description="Legacy agentic flag; omitted means off with no mode")
    mode: Optional[Literal["off", "auto", "always"]] = Field(default=None, description="off, auto, or always")
    document_id: Optional[str] = Field(default=None, description="Document to extract from, if any")

class FrontendChatResponse(BaseModel):
    """Chat response format expected by the frontend."""
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]

@router.post("/chat/completions", response_model=FrontendChatResponse)
async def frontend_chat_completions(
    request: FrontendChatRequest,
):
    """
    Handle chat completions in a format compatible with the frontend.
    This endpoint adapts the frontend request format to our backend workflow.
    """
    # System messages are not passed to the agents.
    messages = [m for m in request.messages if m.role != "system"]
    user_indexes = [i for i, m in enumerate(messages) if m.role == "user"]
    if not user_indexes:
        raise HTTPException(status_code=400, detail="No user message found")
    last_message = messages[user_indexes[-1]]
    prev = [{"role": m.role, "content": m.content} for m in messages[: user_indexes[-1]]]

    # Outside the try below, so a refusal keeps its status instead of becoming a 500.
    task_id = str(uuid4())
    try:
        consented = await run_in_threadpool(
            run_intake_guard, last_message.content, prev, False, None, task_id
        )
    except GuardRefused as refused:
        return JSONResponse(status_code=refused.status, content=refused.body)

    try:
        resolved = resolve_mode(request.mode, request.agentic_toggle)
        prev_user = next((m["content"] for m in reversed(prev) if m["role"] == "user"), None)
        decision = await run_in_threadpool(
            choose_route, resolved, last_message.content, prev_user, False
        )
        use_direct = decision.route == "direct"
        context = {
            "previous_messages": prev,
            "agentic_toggle": not use_direct,
            "triage": decision.to_dict(),
        }
        if consented:
            context["consented"] = True
        if request.document_id:
            context["document_id"] = request.document_id

        task_manager.register_pending(task_id, last_message.content)
        task_manager.start(task_id, last_message.content, context, direct=use_direct)

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
        raise_safe_http_exception(500, e, public_message="Chat completion failed")

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
                            "content": result.get("content", ""),
                            **({"logs": result["logs"]} if result.get("logs") else {}),
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