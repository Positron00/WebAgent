"""Task API compatible with the Next.js web client (/api/task)."""
from typing import Any, Dict, Optional
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel, Field

from app.services.task_manager import TaskManager

router = APIRouter()
task_manager = TaskManager()


class TaskRequest(BaseModel):
    query: str
    image: Optional[str] = None
    agentic: bool = Field(default=True)
    document_id: Optional[str] = None


@router.post("")
async def create_task(req: TaskRequest, background_tasks: BackgroundTasks) -> Dict[str, str]:
    task_id = str(uuid4())
    context: Dict[str, Any] = {"agentic_toggle": req.agentic}
    if req.document_id:
        context["document_id"] = req.document_id
    task_manager.register_pending(task_id, req.query)
    background_tasks.add_task(
        task_manager.run_workflow,
        task_id,
        req.query,
        context,
    )
    return {"task_id": task_id, "status": "processing"}


@router.get("/{task_id}")
async def task_status(task_id: str) -> Dict[str, Any]:
    status, result = task_manager.get_task_status(task_id)
    if status == "not_found":
        return {"status": "failed", "error": "Task not found"}
    client_status = "failed" if status == "error" else status
    return {"status": client_status, "result": result, "error": (result or {}).get("error")}
