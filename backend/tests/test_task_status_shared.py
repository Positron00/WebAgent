"""Task status must be visible to every worker process, not only the one running the task."""
from datetime import datetime, timedelta
from uuid import uuid4

from app.services import result_store
from app.services.task_manager import TaskManager


def _as_other_worker(manager: TaskManager, task_id: str) -> None:
    """Forget the in-memory state, as a different uvicorn worker would not have it."""
    manager.active_tasks.pop(task_id, None)
    manager.task_results.pop(task_id, None)


def test_pending_task_is_processing_for_another_worker():
    manager = TaskManager()
    task_id = str(uuid4())
    manager.register_pending(task_id, "What is an example?")
    _as_other_worker(manager, task_id)

    assert manager.task_exists(task_id)
    assert manager.get_task_status(task_id) == ("processing", None)


def test_stale_processing_record_is_reported_as_interrupted():
    manager = TaskManager()
    task_id = str(uuid4())
    result_store.save_result(
        task_id,
        {
            "task_id": task_id,
            "status": "processing",
            "created_at": datetime.now() - timedelta(hours=2),
            "result": None,
        },
    )

    status, result = manager.get_task_status(task_id)

    assert status == "error"
    assert "interrupted" in result["error"]


def test_finished_task_is_visible_to_another_worker():
    manager = TaskManager()
    task_id = str(uuid4())
    manager.register_pending(task_id, "q")
    result_store.save_result(
        task_id,
        {"task_id": task_id, "status": "completed", "result": {"content": "Answer"}},
    )
    _as_other_worker(manager, task_id)

    assert manager.get_task_status(task_id) == ("completed", {"content": "Answer"})
