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


def _saved_failure(task_id: str) -> None:
    result_store.save_result(task_id, {
        "task_id": task_id, "query": "What is an example?", "status": "error",
        "created_at": datetime.now(), "completed_at": datetime.now(),
        "result": {"error": "The research took too long and was stopped.", "error_code": "timeout"},
    })


def test_task_info_and_error_come_from_the_saved_record_on_another_worker():
    """GET /api/v1/tasks/{id} and /error used to crash (500) for a task run by another worker."""
    manager = TaskManager()
    task_id = str(uuid4())
    _saved_failure(task_id)

    info = manager.get_task_info(task_id)
    assert info is not None and info.status == "error" and info.query == "What is an example?"
    assert info.error == "The research took too long and was stopped."
    assert manager.get_task_error(task_id) == {
        "message": "The research took too long and was stopped.", "error_code": "timeout"}
    assert manager.get_task_info(str(uuid4())) is None


def test_legacy_task_routes_answer_for_a_saved_task(monkeypatch):
    from fastapi.testclient import TestClient

    import main
    from app.api.endpoints import tasks as tasks_endpoint
    from app.core.config import settings

    monkeypatch.setattr(settings, "API_KEY_ENABLED", False)
    manager = TaskManager()
    monkeypatch.setattr(tasks_endpoint, "task_manager", manager)
    task_id = str(uuid4())
    _saved_failure(task_id)

    client = TestClient(main.app)
    info = client.get(f"/api/v1/tasks/{task_id}")
    assert info.status_code == 200 and info.json()["status"] == "error"
    error = client.get(f"/api/v1/tasks/{task_id}/error")
    assert error.status_code == 200 and error.json()["error_code"] == "timeout"
