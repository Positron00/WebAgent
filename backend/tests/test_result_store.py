"""Tests for saving task results to disk and reading them back after a restart."""
import uuid

import pytest

from app.services import result_store
from app.services.task_manager import TaskManager


@pytest.fixture
def results_dir():
    # conftest.py already points RESULTS_DIR at a temporary folder
    return result_store.RESULTS_DIR


def test_save_and_load_round_trip(results_dir):
    task_id = str(uuid.uuid4())
    record = {"task_id": task_id, "status": "completed", "result": {"content": "# Report"}}

    path = result_store.save_result(task_id, record)

    assert path == results_dir / f"{task_id}.json"
    assert result_store.load_result(task_id) == record


def test_rejects_ids_that_are_not_uuids(results_dir):
    assert result_store.save_result("../../etc/passwd", {"x": 1}) is None
    assert result_store.load_result("../../etc/passwd") is None
    assert not results_dir.exists() or list(results_dir.iterdir()) == []


def test_missing_result_returns_none(results_dir):
    assert result_store.load_result(str(uuid.uuid4())) is None


def test_task_manager_falls_back_to_saved_result(results_dir):
    task_id = str(uuid.uuid4())
    result_store.save_result(
        task_id, {"status": "completed", "result": {"content": "saved report"}}
    )

    # A task the in-memory manager never saw, as after a server restart
    status, result = TaskManager().get_task_status(task_id)

    assert status == "completed"
    assert result == {"content": "saved report"}


# --- 3.3.0: cancel markers, progress, delete, list, retention (IMP-050, IMP-052, IMP-071, IMP-072)

import asyncio  # noqa: E402
import os  # noqa: E402
from datetime import datetime, timedelta  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402

from app.core.config import settings  # noqa: E402
from tests.fakes.workers import separate_task_manager  # noqa: E402


def _save(task_id, created_at, query="what is the capital of France?", status="completed"):
    result_store.save_result(
        task_id, {"task_id": task_id, "query": query, "status": status, "created_at": created_at}
    )


def test_cancel_marker_round_trip(results_dir):
    task_id = str(uuid.uuid4())
    assert not result_store.cancel_requested(task_id)
    assert result_store.request_cancel(task_id)
    assert result_store.cancel_requested(task_id)
    assert not result_store.request_cancel("../evil")


def test_save_progress_only_while_processing(results_dir):
    task_id = str(uuid.uuid4())
    assert result_store.save_progress(task_id, {"step": "x"}) is False  # no record yet
    _save(task_id, "2026-10-14T08:00:00", status="processing")
    assert result_store.save_progress(task_id, {"step": "x"}) is True
    assert result_store.load_result(task_id)["progress"] == {"step": "x"}

    result_store.save_final(task_id, status="completed", result={"content": "done"})
    assert result_store.save_progress(task_id, {"step": "y"}) is False
    assert "progress" not in result_store.load_result(task_id)


def test_delete_removes_record_temp_file_and_marker(results_dir):
    task_id = str(uuid.uuid4())
    _save(task_id, "2026-10-14T08:00:00")
    result_store.request_cancel(task_id)
    (results_dir / f"{task_id}.json.tmp").write_text("{}")

    assert result_store.delete_result(task_id) is True
    assert list(results_dir.iterdir()) == []
    assert result_store.delete_result(task_id) is False
    assert result_store.delete_result("not-a-uuid") is False


def test_list_is_newest_first_paginated_and_hides_the_question(results_dir):
    ids = [str(uuid.uuid4()) for _ in range(3)]
    for day, task_id in enumerate(ids, start=1):
        _save(task_id, f"2026-10-0{day}T08:00:00")

    rows = result_store.list_results()
    assert [row["task_id"] for row in rows] == list(reversed(ids))
    assert rows[0]["query"] == f"{len('what is the capital of France?')} chars"
    assert "France" not in str(rows)
    assert set(rows[0]) == {"task_id", "status", "created_at", "completed_at", "query"}
    assert [row["task_id"] for row in result_store.list_results(limit=1, offset=1)] == [ids[1]]


def test_cleanup_by_created_at_with_mtime_fallback(results_dir):
    now = datetime(2026, 10, 14, 12, 0, 0)
    old, new, unreadable_old, unreadable_new = (str(uuid.uuid4()) for _ in range(4))
    _save(old, (now - timedelta(days=31)).isoformat())
    result_store.request_cancel(old)
    _save(new, (now - timedelta(days=29)).isoformat())
    _save(unreadable_old, "not a date")
    _save(unreadable_new, "not a date")
    old_time = (now - timedelta(days=40)).timestamp()
    os.utime(results_dir / f"{unreadable_old}.json", (old_time, old_time))
    new_time = (now - timedelta(days=1)).timestamp()
    os.utime(results_dir / f"{unreadable_new}.json", (new_time, new_time))
    stale_marker = results_dir / f"{uuid.uuid4()}.cancel"
    stale_marker.touch()
    os.utime(stale_marker, (old_time, old_time))

    assert result_store.cleanup(30, now=now) == 4

    left = sorted(path.name for path in results_dir.iterdir())
    assert left == sorted([f"{new}.json", f"{unreadable_new}.json"])


def test_cleanup_never_raises_and_daily_loop_runs_at_start(results_dir, monkeypatch):
    calls = []
    monkeypatch.setattr(settings, "RESULTS_RETENTION_DAYS", 7)

    def failing_cleanup(days, now=None):
        calls.append(days)
        raise OSError("disk gone")

    monkeypatch.setattr(result_store, "cleanup", failing_cleanup)

    async def run_briefly():
        task = asyncio.create_task(result_store.cleanup_daily(interval_sec=0.01))
        await asyncio.sleep(0.05)
        task.cancel()
        assert not task.done() or task.cancelled()

    asyncio.run(run_briefly())
    assert calls[0] == 7 and len(calls) >= 2


def test_list_and_delete_endpoints_use_the_files(results_dir, monkeypatch):
    monkeypatch.setattr(settings, "API_KEY_ENABLED", False)
    from app.api.endpoints import tasks as tasks_endpoint
    from main import app

    writer, reader = separate_task_manager(), separate_task_manager()
    monkeypatch.setattr(tasks_endpoint, "task_manager", reader)
    ids = [str(uuid.uuid4()) for _ in range(3)]
    for task_id in ids:
        writer.register_pending(task_id, "a private question")
    client = TestClient(app)

    body = client.get("/api/v1/tasks/", params={"limit": 2}).json()
    assert len(body["tasks"]) == 2
    assert "private" not in str(body)
    assert client.get("/api/v1/tasks/", params={"limit": 0}).status_code == 422
    assert client.get("/api/v1/tasks/", params={"limit": 101}).status_code == 422

    assert client.delete(f"/api/v1/tasks/{ids[0]}").status_code == 200
    assert result_store.load_result(ids[0]) is None
    assert ids[0] not in [row["task_id"] for row in client.get("/api/v1/tasks/").json()["tasks"]]
