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
