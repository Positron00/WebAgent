"""Runs that take too long are stopped and saved once as a timeout (IMP-041)."""
import asyncio
from uuid import uuid4

import pytest

from app.core.config import settings
from app.models.task import WorkflowState
from app.services import result_store, run_supervisor, task_manager as task_manager_module
from app.services.run_supervisor import TIMEOUT_MESSAGE, RunSupervisor
from app.services.task_manager import TaskManager


class _FakeWorkflow:
    def __init__(self, seconds: float):
        self.seconds = seconds
        self.cancelled = False

    async def ainvoke(self, state):
        try:
            await asyncio.sleep(self.seconds)
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        return WorkflowState(query=state.query, final_report={"title": "R", "content": "done"})


@pytest.fixture
def one_second_limit(monkeypatch):
    monkeypatch.setattr(settings, "TASK_MAX_SECONDS", 1)
    monkeypatch.setattr(run_supervisor, "POLL_SECONDS", 0.1)


def _use_workflow(monkeypatch, workflow):
    monkeypatch.setattr(task_manager_module, "get_agent_workflow", lambda: workflow)


async def _wait_until(check, seconds=5.0):
    for _ in range(int(seconds / 0.05)):
        if check():
            return
        await asyncio.sleep(0.05)
    raise AssertionError("condition not reached in time")


@pytest.mark.asyncio
async def test_slow_run_is_stopped_and_saved_as_timeout(one_second_limit, monkeypatch):
    workflow = _FakeWorkflow(seconds=30)
    _use_workflow(monkeypatch, workflow)
    manager = TaskManager()
    task_id = str(uuid4())
    manager.register_pending(task_id, "slow question")

    manager.start(task_id, "slow question", {}, direct=False)
    await _wait_until(lambda: (result_store.load_result(task_id) or {}).get("error_code") == "timeout")
    await _wait_until(lambda: not manager.supervisor.is_running(task_id))

    saved = result_store.load_result(task_id)
    assert saved["status"] == "error"
    assert saved["error"] == TIMEOUT_MESSAGE
    assert saved["query"] == "slow question"
    assert workflow.cancelled
    assert manager.get_task_status(task_id) == ("error", {"error": TIMEOUT_MESSAGE, "error_code": "timeout"})
    manager.active_tasks.pop(task_id, None)
    manager.task_results.pop(task_id, None)
    assert manager.get_task_status(task_id) == ("error", {"error": TIMEOUT_MESSAGE, "error_code": "timeout"})


@pytest.mark.asyncio
async def test_late_success_after_a_stop_is_not_saved(one_second_limit):
    task_id = str(uuid4())
    result_store.save_result(task_id, {"task_id": task_id, "query": "q", "status": "processing"})

    async def ignores_cancel():
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            pass  # a run that swallows the cancel and still finishes
        assert result_store.save_final(task_id, status="completed", result={"content": "late"}) is False

    supervisor = RunSupervisor()
    supervisor.start(task_id, ignores_cancel())
    await _wait_until(lambda: not supervisor.is_running(task_id))

    saved = result_store.load_result(task_id)
    assert (saved["status"], saved["error_code"]) == ("error", "timeout")


@pytest.mark.asyncio
async def test_normal_run_is_unaffected(one_second_limit, monkeypatch):
    _use_workflow(monkeypatch, _FakeWorkflow(seconds=0.1))
    manager = TaskManager()
    task_id = str(uuid4())
    manager.register_pending(task_id, "quick question")

    manager.start(task_id, "quick question", {}, direct=False)
    await _wait_until(lambda: (result_store.load_result(task_id) or {}).get("status") == "completed")
    await asyncio.sleep(1.3)  # past the limit: nothing may overwrite the finished record

    saved = result_store.load_result(task_id)
    assert saved["status"] == "completed"
    assert "error_code" not in saved
    assert saved["result"]["content"] == "done"
    assert not manager.supervisor.is_running(task_id)


def test_save_final_keeps_identity_fields_and_refuses_after_a_stop():
    task_id = str(uuid4())
    result_store.save_result(task_id, {"task_id": task_id, "query": "q", "created_at": "2026-10-14T08:00:00",
                                       "status": "processing"})

    assert result_store.save_final(task_id, status="error", error_code="timeout", error="stopped") is True
    saved = result_store.load_result(task_id)
    assert (saved["query"], saved["created_at"], saved["error_code"]) == ("q", "2026-10-14T08:00:00", "timeout")

    assert result_store.save_final(task_id, status="completed") is False
    assert result_store.load_result(task_id)["status"] == "error"
