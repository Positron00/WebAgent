"""Tests for tagging log lines with the running task's ID."""
import asyncio
import logging
import uuid

from app.core.log_context import current_task_id
from app.services import task_manager as task_manager_module
from app.services.task_manager import TaskManager


class ListHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


def capture(name="test.log_context"):
    logger = logging.getLogger(name)
    handler = ListHandler()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger, handler


def test_messages_outside_a_task_are_unchanged():
    logger, handler = capture()
    logger.info("hello %s", "world")
    record = handler.records[-1]
    assert record.getMessage() == "hello world"
    assert record.task_id == "-"


def test_messages_inside_a_task_are_tagged():
    logger, handler = capture()
    token = current_task_id.set("abcdef12-3456-7890-abcd-ef1234567890")
    try:
        logger.info("searching %s", "web")
    finally:
        current_task_id.reset(token)
    record = handler.records[-1]
    assert record.getMessage() == "[task abcdef12] searching web"
    assert record.task_id == "abcdef12-3456-7890-abcd-ef1234567890"


async def test_concurrent_tasks_keep_their_own_ids():
    logger, handler = capture("test.log_context.concurrent")

    async def agent(name):
        await asyncio.sleep(0)
        logger.info(f"{name} step")
        await asyncio.to_thread(logger.info, f"{name} thread step")

    async def task(task_id, name):
        current_task_id.set(task_id)
        await asyncio.gather(agent(name), agent(name + "-b"))

    await asyncio.gather(
        asyncio.create_task(task("11111111-aaaa", "one")),
        asyncio.create_task(task("22222222-bbbb", "two")),
    )

    for record in handler.records:
        expected = "11111111" if "one" in record.getMessage() else "22222222"
        assert record.getMessage().startswith(f"[task {expected}]")
    assert len(handler.records) == 8


async def test_task_manager_tags_logs_written_inside_the_workflow(monkeypatch):
    logger, handler = capture("test.log_context.workflow")

    class FakeWorkflow:
        async def ainvoke(self, state):
            logger.info("agent working")
            state.mark_completed({"content": "done"})
            return state

    monkeypatch.setattr(task_manager_module, "get_agent_workflow", lambda: FakeWorkflow())
    task_id = str(uuid.uuid4())

    await TaskManager().run_workflow(task_id, "q", {})

    messages = [r.getMessage() for r in handler.records]
    assert f"[task {task_id[:8]}] agent working" in messages
    assert current_task_id.get() is None
