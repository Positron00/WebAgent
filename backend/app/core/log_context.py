"""Tag every log line written while a task runs with that task's ID.

Set `current_task_id` at the start of a task. Python copies context variables into
asyncio tasks and `asyncio.to_thread` workers, so agents running in parallel keep
their own task's ID.
"""
import logging
from contextvars import ContextVar
from typing import Optional

current_task_id: ContextVar[Optional[str]] = ContextVar("current_task_id", default=None)

_installed = False


def short_task_id(task_id: str) -> str:
    return task_id[:8]


def install_task_id_logging() -> None:
    """Add a `task_id` attribute to every log record and prefix messages during a task."""
    global _installed
    if _installed:
        return
    base_factory = logging.getLogRecordFactory()

    def factory(*args, **kwargs) -> logging.LogRecord:
        record = base_factory(*args, **kwargs)
        task_id = current_task_id.get()
        record.task_id = task_id or "-"
        if task_id and isinstance(record.msg, str):
            record.msg = f"[task {short_task_id(task_id)}] {record.msg}"
        return record

    logging.setLogRecordFactory(factory)
    _installed = True


install_task_id_logging()
