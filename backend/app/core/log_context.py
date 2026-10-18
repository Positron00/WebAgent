"""Tag every log line written while a task runs with that task's ID.

Set `current_task_id` at the start of a task. Python copies context variables into
asyncio tasks and `asyncio.to_thread` workers, so agents running in parallel keep
their own task's ID.
"""
import re
import logging
from contextvars import ContextVar
from typing import Optional

current_task_id: ContextVar[Optional[str]] = ContextVar("current_task_id", default=None)

_installed = False


def short_task_id(task_id: str) -> str:
    return task_id[:8]


def describe_text(text: Optional[str]) -> str:
    """Log-safe description of user text: its length only (IMP-070)."""
    return f"{len(text or '')} chars"


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


# Query parameters that carry bearer credentials (recurring-research confirm and unsubscribe tokens).
_SECRET_PARAMS = re.compile(r"((?:^|[?&])(?:token|api_key|key)=)[^&#\s]*", re.IGNORECASE)


def redact_url(url: str) -> str:
    """`url` with the values of credential query parameters replaced, for logs."""
    return _SECRET_PARAMS.sub(r"\1[redacted]", url or "")


class RedactAccessLogFilter(logging.Filter):
    """Redacts credential query parameters in URL-carrying log records: uvicorn's access log (its args
    hold the path) and httpx's request log (its args hold the URL)."""

    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args
        if isinstance(args, tuple):
            record.args = tuple(redact_url(a) if isinstance(a, str) else
                                (redact_url(str(a)) if type(a).__name__ == "URL" else a) for a in args)
        return True


def install_access_log_redaction() -> None:
    for name in ("uvicorn.access", "httpx"):
        target = logging.getLogger(name)
        if not any(isinstance(f, RedactAccessLogFilter) for f in target.filters):
            target.addFilter(RedactAccessLogFilter())
