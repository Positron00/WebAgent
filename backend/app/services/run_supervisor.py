"""Run each task as an asyncio.Task and stop it when it runs too long (IMP-041).

One supervisor per worker process. A watch loop checks each run every POLL_SECONDS;
`_stop_reason` decides whether to stop it (3.3.0 adds the cancel marker there). The stop
record goes through `result_store.save_final`, so whichever of "stopped" and "finished"
is written first stays final.
"""
import asyncio
import logging
import time
from datetime import datetime
from typing import Awaitable, Callable, Dict, Optional, Set, Tuple

from app.core.config import settings
from app.services import result_store

logger = logging.getLogger(__name__)

POLL_SECONDS = 2.0
# How long a stopped run gets to unwind after cancel() before the watcher stops waiting.
STOP_GRACE_SECONDS = 5.0
TIMEOUT_MESSAGE = "The research took too long and was stopped."

# Called with (task_id, error_code, message) after a stop record is saved.
OnStop = Callable[[str, str, str], None]


def _log_crash(run: asyncio.Task) -> None:
    """Retrieve a run's unexpected exception so asyncio doesn't report it as never retrieved."""
    if run.cancelled():
        return
    error = run.exception()
    if error is not None:
        logger.error(f"{run.get_name()} ended with an unhandled error", exc_info=error)


class RunSupervisor:
    """Starts runs in the background and enforces settings.TASK_MAX_SECONDS."""

    def __init__(self, on_stop: Optional[OnStop] = None):
        self._runs: Dict[str, asyncio.Task] = {}
        # Holds the watcher tasks so they aren't garbage-collected while running.
        self._watchers: Set[asyncio.Task] = set()
        self._on_stop = on_stop

    def start(self, task_id: str, coro: Awaitable[None]) -> None:
        """Run `coro` as a task and watch it; returns at once. Needs a running event loop."""
        run = asyncio.create_task(coro, name=f"run-{task_id}")
        run.add_done_callback(_log_crash)
        self._runs[task_id] = run
        watcher = asyncio.create_task(self._watch(task_id, run), name=f"watch-{task_id}")
        self._watchers.add(watcher)
        watcher.add_done_callback(self._watchers.discard)

    def is_running(self, task_id: str) -> bool:
        run = self._runs.get(task_id)
        return run is not None and not run.done()

    async def _watch(self, task_id: str, run: asyncio.Task) -> None:
        deadline = time.monotonic() + settings.TASK_MAX_SECONDS
        try:
            while not run.done():
                await asyncio.wait({run}, timeout=POLL_SECONDS)
                if run.done():
                    break
                reason = self._stop_reason(task_id, deadline)
                if reason is not None:
                    await self._stop(task_id, run, *reason)
                    break
        except Exception:
            logger.exception(f"Watcher for task {task_id} failed; the run continues unwatched")
        finally:
            self._runs.pop(task_id, None)

    def _stop_reason(self, task_id: str, deadline: float) -> Optional[Tuple[str, str]]:
        """(error_code, message) if the run should stop now, else None."""
        if time.monotonic() >= deadline:
            return "timeout", TIMEOUT_MESSAGE
        return None

    async def _stop(self, task_id: str, run: asyncio.Task, code: str, message: str) -> None:
        logger.warning(f"Stopping task {task_id}: {code}")
        run.cancel()
        # Saved before the run unwinds, so a late "completed" save from the run can't win.
        result_store.save_final(
            task_id,
            status="error",
            error_code=code,
            error=message,
            result={"error": message, "error_code": code},
            completed_at=datetime.now(),
        )
        if self._on_stop is not None:
            try:
                self._on_stop(task_id, code, message)
            except Exception:
                logger.exception(f"on_stop callback failed for task {task_id}")
        await asyncio.wait({run}, timeout=STOP_GRACE_SECONDS)
