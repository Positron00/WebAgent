"""Files of recurring research (recurring-research spec REC-101).

One writer per file:
- the backend writes topic definitions (`topics/<id>.json`) and the email ledger (`notify/`), always
  read-modify-write under `locked()` because it runs several workers;
- the scheduler (one process) writes `topics/<id>/state.json`, `topics/<id>/runs/*.json`,
  `topics/usage-<YYYY-MM>.json`, `topics/counters.json` and `topics/.scheduler.lock`.
The one exception is REC-114: with no live scheduler, the backend removes a deleted topic's files
while holding the scheduler lock.

This module imports nothing heavy: the scheduler process uses it (RNFR-02).
"""
from __future__ import annotations

import fcntl
import json
import os
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from app.core.config import settings

TOPIC_NAMESPACE = uuid.UUID("6f1d3b0e-6c1b-4f7e-9a51-2b7a1c0d9e11")  # run_id -> task_id (REC-106)


def topics_dir() -> Path:
    return Path(settings.DATA_DIR) / "topics"


def notify_dir() -> Path:
    return Path(settings.DATA_DIR) / "notify"


def is_uuid(value: Any) -> bool:
    try:
        return str(uuid.UUID(str(value))) == str(value).lower()
    except (ValueError, TypeError, AttributeError):
        return False


def _checked(value: str) -> str:
    if not is_uuid(value):
        raise ValueError("not a UUID")
    return str(value).lower()


def definition_path(topic_id: str) -> Path:
    return topics_dir() / f"{_checked(topic_id)}.json"


def topic_dir(topic_id: str) -> Path:
    return topics_dir() / _checked(topic_id)


def state_path(topic_id: str) -> Path:
    return topic_dir(topic_id) / "state.json"


def runs_dir(topic_id: str) -> Path:
    return topic_dir(topic_id) / "runs"


def run_path(topic_id: str, run_id: str) -> Path:
    return runs_dir(topic_id) / f"{_checked(run_id)}.json"


def usage_path(month: Optional[str] = None) -> Path:
    month = month or datetime.now(timezone.utc).strftime("%Y-%m")
    return topics_dir() / f"usage-{month}.json"


def counters_path() -> Path:
    return topics_dir() / "counters.json"


def scheduler_lock_path() -> Path:
    return topics_dir() / ".scheduler.lock"


def task_id_for_run(run_id: str) -> str:
    """The fixed task ID of a scheduled run (REC-106)."""
    return str(uuid.uuid5(TOPIC_NAMESPACE, _checked(run_id)))


def write_json_atomic(path: Path, data: Dict[str, Any]) -> None:
    """Write `data` as JSON through a temp file and a rename, so readers never see half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


@contextmanager
def locked(path: Path) -> Iterator[None]:
    """Exclusive flock on `<path>.lock`, for the backend's read-modify-write of `path` (REC-101)."""
    lock = path.with_name(path.name + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    with open(lock, "a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def list_topic_ids() -> List[str]:
    folder = topics_dir()
    if not folder.is_dir():
        return []
    return sorted(p.stem for p in folder.glob("*.json") if is_uuid(p.stem))


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def parse_time(value: Any) -> Optional[datetime]:
    """An aware UTC datetime from an ISO string (or a datetime), else None."""
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
