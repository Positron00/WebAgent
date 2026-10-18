"""Topics: models, validation, reading and the backend's definition writes (recurring-research spec
REC-100 – REC-102, REC-114).

The scheduler imports this module for the models and the read helpers only; it never calls the
`*_definition` writers (REC-101). Keep imports light (RNFR-02).
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterator, List, Literal, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, Field, field_validator, model_validator

from app.services import topic_files as files

logger = logging.getLogger(__name__)

_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_EMAIL = re.compile(r"^[^@\s]{1,64}@[^@\s]+\.[^@\s]+$")

Weekday = Literal["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
EffectiveState = Literal["active", "paused", "held_failures", "held_budget", "deleted"]


def _no_control(value: str, field: str) -> str:
    value = value.strip()
    if _CONTROL.search(value):
        raise ValueError(f"{field} must not contain control characters")
    return value


class Schedule(BaseModel):
    every: Literal["day", "week"]
    weekday: Optional[Weekday] = None
    time: str = Field(pattern=r"^([01]\d|2[0-3]):[0-5]\d$")
    tz: str

    @field_validator("tz")
    @classmethod
    def _zone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError):
            raise ValueError("unknown time zone") from None
        return value

    @model_validator(mode="after")
    def _weekday(self) -> "Schedule":
        if self.every == "week" and not self.weekday:
            raise ValueError("a weekly schedule needs a weekday")
        if self.every == "day":
            self.weekday = None
        return self


class Subscriber(BaseModel):
    email: str
    added_at: datetime
    confirmed_at: Optional[datetime] = None

    @field_validator("email")
    @classmethod
    def _email(cls, value: str) -> str:
        value = value.strip().lower()
        if len(value) > 254 or not _EMAIL.match(value) or _CONTROL.search(value):
            raise ValueError("not a valid email address")
        return value


class Notify(BaseModel):
    when: Literal["changes", "always"] = "changes"
    subscribers: List[Subscriber] = Field(default_factory=list)


class TopicDefinition(BaseModel):
    topic_id: str
    name: str = Field(min_length=1, max_length=80)
    query: str = Field(min_length=1, max_length=2000)
    schedule: Schedule
    notify: Notify = Field(default_factory=Notify)
    status: Literal["active", "paused", "deleted"] = "active"
    created_at: datetime
    created_from_task: Optional[str] = None
    run_requests: int = 0
    resumes: int = 0

    @field_validator("name")
    @classmethod
    def _name(cls, value: str) -> str:
        return _no_control(value, "name")

    @field_validator("query")
    @classmethod
    def _query(cls, value: str) -> str:
        return _no_control(value, "query")

    @field_validator("topic_id", "created_from_task")
    @classmethod
    def _uuid(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and not files.is_uuid(value):
            raise ValueError("must be a UUID")
        return value


def new_topic_id() -> str:
    return str(uuid.uuid4())


# --- reading (backend and scheduler) ------------------------------------------------------------

def load_definition(topic_id: str) -> Optional[TopicDefinition]:
    data = files.read_json(files.definition_path(topic_id))
    if data is None:
        return None
    try:
        return TopicDefinition.model_validate(data)
    except ValueError:
        return None


def list_definitions(include_deleted: bool = False) -> List[TopicDefinition]:
    out = []
    for topic_id in files.list_topic_ids():
        defn = load_definition(topic_id)
        if defn is not None and (include_deleted or defn.status != "deleted"):
            out.append(defn)
    return out


def load_state(topic_id: str) -> Dict[str, Any]:
    return files.read_json(files.state_path(topic_id)) or {}


def effective_state(defn: TopicDefinition, state: Optional[Dict[str, Any]] = None) -> EffectiveState:
    """The state shown to people: the definition's status first, then the scheduler's hold."""
    if defn.status in ("deleted", "paused"):
        return defn.status
    hold = (state or {}).get("hold")
    if hold == "failures":
        return "held_failures"
    if hold == "budget":
        return "held_budget"
    return "active"


def active_count(exclude: Optional[str] = None) -> int:
    return sum(1 for d in list_definitions() if d.status == "active" and d.topic_id != exclude)


# --- writing (backend only, REC-101) -------------------------------------------------------------

def save_new_definition(defn: TopicDefinition) -> None:
    path = files.definition_path(defn.topic_id)
    with files.locked(path):
        files.write_json_atomic(path, defn.model_dump(mode="json"))


def update_definition(topic_id: str, change: Callable[[TopicDefinition], None]) -> Optional[TopicDefinition]:
    """Read, apply `change`, validate and write, under the definition's lock. None if missing."""
    path = files.definition_path(topic_id)
    with files.locked(path):
        defn = load_definition(topic_id)
        if defn is None:
            return None
        change(defn)
        defn = TopicDefinition.model_validate(defn.model_dump())
        files.write_json_atomic(path, defn.model_dump(mode="json"))
        return defn


# --- run records (read-only here; the scheduler writes them) -------------------------------------

def _run_sort_key(record: Dict[str, Any]) -> Any:
    moment = files.parse_time(record.get("started_at")) or files.parse_time(record.get("completed_at"))
    return moment or datetime.min.replace(tzinfo=timezone.utc)


def list_runs(topic_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """The topic's run records, newest first (by `started_at`, then `completed_at`)."""
    folder = files.runs_dir(topic_id)
    if not folder.is_dir():
        return []
    records = []
    for path in folder.glob("*.json"):
        if not files.is_uuid(path.stem):
            continue
        record = files.read_json(path)
        if record is not None:
            record.setdefault("run_id", path.stem)
            records.append(record)
    records.sort(key=_run_sort_key, reverse=True)
    return records[:limit] if limit is not None else records


def load_run(topic_id: str, run_id: str) -> Optional[Dict[str, Any]]:
    if not files.is_uuid(run_id):
        return None
    return files.read_json(files.run_path(topic_id, run_id))


def new_source_count(record: Optional[Dict[str, Any]]) -> Optional[int]:
    """How many sources a run found that the last runs didn't (REC-111), or None."""
    if not record:
        return None
    new = (record.get("change") or {}).get("new_sources")
    if isinstance(new, list):
        return len(new)
    if isinstance(new, int) and not isinstance(new, bool):
        return new
    return None


# --- deletion and retention (REC-114) ------------------------------------------------------------

SCHEDULER_STALE_SECONDS = 180


def scheduler_alive(now: Optional[datetime] = None) -> bool:
    """True if `.scheduler.lock` holds a heartbeat younger than 180 s (from anyone, backend too)."""
    lock = files.read_json(files.scheduler_lock_path())
    if not lock:
        return False
    beat = files.parse_time(lock.get("heartbeat"))
    if beat is None:
        return False
    now = now or files.utcnow()
    return (now - beat).total_seconds() < SCHEDULER_STALE_SECONDS


def _run_task_ids(topic_id: str) -> List[str]:
    task_ids = []
    for record in list_runs(topic_id):
        task_id = record.get("task_id")
        if not files.is_uuid(task_id) and files.is_uuid(record.get("run_id")):
            task_id = files.task_id_for_run(record["run_id"])
        if files.is_uuid(task_id):
            task_ids.append(str(task_id))
    return task_ids


def _remove_topic_files(topic_id: str) -> None:
    """The scheduler's part of a delete, done by the backend while it holds the lock (REC-114)."""
    from app.services import result_store

    for task_id in _run_task_ids(topic_id):
        result_store.delete_result(task_id)
    shutil.rmtree(files.topic_dir(topic_id), ignore_errors=True)


def _remove_definition(topic_id: str) -> bool:
    """Delete a tombstoned definition and its lock file. False if it isn't (or no longer) deleted."""
    path = files.definition_path(topic_id)
    with files.locked(path):
        defn = load_definition(topic_id)
        if defn is not None and defn.status != "deleted":
            return False
        path.unlink(missing_ok=True)
        # The topic is gone for good, so no writer will want this lock again.
        path.with_name(path.name + ".lock").unlink(missing_ok=True)
    return True


@contextmanager
def _backend_owns_scheduler_lock(now: datetime) -> Iterator[bool]:
    """Take `.scheduler.lock` as `owner: backend` if no scheduler is live. Yields whether it did.

    The flock on the lock file's own lock keeps the two backend workers from both taking over;
    the heartbeat written as `owner: backend` makes a starting scheduler wait (REC-114).
    """
    lock_path = files.scheduler_lock_path()
    with files.locked(lock_path):
        if scheduler_alive(now):
            yield False
            return
        mine = {"pid": os.getpid(), "heartbeat": now.isoformat(), "enabled": False, "owner": "backend"}
        files.write_json_atomic(lock_path, mine)
        try:
            yield True
        finally:
            current = files.read_json(lock_path) or {}
            if current.get("owner") == "backend" and current.get("pid") == os.getpid():
                lock_path.unlink(missing_ok=True)


def cleanup_deleted(now: Optional[datetime] = None, topic_ids: Optional[List[str]] = None) -> int:
    """Finish deleting tombstoned topics (REC-114). Returns how many definitions were removed.

    A topic whose folder is gone loses its definition. A topic whose folder remains is left to a
    live scheduler; with none, the backend removes the runs' results and the folder itself under
    the scheduler lock, then the definition. Never raises.
    """
    now = now or files.utcnow()
    removed = 0
    try:
        candidates = [
            d.topic_id for d in list_definitions(include_deleted=True)
            if d.status == "deleted" and (topic_ids is None or d.topic_id in topic_ids)
        ]
        if not candidates:
            return 0
        leftover = []
        for topic_id in candidates:
            if files.topic_dir(topic_id).exists():
                leftover.append(topic_id)
            elif _remove_definition(topic_id):
                _forget_ledger(topic_id)
                removed += 1
        if leftover:
            with _backend_owns_scheduler_lock(now) as owned:
                if owned:
                    for topic_id in leftover:
                        defn = load_definition(topic_id)
                        if defn is None or defn.status != "deleted":
                            continue
                        _remove_topic_files(topic_id)
                        if _remove_definition(topic_id):
                            _forget_ledger(topic_id)
                            removed += 1
    except Exception as exc:  # noqa: BLE001 - a cleanup must never take a worker down
        logger.warning("Topic cleanup failed: %s", type(exc).__name__)
    if removed:
        logger.info("Topic cleanup removed %d deleted topics", removed)
    return removed


def _forget_ledger(topic_id: str) -> None:
    """Remove a deleted topic's entries from the email ledgers (REC-114, phase B). Never raises."""
    try:
        from app.services import notify_email   # lazy: the scheduler imports this module (RNFR-02)

        notify_email.forget_topic(topic_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Ledger cleanup for topic %s failed: %s", topic_id, type(exc).__name__)


def expire_unconfirmed(now: Optional[datetime] = None) -> int:
    """Remove unconfirmed subscribers added more than 48 h ago (REC-115). Returns how many. Never raises."""
    now = now or files.utcnow()
    removed = 0
    try:
        from app.services.notify_email import expired_unconfirmed   # lazy (RNFR-02)

        for defn in list_definitions():
            if not any(expired_unconfirmed(s, now) for s in defn.notify.subscribers):
                continue
            dropped: List[int] = []

            def drop(d: TopicDefinition) -> None:
                kept = [s for s in d.notify.subscribers if not expired_unconfirmed(s, now)]
                dropped.append(len(d.notify.subscribers) - len(kept))
                d.notify.subscribers = kept

            update_definition(defn.topic_id, drop)
            removed += sum(dropped)
    except Exception as exc:  # noqa: BLE001 - a cleanup must never take a worker down
        logger.warning("Subscriber expiry failed: %s", type(exc).__name__)
    if removed:
        logger.info("Subscriber expiry removed %d unconfirmed entries", removed)
    return removed


def _cleanup_notify(now: Optional[datetime] = None) -> None:
    """Phase B's part of the daily cleanup: unconfirmed subscribers and old email ledgers. Never raises."""
    expire_unconfirmed(now)
    try:
        from app.services import notify_email

        notify_email.cleanup_ledgers(now)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Ledger retention cleanup failed: %s", type(exc).__name__)


async def cleanup_daily(interval_sec: float = 24 * 3600) -> None:
    """Run `cleanup_deleted` now and then once a day, as `result_store.cleanup_daily` (REC-114), with
    the expiry of unconfirmed subscribers and the email ledgers' retention (REC-115, phase B)."""
    while True:
        await asyncio.to_thread(cleanup_deleted)
        await asyncio.to_thread(_cleanup_notify)
        await asyncio.sleep(interval_sec)
