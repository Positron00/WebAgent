"""Save task results to disk so they survive restarts and every worker sees them.

Each task has `<id>.json` (the record), written atomically through `<id>.json.tmp`, and
may have an empty `<id>.cancel` marker that asks the worker running it to stop.
"""
import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.core.log_context import describe_text

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(settings.DATA_DIR) / "results"

# A record with one of these codes was written when the run was stopped; it is final.
STOP_CODES = {"timeout", "cancelled", "budget"}
_KEPT_KEYS = ("task_id", "query", "created_at")
_SUFFIXES = (".json", ".json.tmp", ".cancel")


def _task_path(task_id: str, suffix: str) -> Optional[Path]:
    # Task IDs arrive from the URL; only accept real UUIDs so they can't point outside RESULTS_DIR.
    try:
        return RESULTS_DIR / f"{uuid.UUID(task_id)}{suffix}"
    except (ValueError, AttributeError, TypeError):
        return None


def _result_path(task_id: str) -> Optional[Path]:
    return _task_path(task_id, ".json")


def save_result(task_id: str, record: Dict[str, Any]) -> Optional[Path]:
    """Write one task's record as JSON. Returns the file path, or None if saving failed."""
    path = _result_path(task_id)
    if path is None:
        logger.warning(f"Not saving result for invalid task id: {task_id!r}")
        return None
    try:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
        tmp.replace(path)
        logger.info(f"Saved result for task {task_id} to {path}")
        return path
    except OSError as e:
        logger.error(f"Could not save result for task {task_id}: {e}")
        return None


def load_result(task_id: str) -> Optional[Dict[str, Any]]:
    """Read a saved task record, or None if there isn't one."""
    path = _result_path(task_id)
    if path is None or not path.is_file():
        return None
    return _read(path)


def _read(path: Path) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError) as e:
        logger.error(f"Could not read saved result {path.name}: {e}")
        return None


def save_final(task_id: str, /, **record: Any) -> bool:
    """Write a task's final record unless a stop (timeout, cancel, budget) already wrote one.

    Workers don't share memory, so the check reads the file. Fields the record doesn't give
    (task_id, query, created_at) are kept from the saved file. Returns whether it wrote.
    """
    saved = load_result(task_id) or {}
    if saved.get("error_code") in STOP_CODES:
        logger.info(f"Task {task_id} was already stopped ({saved['error_code']}); not overwriting")
        return False
    merged = {key: saved[key] for key in _KEPT_KEYS if key in saved}
    merged.update(record)
    merged.setdefault("task_id", task_id)
    return save_result(task_id, merged) is not None


def save_progress(task_id: str, progress: Dict[str, Any]) -> bool:
    """Add `progress` to a processing record. Does nothing once the record is final or gone."""
    saved = load_result(task_id)
    if saved is None or saved.get("status") != "processing":
        return False
    saved["progress"] = progress
    return save_result(task_id, saved) is not None


def request_cancel(task_id: str) -> bool:
    """Create the task's cancel marker. Returns whether it exists afterwards."""
    path = _task_path(task_id, ".cancel")
    if path is None:
        return False
    try:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        path.touch()
        return True
    except OSError as e:
        logger.error(f"Could not write cancel marker for task {task_id}: {e}")
        return False


def cancel_requested(task_id: str) -> bool:
    path = _task_path(task_id, ".cancel")
    return path is not None and path.exists()


def delete_result(task_id: str) -> bool:
    """Remove a task's record, temp file and cancel marker. Returns whether anything was removed."""
    removed = False
    for suffix in _SUFFIXES:
        path = _task_path(task_id, suffix)
        if path is None:
            return False
        try:
            path.unlink()
            removed = True
        except FileNotFoundError:
            pass
        except OSError as e:
            logger.warning(f"Could not delete {path.name}: {e}")
    return removed


def _created_at(record: Optional[Dict[str, Any]], path: Path) -> datetime:
    """The record's created_at, or the file's modified time if that can't be read."""
    try:
        created = datetime.fromisoformat(str((record or {})["created_at"]))
        return created.replace(tzinfo=None)
    except (KeyError, ValueError, TypeError):
        return datetime.fromtimestamp(path.stat().st_mtime)


def _record_files() -> List[Path]:
    if not RESULTS_DIR.is_dir():
        return []
    return list(RESULTS_DIR.glob("*.json"))


def list_results(limit: int = 20, offset: int = 0) -> List[Dict[str, Any]]:
    """Saved tasks, newest first, with only the TaskInfo fields and the question's length."""
    rows = []
    for path in _record_files():
        record = _read(path)
        if record is None:
            continue
        try:
            created = _created_at(record, path)
        except OSError:
            continue
        rows.append((created, path.stem, record))
    rows.sort(key=lambda row: row[0], reverse=True)
    return [
        {
            "task_id": record.get("task_id") or stem,
            "status": record.get("status", "completed"),
            "created_at": created,
            "completed_at": record.get("completed_at"),
            "query": describe_text(record.get("query")),
        }
        for created, stem, record in rows[offset:offset + limit]
    ]


def cleanup(retention_days: int, now: Optional[datetime] = None) -> int:
    """Delete records (and their temp files and markers) older than `retention_days`.

    Also deletes markers whose record is gone and that are older than the cutoff.
    Returns how many files were removed. Never raises.
    """
    cutoff = (now or datetime.now()) - timedelta(days=retention_days)
    removed = 0
    for path in _record_files():
        try:
            if _created_at(_read(path), path) < cutoff:
                for suffix in _SUFFIXES:
                    sibling = RESULTS_DIR / f"{path.stem}{suffix}"
                    if sibling.exists():
                        sibling.unlink(missing_ok=True)
                        removed += 1
        except Exception as e:
            logger.warning(f"Could not clean up {path.name}: {e}")
    if RESULTS_DIR.is_dir():
        for marker in RESULTS_DIR.glob("*.cancel"):
            try:
                stale = datetime.fromtimestamp(marker.stat().st_mtime) < cutoff
                if stale and not marker.with_suffix(".json").exists():
                    marker.unlink(missing_ok=True)
                    removed += 1
            except Exception as e:
                logger.warning(f"Could not clean up {marker.name}: {e}")
    if removed:
        logger.info(f"Results cleanup removed {removed} files older than {retention_days} days")
    return removed


def _cleanup_safely() -> None:
    try:
        cleanup(settings.RESULTS_RETENTION_DAYS)
    except Exception as e:
        logger.warning(f"Results cleanup failed: {e}")


async def cleanup_daily(interval_sec: float = 24 * 3600) -> None:
    """Run `cleanup` now and then once a day for as long as the server runs."""
    while True:
        _cleanup_safely()
        await asyncio.sleep(interval_sec)
