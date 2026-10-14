"""Save finished task results to disk so they survive server restarts."""
import json
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from app.core.config import settings

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(settings.DATA_DIR) / "results"

# A record with one of these codes was written when the run was stopped; it is final.
STOP_CODES = {"timeout", "cancelled", "budget"}
_KEPT_KEYS = ("task_id", "query", "created_at")


def _result_path(task_id: str) -> Optional[Path]:
    # Task IDs arrive from the URL; only accept real UUIDs so they can't point outside RESULTS_DIR.
    try:
        return RESULTS_DIR / f"{uuid.UUID(task_id)}.json"
    except (ValueError, AttributeError, TypeError):
        return None


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
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        logger.error(f"Could not read saved result for task {task_id}: {e}")
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
