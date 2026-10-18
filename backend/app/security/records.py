"""Security collection, escalation queue, and retention cleanup (SEC-060, SEC-064)."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.core.metrics import GUARD_ESCALATIONS
from app.security.types import Decision, ToolAction
from app.services import run_origin

logger = logging.getLogger(__name__)
_lock = threading.Lock()


def _security_root() -> Path:
    return Path(settings.DATA_DIR) / "security"


def _collected_dir() -> Path:
    return _security_root() / "collected"


def _escalations_dir() -> Path:
    return _security_root() / "escalations"


def _tokens_dir() -> Path:
    return _security_root() / "tokens"


def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, ensure_ascii=False) + "\n"
    with _lock:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(fd, line.encode("utf-8"))
        finally:
            os.close(fd)


def _text_field(text: Optional[str]) -> Dict[str, Any]:
    if not text:
        return {}
    if settings.WEBAGENT_ENV == "dev":
        return {"text": text[:2000]}
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return {"text_sha256": digest}


def record_collection(
    *,
    checkpoint: int,
    action: str,
    decision: Decision,
    state_context: Dict[str, Any],
    args: Optional[Dict[str, Any]] = None,
    item: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None,
) -> None:
    """One collected decision. `item` (checkpoint 1) and `context` (checkpoint 2) carry scores and
    features for the shadow-week review; text only in dev (SNFR-09)."""
    cfg = getattr(settings, "SECURITY_GUARD", None)
    if cfg is None or not cfg.collect:
        return
    if decision.outcome == "allow" and not decision.shadow and not decision.shadow_policy_ids:
        return
    row = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "checkpoint": checkpoint,
        "action": action,
        "outcome": decision.outcome,
        "policy_ids": list(decision.policy_ids),
        "shadow_policy_ids": list(decision.shadow_policy_ids),
        "shadow": decision.shadow,
        "task_id": state_context.get("task_id"),
        "args": args or {},
        **_text_field(state_context.get("query")),
    }
    topic_id = run_origin.topic_run.get()
    if topic_id:
        row["origin"] = "schedule"
        row["topic_id"] = topic_id
    elif state_context.get("origin") == "topic_create":
        # The intake check when a topic is created (recurring-research spec REC-103).
        row["origin"] = "topic_create"
        row["topic_id"] = state_context.get("topic_id")
    if item is not None:
        row["item"] = item
    if context is not None:
        row["context"] = context
    try:
        path = _collected_dir() / f"{datetime.now(timezone.utc).date().isoformat()}.jsonl"
        _append_jsonl(path, row)
    except Exception as exc:
        logger.warning("Failed to record security collection row: %s", exc)


def record_escalation(
    *,
    hop: Optional[str],
    action: str,
    decision: Decision,
    state_context: Dict[str, Any],
    args: Optional[Dict[str, Any]] = None,
    scores: Optional[Dict[str, Any]] = None,
) -> None:
    row = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "task_id": state_context.get("task_id"),
        "hop": hop,
        "action": action,
        "policy_ids": list(decision.policy_ids),
        "reason": decision.reason,
        "args": args or {},
        "scores": scores or {},
        **_text_field(state_context.get("query")),
    }
    try:
        path = _escalations_dir() / f"{datetime.now(timezone.utc).date().isoformat()}.jsonl"
        _append_jsonl(path, row)
        GUARD_ESCALATIONS.inc()
    except Exception as exc:
        logger.warning("Failed to record security escalation: %s", exc)


def _cleanup_dir(directory: Path, retention_days: int, now: Optional[datetime] = None) -> int:
    cutoff = (now or datetime.now(timezone.utc)) - timedelta(days=retention_days)
    deleted = 0
    if not directory.is_dir():
        return 0
    for path in directory.glob("*.jsonl"):
        try:
            file_date = datetime.strptime(path.stem, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        if file_date < cutoff:
            try:
                path.unlink(missing_ok=True)
                deleted += 1
            except OSError as exc:
                logger.warning("Could not delete old security file %s: %s", path.name, exc)
    return deleted


def cleanup(now: Optional[datetime] = None) -> int:
    """Delete collected, escalation and token files past retention. Returns total deleted."""
    cfg = getattr(settings, "SECURITY_GUARD", None)
    retention = cfg.collect_retention_days if cfg else 30
    total = 0
    total += _cleanup_dir(_collected_dir(), retention, now)
    total += _cleanup_dir(_escalations_dir(), retention, now)
    total += _cleanup_token_claims(now)
    return total


TOKEN_CLAIM_RETENTION = timedelta(hours=1)  # tokens expire after 15 minutes; claims only need to outlive them


def _cleanup_token_claims(now: Optional[datetime] = None) -> int:
    """Delete used-token claim files (named by token id) older than the token lifetime (SEC-053)."""
    directory = _tokens_dir()
    if not directory.is_dir():
        return 0
    cutoff = ((now or datetime.now(timezone.utc)) - TOKEN_CLAIM_RETENTION).timestamp()
    deleted = 0
    for path in directory.iterdir():
        try:
            if path.is_file() and path.stat().st_mtime < cutoff:
                path.unlink(missing_ok=True)
                deleted += 1
        except OSError as exc:
            logger.warning("Could not delete token claim %s: %s", path.name, exc)
    return deleted


async def cleanup_daily(interval_sec: float = 24 * 3600) -> None:
    while True:
        await asyncio.sleep(interval_sec)
        try:
            cleanup()
        except Exception as exc:
            logger.warning("Daily security cleanup failed: %s", exc)
