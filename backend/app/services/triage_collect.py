"""Append shadow/override triage rows for later review."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, List, Literal, Optional

from app.core.config import settings
from app.services.triage import TriageDecision

logger = logging.getLogger(__name__)
_lock = threading.Lock()
Kind = Literal["shadow", "override"]


def _collect_dir() -> Path:
    return Path(settings.DATA_DIR) / "triage" / "collected"


def record(
    kind: Kind,
    query: str,
    decision: Optional[TriageDecision],
    previous_user_question: Optional[str],
) -> None:
    cfg = getattr(settings, "TRIAGE", None)
    if cfg is None or not cfg.collect:
        return
    row = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "kind": kind,
        "text": query[:2000],
        "previous_user_question": (previous_user_question or "")[:1000] or None,
        "route": decision.route if decision else "agentic",
        "p_agentic": decision.p_agentic if decision else None,
        "decided_by": decision.decided_by if decision else "override",
        "classifier_version": decision.classifier_version if decision else "none",
    }
    try:
        d = _collect_dir()
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{datetime.now(timezone.utc).date().isoformat()}.jsonl"
        line = json.dumps(row, ensure_ascii=False) + "\n"
        with _lock:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
            try:
                os.write(fd, line.encode("utf-8"))
            finally:
                os.close(fd)
    except Exception as exc:
        logger.warning("Failed to record triage collection row: %s", exc)


def cleanup(now: Optional[datetime] = None) -> int:
    """Delete collected files older than the retention period. Returns how many were deleted."""
    cfg = getattr(settings, "TRIAGE", None)
    if cfg is None:
        return 0
    cutoff = (now or datetime.now(timezone.utc)) - timedelta(days=cfg.collect_retention_days)
    deleted = 0
    d = _collect_dir()
    if not d.is_dir():
        return 0
    for path in d.glob("*.jsonl"):
        try:
            file_date = datetime.strptime(path.stem, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        if file_date < cutoff:
            try:
                path.unlink(missing_ok=True)
                deleted += 1
            except OSError as exc:
                logger.warning("Could not delete old triage file %s: %s", path.name, exc)
    return deleted


async def cleanup_daily(interval_sec: float = 24 * 3600) -> None:
    """Run `cleanup` once a day for as long as the server runs."""
    while True:
        await asyncio.sleep(interval_sec)
        try:
            cleanup()
        except Exception as exc:
            logger.warning("Daily triage cleanup failed: %s", exc)
