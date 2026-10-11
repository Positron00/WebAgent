"""Load and validate triage JSONL rows (shared by train_triage.py and eval_triage.py)."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

REQUIRED_KEYS = {"text", "label", "category", "source"}
VALID_LABELS = {"agentic", "direct"}
VALID_SOURCES = {"seed", "generated", "collected"}


def load_rows(path: Path) -> List[Dict[str, Any]]:
    """Read curated rows. A line that isn't valid JSON is skipped with a warning;
    a row with missing keys or a bad label/source raises ValueError."""
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("%s:%d is not valid JSON; skipped", path, line_no)
                continue
            missing = REQUIRED_KEYS - row.keys()
            if missing:
                raise ValueError(f"{path}:{line_no} missing keys {sorted(missing)}")
            if row["label"] not in VALID_LABELS:
                raise ValueError(f"{path}:{line_no} bad label {row['label']!r}")
            if row["source"] not in VALID_SOURCES:
                raise ValueError(f"{path}:{line_no} bad source {row['source']!r}")
            if not isinstance(row["text"], str) or not row["text"].strip():
                raise ValueError(f"{path}:{line_no} empty text")
            rows.append(row)
    return rows


def load_categories(path: Path) -> Dict[str, Dict[str, Any]]:
    """Category id -> {"label", "class", "hard_case"} from categories.json."""
    return {c["id"]: c for c in json.loads(path.read_text(encoding="utf-8"))}
