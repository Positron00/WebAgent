"""Load and validate ContextDefender JSONL rows (train/eval scripts)."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Set

logger = logging.getLogger(__name__)

AXES = ("injection", "harm", "sensitive_data", "jailbreak")
LABEL_LEVELS = ("none", "low", "high")
CONTEXT_SOURCES = (
    "user_turn",
    "history",
    "agent_output",
    "web_result",
    "retrieved_doc",
    "document",
    "tool_output",
    "unlabelled",
)
# Axis/source combinations that occur in practice and must have training positives (SEC-027).
# Others (harm inside code output, say) are rare, and requiring them would force artificial rows.
REQUIRED_CELLS = {
    "injection": ("user_turn", "history", "agent_output", "web_result", "retrieved_doc", "document",
                  "tool_output", "unlabelled"),
    "harm": ("user_turn", "history"),
    "jailbreak": ("user_turn", "history", "web_result"),
    "sensitive_data": ("user_turn", "history", "document", "retrieved_doc", "tool_output"),
}
UNTRUSTED_SOURCES = frozenset(
    {"web_result", "retrieved_doc", "document", "tool_output", "unlabelled"}
)
REQUIRED_KEYS = {"text", "source", "labels", "origin"}


def _validate_labels(labels: Any, path: Path, line_no: int) -> None:
    if not isinstance(labels, dict):
        raise ValueError(f"{path}:{line_no} labels must be an object")
    for axis in AXES:
        level = labels.get(axis)
        if level not in LABEL_LEVELS:
            raise ValueError(f"{path}:{line_no} bad label {axis}={level!r}")


def load_rows(path: Path) -> List[Dict[str, Any]]:
    """Read curated rows; invalid JSON lines are skipped with a warning."""
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
            if row["source"] not in CONTEXT_SOURCES:
                raise ValueError(f"{path}:{line_no} bad source {row['source']!r}")
            if not isinstance(row["text"], str) or not row["text"].strip():
                raise ValueError(f"{path}:{line_no} empty text")
            if not isinstance(row["origin"], str) or not row["origin"].strip():
                raise ValueError(f"{path}:{line_no} empty origin")
            _validate_labels(row["labels"], path, line_no)
            rows.append(row)
    return rows


def label_index(level: str) -> int:
    return LABEL_LEVELS.index(level)


def axis_level(row: Dict[str, Any], axis: str) -> str:
    return row["labels"][axis]


def is_benign_row(row: Dict[str, Any]) -> bool:
    return all(row["labels"][a] == "none" for a in AXES)


# Origins of sensitive-data rows by kind (`<dataset>:<role>`; the bare prefixes are older names).
SECRET_ORIGINS = ("generated:secret", "secret")
PERSONAL_ORIGINS = ("nemotron:pii", "personal")


# Benign sets written to look harmful (XSTest, JBB, Aegis safe prompts) or like jailbreaks
# (TrustAIRLab role-play prompts). The 1% false-positive limits apply to ordinary questions;
# these are reported separately, and the LLM judge settles them at runtime (SEC-028, SEC-035a).
LOOKALIKE_ORIGINS = {
    "xstest:safe": ("harm",),
    "jbb:benign": ("harm",),
    "aegis:safe": ("harm",),
    "aegis:soft": ("harm",),  # Aegis "unsafe" in soft categories, relabelled benign (build_security_dataset.py)
    # NotInject is built to test over-blocking: benign prompts packed with trigger words (2026-10-12).
    "notinject:benign": ("harm", "jailbreak", "injection"),
    # Benign text full of instructions to the reader (weighted on injection when training).
    "cosmopedia:howto": ("injection",),
    "cosmopedia:web": ("injection",),
    "oasst:assistant": ("injection",),
    "trustairlab:regular": ("jailbreak",),
}


def is_ordinary_question(row: Dict[str, Any]) -> bool:
    return row["source"] == "user_turn" and is_benign_row(row) and row["origin"] not in LOOKALIKE_ORIGINS


def is_lookalike_question(row: Dict[str, Any]) -> bool:
    return row["source"] == "user_turn" and is_benign_row(row) and row["origin"] in LOOKALIKE_ORIGINS


def is_secret_row(row: Dict[str, Any]) -> bool:
    return row["labels"]["sensitive_data"] in ("low", "high") and row["origin"].startswith(SECRET_ORIGINS)


def is_personal_row(row: Dict[str, Any]) -> bool:
    return row["labels"]["sensitive_data"] in ("low", "high") and row["origin"].startswith(PERSONAL_ORIGINS)


def sources_in_rows(rows: List[Dict[str, Any]]) -> Set[str]:
    return {r["source"] for r in rows}
