"""Signals that check a call or subgoal against the user's intent (checkpoint2-intent spec).

SEC-103 – SEC-105, SEC-110: one base MiniLM-L6 encode of the text gives both its alignment with the
intent and with the task text (the existing ``alignment``); ContextDefender's harm score runs at
the same time on another thread, so checkpoint 2 costs about the slower of the two. Any failure
leaves defaults that match no rule and ``available=False``.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from app.models.task import WorkflowState
from app.security import alignment, ledger

logger = logging.getLogger(__name__)

HARM_TIMEOUT_SEC = 2.0
# Arguments and the intent are scored on the same ContextDefender route, so their harm scores come
# from one calibrated head and can be compared (risk above intent).
HARM_SOURCE = "user_turn"

_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="guard-signals")
_warned: set = set()


@dataclass(frozen=True)
class TextSignals:
    intent_alignment: int = 100
    task_alignment: int = 100
    harm: int = 0
    available: bool = True


def _warn_once(kind: str, exc: BaseException) -> None:
    if kind not in _warned:
        _warned.add(kind)
        logger.warning("Guard signal %s unavailable: %s", kind, exc)


def harm_of(text: str) -> Optional[int]:
    """ContextDefender's harm score (0-100) of ``text``, or None without a model."""
    from app.security import context_defender
    from app.security.types import ContextItem

    scores = context_defender.score([ContextItem(HARM_SOURCE, text)])
    if not scores or not scores[0].scores_available:
        return None
    return int(scores[0].axes.get("harm", 0))


def cos100(a: Optional[Sequence[float]], b: Optional[Sequence[float]]) -> Optional[int]:
    if a is None or b is None:
        return None
    return int(round(100 * max(0.0, float(np.dot(np.asarray(a), np.asarray(b))))))


def _intent_record(state: Optional[WorkflowState]) -> Optional[dict]:
    if state is None:
        return None
    record = ledger.get_ledger(state).get("intent")
    return record if isinstance(record, dict) and record.get("text") else None


def intent_embedding(state: Optional[WorkflowState]) -> Optional[Sequence[float]]:
    """The intent's MiniLM-L6 embedding, computed on first use and cached in the record (SEC-100)."""
    record = _intent_record(state)
    if record is None:
        return None
    if "embedding" not in record:
        try:
            vec = alignment.encode(record["text"])
        except Exception as exc:  # noqa: BLE001
            _warn_once("encoder", exc)
            vec = None
        record["embedding"] = vec.tolist() if vec is not None else None
    return record["embedding"]


def intent_harm(state: Optional[WorkflowState]) -> int:
    """ContextDefender's harm score of the user's turn, computed on first use and cached (SEC-100)."""
    record = _intent_record(state)
    if record is None:
        return 0
    if "harm" not in record:
        try:
            record["harm"] = harm_of(record.get("turn") or record["text"])
        except Exception as exc:  # noqa: BLE001
            _warn_once("harm", exc)
            record["harm"] = None
    return int(record.get("harm") or 0)


def score_text(state: WorkflowState, text: str) -> TextSignals:
    """Intent alignment, task alignment and harm of ``text`` (an argument or a subgoal)."""
    text = (text or "").strip()
    if not text:
        return TextSignals()
    harm_future = _POOL.submit(harm_of, text)
    vec = None
    try:
        vec = alignment.encode(text)
    except Exception as exc:  # noqa: BLE001 - a signal failure never blocks a call (SEC-110)
        _warn_once("encoder", exc)
    harm: Optional[int] = None
    try:
        harm = harm_future.result(timeout=HARM_TIMEOUT_SEC)
    except Exception as exc:  # noqa: BLE001
        _warn_once("harm", exc)
    if vec is None or harm is None:
        if vec is None:
            _warn_once("encoder", RuntimeError("no encoder"))
        if harm is None:
            _warn_once("harm", RuntimeError("no ContextDefender model"))
        return TextSignals(available=False)
    intent_alignment = cos100(vec, intent_embedding(state))
    task_alignment = cos100(vec, alignment.task_embedding(state))
    return TextSignals(
        intent_alignment=100 if intent_alignment is None else intent_alignment,
        task_alignment=100 if task_alignment is None else task_alignment,
        harm=harm,
        available=intent_alignment is not None,
    )


def warm_intent(state: WorkflowState) -> None:
    """Compute the intent's embedding, harm and the task-text embedding now (called by the plan check,
    during the supervisor step), so a tool call never pays for them. Never raises."""
    try:
        intent_embedding(state)
        intent_harm(state)
        alignment.task_embedding(state)
    except Exception as exc:  # noqa: BLE001
        _warn_once("warm", exc)


def risk_above_intent(state: WorkflowState, harm: int) -> int:
    return max(0, int(harm) - intent_harm(state))
