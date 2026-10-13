"""Concurrency-safe MLflow tracking for agent runs.

MLflow's fluent API keeps one "active run" per thread, and every task runs on the same event-loop
thread, so two concurrent tasks in one worker collided ("Run ... is already active") and the second
task failed. Here each run is created with an explicit id through `MlflowClient`, the id is held in a
context variable (one per asyncio task, copied into threads started from it), and tracking errors are
logged and dropped: tracking must never fail a task.
"""
from __future__ import annotations

import contextvars
import logging
import os
import threading
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_RUN_ID: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("mlflow_task_run_id", default=None)
_experiments: Dict[str, str] = {}
_lock = threading.Lock()


def enabled() -> bool:
    return os.environ.get("MLFLOW_TRACKING_URI") is not None


def _client():
    from mlflow.tracking import MlflowClient

    return MlflowClient()


def _experiment_id(name: str) -> Optional[str]:
    if name in _experiments:
        return _experiments[name]
    with _lock:
        if name in _experiments:
            return _experiments[name]
        from app.core.config import settings

        client = _client()
        exp = client.get_experiment_by_name(name)
        if exp is None:
            try:
                client.create_experiment(name, artifact_location=getattr(settings, "MLFLOW_ARTIFACT_LOCATION", None))
            except Exception:  # noqa: BLE001 - another worker created it first
                pass
            exp = client.get_experiment_by_name(name)
        if exp is None:
            return None
        _experiments[name] = exp.experiment_id
        return exp.experiment_id


def start_run(run_name: str, params: Optional[Dict[str, Any]] = None,
              experiment: str = "rag_agent_performance") -> Optional[str]:
    """Start a run for the current task; returns its id (or None when tracking is off or failed)."""
    if not enabled():
        return None
    try:
        exp_id = _experiment_id(experiment)
        if exp_id is None:
            return None
        run = _client().create_run(exp_id, run_name=run_name)
        _RUN_ID.set(run.info.run_id)
        for key, value in (params or {}).items():
            log_param(key, value)
        return run.info.run_id
    except Exception as exc:  # noqa: BLE001
        logger.debug("MLflow run not started: %s", exc)
        return None


def log_param(key: str, value: Any) -> None:
    run_id = _RUN_ID.get()
    if run_id is None:
        return
    try:
        _client().log_param(run_id, key, str(value)[:500])
    except Exception as exc:  # noqa: BLE001 - e.g. a param logged twice with another value
        logger.debug("MLflow param %s not logged: %s", key, exc)


def log_metric(key: str, value: float) -> None:
    run_id = _RUN_ID.get()
    if run_id is None:
        return
    try:
        _client().log_metric(run_id, key, float(value))
    except Exception as exc:  # noqa: BLE001
        logger.debug("MLflow metric %s not logged: %s", key, exc)


def end_run(status: str = "FINISHED") -> None:
    run_id = _RUN_ID.get()
    if run_id is None:
        return
    try:
        _client().set_terminated(run_id, status=status)
    except Exception as exc:  # noqa: BLE001
        logger.debug("MLflow run not ended: %s", exc)
    finally:
        _RUN_ID.set(None)
