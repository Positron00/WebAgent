"""Experiment tracking is optional: a tracking store that fails must not stop the server (3.10.0)."""
import importlib

import pytest


def test_a_failing_tracking_store_turns_tracking_off(monkeypatch):
    mlflow = pytest.importorskip("mlflow")
    from app.agents import internal_research

    def broken(*args, **kwargs):
        raise RuntimeError("tracking store unavailable")

    monkeypatch.setenv("MLFLOW_TRACKING_URI", "file:///nonexistent/mlruns")
    monkeypatch.setattr(mlflow, "get_experiment_by_name", broken)
    module = importlib.reload(internal_research)
    try:
        assert module.TRACKING_ENABLED is False
    finally:
        monkeypatch.delenv("MLFLOW_TRACKING_URI")
        monkeypatch.undo()
        importlib.reload(internal_research)
