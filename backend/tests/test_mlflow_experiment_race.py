"""Two workers creating the MLflow experiment at once must not crash the second one."""
import importlib
import sys

import mlflow
import pytest


def test_import_survives_experiment_created_by_another_worker(monkeypatch, tmp_path):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file://{tmp_path}/mlruns")
    original_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"file://{tmp_path}/mlruns")
    try:
        # The other worker wins the race: the lookup says "missing", but create then finds it exists.
        mlflow.create_experiment("rag_agent_performance")
        real_lookup = mlflow.get_experiment_by_name
        calls = []

        def lookup(name):
            calls.append(name)
            return None if len(calls) == 1 else real_lookup(name)

        monkeypatch.setattr(mlflow, "get_experiment_by_name", lookup)
        sys.modules.pop("app.agents.internal_research", None)
        importlib.import_module("app.agents.internal_research")  # must not raise
        assert len(calls) == 2
    finally:
        mlflow.set_tracking_uri(original_uri)
        sys.modules.pop("app.agents.internal_research", None)
        importlib.import_module("app.agents.internal_research")
