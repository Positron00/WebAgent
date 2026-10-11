"""Repeated MLflow logging inside an active run must not fail (params can't change once logged)."""
import pytest

mlflow = pytest.importorskip("mlflow")

from app.core import observability  # noqa: E402
from app.core.config import settings  # noqa: E402


@pytest.fixture
def isolated_mlflow(tmp_path, monkeypatch):
    """A throwaway file store; MLflow's global tracking URI and experiment are restored after."""
    previous_uri = mlflow.get_tracking_uri()
    uri = f"file://{tmp_path / 'mlruns'}"
    monkeypatch.setattr(settings, "MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("logging_test")
    yield
    while mlflow.active_run():
        mlflow.end_run()
    mlflow.set_tracking_uri(previous_uri)
    monkeypatch.delenv("MLFLOW_EXPERIMENT_ID", raising=False)


def test_repeated_params_inside_an_active_run_are_logged_to_child_runs(isolated_mlflow, caplog):
    with caplog.at_level("WARNING"):
        with mlflow.start_run(run_name="rag_agent_run") as parent:
            observability.log_to_mlflow(params={"query_length": 300}, metrics={"search_latency": 0.1})
            observability.log_to_mlflow(params={"query_length": 42}, metrics={"search_latency": 0.2})

    assert "Failed to log to MLflow" not in caplog.text
    children = mlflow.search_runs(
        experiment_ids=[parent.info.experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{parent.info.run_id}'",
    )
    assert sorted(children["params.query_length"].tolist()) == ["300", "42"]
    assert mlflow.active_run() is None
