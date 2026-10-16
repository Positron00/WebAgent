"""Observability API routes behind the API key (IMP-128)."""
import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from main import app

KEY = "test-key-observability"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(settings, "API_KEY_ENABLED", True)
    monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", False)
    monkeypatch.setenv("WEBAGENT_API_KEY", KEY)
    return TestClient(app)


@pytest.mark.parametrize(
    "path",
    [
        "/api/v1/observability/metrics",
        "/api/v1/observability/drift",
        "/api/v1/observability/health",
    ],
)
def test_observability_routes_need_api_key(client, path):
    assert client.get(path).status_code == 401
    assert client.get(path, headers={"X-API-Key": "wrong"}).status_code == 401


def test_observability_metrics_with_key(client):
    response = client.get(
        "/api/v1/observability/metrics", headers={"X-API-Key": KEY}
    )
    assert response.status_code == 200
    body = response.json()
    assert "model_metrics" in body
    assert "document_metrics" in body
    assert "timestamp" in body


def test_observability_drift_with_key(client):
    response = client.get(
        "/api/v1/observability/drift", headers={"X-API-Key": KEY}
    )
    assert response.status_code == 200
    body = response.json()
    assert "global_drift_stats" in body
    assert "llm_service_drift" in body
    assert "doc_service_drift" in body


# --- 3.6.0: answers come from the shared drift store (OBS-013) ---

def _seed_drift_and_errors():
    """A drifting llm_speed detector and 10% model errors, computed into state.json."""
    import json as _json
    from datetime import datetime, timedelta

    import numpy as np

    from app.core import drift_store

    rng = np.random.default_rng(5)
    start = datetime.now() - timedelta(minutes=50)
    values = list(rng.normal(5, 1, 50)) + list(rng.normal(15, 1, 30))
    rows = [(start + timedelta(seconds=30 * i), "llm_speed:supervisor", v) for i, v in enumerate(values)]
    rows += [(start, "llm_outcome", "ok")] * 18 + [(start, "llm_outcome", "error")] * 2
    rows += [(start, "llm_tokens:supervisor", 100)]
    folder = drift_store.drift_dir()
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / f"samples-{datetime.now():%Y-%m-%d}.jsonl", "a") as handle:
        for ts, detector, value in rows:
            handle.write(_json.dumps({"ts": ts.isoformat(), "d": detector, "v": value}) + "\n")
    drift_store.compute_once()


def _get(client, path):
    response = client.get(f"/api/v1/observability/{path}", headers={"X-API-Key": KEY})
    assert response.status_code == 200
    return response.json()


def test_drift_lists_detectors_without_reference_values(client):
    _seed_drift_and_errors()
    body = _get(client, "drift")
    det = body["detectors"]["llm_speed:supervisor"]
    assert det["drift"] and det["enough_data"] and det["group"] == "llm" and det["kind"] == "ks"
    assert {"score", "reference_count", "current_count", "reference_start", "last_drift_at"} <= set(det)
    assert not {"reference", "reference_centroid", "reference_end"} & set(det)
    assert body["global_drift_stats"] == body["detectors"] and body["retrain_recommendations"] == []
    assert body["computed_at"]
    llm = body["llm_service_drift"]
    assert llm["latency_drift"]["detector"] == "llm_speed:supervisor"
    assert llm["performance_drift"] is None and llm["retrain_recommended"] is False
    assert llm["embedding_drift"] is None and llm["baseline_count"] == 50
    assert llm["collecting_baseline"]  # llm_tokens:supervisor has 1 of 50 reference samples
    assert body["doc_service_drift"]["pipeline_adjustments_recommended"] is False


def test_alerts_health_and_metrics_agree(client):
    _seed_drift_and_errors()
    alerts = {a["name"]: a for a in _get(client, "alerts")}
    assert set(alerts) == {"model_drift", "model_errors"}
    assert alerts["model_errors"]["timestamp"] == alerts["model_errors"]["since"]
    health = _get(client, "health")
    assert health["overall_status"] == "unhealthy"  # model_errors is critical
    assert health["alert_counts"]["critical"] == 1 and health["alert_counts"]["warning"] == 1
    assert health["services"]["llm"]["status"] == "unhealthy"
    assert health["services"]["document_processing"]["status"] == "healthy"
    model = _get(client, "metrics")["model_metrics"]
    assert (model["requests"], model["errors"]) == (20, 2)
    assert model["by_label"]["supervisor"]["tokens_24h"] == 100


def test_reset_starts_new_references(client):
    _seed_drift_and_errors()
    reply = client.post("/api/v1/observability/llm/reset_baseline", headers={"X-API-Key": KEY})
    assert reply.status_code == 200 and reply.json()["status"] == "success"
    assert "llm_speed:supervisor" not in _get(client, "drift")["detectors"]


def test_drift_off_answers_as_if_empty(client, monkeypatch):
    _seed_drift_and_errors()
    monkeypatch.setattr(settings, "DRIFT_ENABLED", False)
    assert _get(client, "drift")["detectors"] == {}
    assert _get(client, "alerts") == []
    assert _get(client, "health")["overall_status"] == "healthy"
