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
