"""IMP-082: consistent error envelope on API responses."""
import pytest
from fastapi.testclient import TestClient

from app.api.endpoints import health
from app.core import security
from app.core.config import settings
from main import app

KEY = "test-key-envelope"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(settings, "API_KEY_ENABLED", True)
    monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", False)
    monkeypatch.setenv("WEBAGENT_API_KEY", KEY)

    async def fake_status():
        return {"status": "ok"}

    monkeypatch.setattr(health, "get_llm_status", fake_status)
    monkeypatch.setattr(health, "get_vectordb_status", fake_status)
    return TestClient(app, raise_server_exceptions=False)


def _error(payload):
    assert "error" in payload
    assert isinstance(payload["error"], dict)
    assert "code" in payload["error"]
    assert "message" in payload["error"]
    return payload["error"]


def test_401_unauthorized(client):
    response = client.get("/api/v1/health/detailed")
    assert response.status_code == 401
    err = _error(response.json())
    assert err["code"] == "unauthorized"


def test_404_not_found(client):
    missing = "00000000-0000-0000-0000-000000000000"
    response = client.post(
        f"/api/task/{missing}/cancel",
        headers={"X-API-Key": KEY},
    )
    assert response.status_code == 404
    err = _error(response.json())
    assert err["code"] == "not_found"
    assert "detail" in response.json()


def test_413_request_too_large(client, monkeypatch):
    monkeypatch.setattr(settings, "REQUEST_SIZE_LIMIT", 10)
    response = client.post(
        "/api/triage",
        headers={"X-API-Key": KEY, "Content-Length": "999999"},
        json={"query": "hi"},
    )
    assert response.status_code == 413
    err = _error(response.json())
    assert err["code"] == "too_large"


def test_422_validation_error(client):
    response = client.post(
        "/api/task",
        headers={"X-API-Key": KEY},
        json={"mode": "always"},
    )
    assert response.status_code == 422
    body = response.json()
    err = _error(body)
    assert err["code"] == "validation_error"
    assert "detail" in body


def test_429_rate_limited(client, monkeypatch):
    monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", True)
    monkeypatch.setattr(settings, "RATE_LIMIT_MAX_REQUESTS", 1)
    monkeypatch.setattr(security, "rate_limit_data", {})
    monkeypatch.setattr(security, "_last_rate_limit_sweep", 0.0)

    client.get("/api/task/00000000-0000-0000-0000-000000000000", headers={"X-API-Key": KEY})
    limited = client.get(
        "/api/task/00000000-0000-0000-0000-000000000000",
        headers={"X-API-Key": KEY},
    )
    assert limited.status_code == 429
    err = _error(limited.json())
    assert err["code"] == "rate_limited"


def test_500_internal_error(client, monkeypatch):
    async def boom():
        raise RuntimeError("boom")

    monkeypatch.setattr(health, "get_llm_status", boom)
    response = client.get("/api/v1/health/detailed", headers={"X-API-Key": KEY})
    assert response.status_code == 500
    body = response.json()
    err = _error(body)
    assert err["code"] == "internal_error"
    assert "error_id" in body
