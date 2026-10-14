"""Small open endpoints (IMP-063) and no static dev keys (IMP-065)."""
import pytest
from fastapi.testclient import TestClient

from app.api.endpoints import health
from app.core import security
from app.core.config import settings
from main import app

KEY = "test-key-for-hardening"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(settings, "API_KEY_ENABLED", True)
    monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", False)
    monkeypatch.setenv("WEBAGENT_API_KEY", KEY)

    async def fake_status():
        return {"status": "ok"}

    monkeypatch.setattr(health, "get_llm_status", fake_status)
    monkeypatch.setattr(health, "get_vectordb_status", fake_status)
    return TestClient(app)


def test_detailed_health_needs_the_api_key(client):
    assert client.get("/api/v1/health/detailed").status_code == 401
    assert client.get("/api/v1/health/detailed", headers={"X-API-Key": "wrong"}).status_code == 401


def test_detailed_health_works_with_the_api_key(client):
    response = client.get("/api/v1/health/detailed", headers={"X-API-Key": KEY})

    assert response.status_code == 200
    assert response.json()["llm_service"] == {"status": "ok"}


@pytest.mark.parametrize("path", ["/health", "/api/v1/health/"])
def test_basic_health_stays_open(client, path):
    assert client.get(path).status_code == 200


@pytest.mark.parametrize("old_dev_key", ["dev_admin_key", "dev_user_key"])
def test_old_dev_keys_are_rejected_in_dev(client, monkeypatch, old_dev_key):
    monkeypatch.setattr(settings, "WEBAGENT_ENV", "dev")

    response = client.get("/api/v1/health/detailed", headers={"X-API-Key": old_dev_key})

    assert response.status_code == 401


def test_only_webagent_api_key_is_valid(monkeypatch):
    monkeypatch.setattr(settings, "WEBAGENT_ENV", "dev")
    monkeypatch.setenv("WEBAGENT_API_KEY", KEY)
    assert security._configured_api_keys() == [KEY]

    monkeypatch.delenv("WEBAGENT_API_KEY")
    assert security._configured_api_keys() == []


def test_jwt_helpers_are_gone():
    for name in ("DEV_API_KEYS", "create_jwt", "decode_jwt", "validate_auth"):
        assert not hasattr(security, name)
