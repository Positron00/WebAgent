"""POST /api/guard/input contract (SEC-034)."""
import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from main import app


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(settings, "API_KEY_ENABLED", False)
    return TestClient(app)


def test_guard_input_allow_in_shadow(client, monkeypatch):
    monkeypatch.setattr(
        settings,
        "SECURITY_GUARD",
        settings.SECURITY_GUARD.model_copy(update={"enabled": True, "input_mode": "shadow"}),
    )
    response = client.post(
        "/api/guard/input",
        json={"messages": [{"role": "user", "content": "What is the capital of France?"}]},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["outcome"] in ("allow", "verify")
    assert body["decision_id"].startswith("chat-")


def test_guard_input_requires_user_message(client):
    response = client.post(
        "/api/guard/input",
        json={"messages": [{"role": "assistant", "content": "hi"}]},
    )
    assert response.status_code == 422
