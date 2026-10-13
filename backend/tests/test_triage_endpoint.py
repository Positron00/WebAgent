"""POST /api/triage contract (REQ-010)."""
import pytest
from fastapi.testclient import TestClient

from app.core import security
from app.core.config import settings
from app.services import triage_model
from main import app
from tests.fakes.triage import FakeTriageModel

client = TestClient(app)


@pytest.fixture(autouse=True)
def _isolated_triage_client(monkeypatch):
    """Other suites share the global rate-limit bucket; reset it so contract tests are not 429."""
    monkeypatch.setattr(security, "rate_limit_data", {})
    monkeypatch.setattr(security, "_last_rate_limit_sweep", 0.0)
    monkeypatch.setattr(settings, "API_KEY_ENABLED", False)
    monkeypatch.delenv("WEBAGENT_API_KEY", raising=False)


def test_empty_query_answered_by_rule():
    r = client.post("/api/triage", json={"query": "", "has_image": False})
    assert r.status_code == 200
    assert r.json()["route"] == "direct"
    assert r.json()["decided_by"] == "rule"


def test_response_contract():
    triage_model.set_model_for_tests(FakeTriageModel(p=0.8, category="current events", version="v-test"))
    r = client.post(
        "/api/triage",
        json={"query": "what happened in markets today", "previous_user_question": None, "has_image": False},
    )
    assert r.status_code == 200
    assert r.json() == {
        "route": "agentic",
        "p_agentic": 0.8,
        "decided_by": "model",
        "reason": "Looks like: current events",
        "classifier_version": "v-test",
    }


def test_broken_model_still_answers_200():
    r = client.post("/api/triage", json={"query": "a question that needs the model"})
    assert r.status_code == 200
    assert r.json()["decided_by"] == "fallback"
    assert r.json()["route"] == "agentic"


@pytest.mark.parametrize(
    "body",
    [
        {"query": "x" * 8001},
        {"query": "ok", "previous_user_question": "y" * 8001},
    ],
)
def test_over_long_input_is_422(body):
    assert client.post("/api/triage", json=body).status_code == 422


def test_requires_api_key_when_enabled(monkeypatch):
    monkeypatch.setattr(settings, "API_KEY_ENABLED", True)
    monkeypatch.setenv("WEBAGENT_API_KEY", "right-key")
    assert client.post("/api/triage", json={"query": "hi"}).status_code == 401
    ok = client.post("/api/triage", json={"query": "hi"}, headers={"X-API-Key": "right-key"})
    assert ok.status_code == 200


def test_rate_limit_applies(monkeypatch):
    from app.core import security

    monkeypatch.setattr(security, "rate_limit_data", {})
    monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", True)
    monkeypatch.setattr(settings, "RATE_LIMIT_MAX_REQUESTS", 1)
    assert client.post("/api/triage", json={"query": "hi"}).status_code == 200
    assert client.post("/api/triage", json={"query": "hi"}).status_code == 429
