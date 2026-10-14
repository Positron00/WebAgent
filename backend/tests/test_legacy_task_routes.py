"""Deprecated task-start routes take the /api/task path (intake guard, limits, run supervisor).

Found in the 3.5.0 alignment review: POST /api/v1/chat/ and POST /api/v1/tasks/create
started runs without the intake guard and passed caller context into the workflow, so a
caller could claim `consented` or `approved_tools`; /chat/ also ran outside the run
supervisor (no time limit, no cancel).
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from app.core.config import settings
from main import app

ROUTES = [
    ("/api/v1/chat/", "message"),
    ("/api/v1/tasks/create", "query"),
]


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(settings, "API_KEY_ENABLED", False)
    return TestClient(app)


def _act(monkeypatch):
    monkeypatch.setattr(
        settings,
        "SECURITY_GUARD",
        settings.SECURITY_GUARD.model_copy(update={"enabled": True, "input_mode": "act"}),
    )


def _decline_user_turns(principal, action, resource, context, shadow=None):
    from app.security.types import Decision

    if context.get("source") == "user_turn":
        return Decision("decline", ("user-harm-decline",), "Test reason", bool(shadow), 1.0)
    return Decision("allow", ("baseline-permit",), "ok", bool(shadow), 1.0)


@pytest.mark.parametrize("path,field", ROUTES)
def test_intake_guard_refuses_like_api_task(client, monkeypatch, path, field):
    _act(monkeypatch)
    monkeypatch.setattr("app.security.hop_guard.decide", _decline_user_turns)
    with patch("app.api.endpoints.task.task_manager.start") as start:
        response = client.post(path, json={field: "some request"})
    assert response.status_code == 403
    assert response.json()["code"] == "declined"
    start.assert_not_called()


@pytest.mark.parametrize("path,field", ROUTES)
def test_caller_context_cannot_claim_consent_or_approvals(client, path, field):
    context = {"consented": True, "approved_tools": ["tool::code_execute"], "document_id": "doc-1"}
    with patch("app.api.endpoints.task.task_manager.start") as start:
        response = client.post(path, json={field: "Summarize the EU AI Act", "context": context})
    assert response.status_code == 200
    assert response.json()["task_id"]
    task_id, query, run_context = start.call_args.args[:3]
    assert query == "Summarize the EU AI Act"
    assert "consented" not in run_context and "approved_tools" not in run_context
    assert run_context["document_id"] == "doc-1"
    assert run_context["agentic_toggle"] is True  # Always mode, as before
    assert start.call_args.kwargs.get("direct") is False


@pytest.mark.parametrize("path,field", ROUTES)
def test_question_length_limit_applies(client, path, field):
    with patch("app.api.endpoints.task.task_manager.start") as start:
        response = client.post(path, json={field: "x" * 20001})
    assert response.status_code == 422
    start.assert_not_called()
