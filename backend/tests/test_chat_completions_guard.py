"""Intake guard and input limits on /api/v1/chat/completions (IMP-020, IMP-022)."""
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.security.types import Decision
from main import app

URL = f"{settings.API_V1_STR}/chat/completions"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(settings, "API_KEY_ENABLED", False)
    monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", False)
    return TestClient(app)


def _guard(monkeypatch, mode):
    monkeypatch.setattr(
        settings,
        "SECURITY_GUARD",
        settings.SECURITY_GUARD.model_copy(update={"enabled": True, "input_mode": mode}),
    )


def _fake_decide(outcome, policy, calls):
    def decide(principal, action, resource, context, shadow=None):
        if context.get("source") == "user_turn":
            calls.append(bool(shadow))
            return Decision(outcome, (policy,), "Test reason", bool(shadow), 1.0)
        return Decision("allow", ("baseline-permit",), "ok", bool(shadow), 1.0)

    return decide


def _user(content="some request"):
    return {"role": "user", "content": content}


@pytest.mark.parametrize("outcome,status,code", [
    ("decline", 403, "declined"),
    ("consent", 409, "consent_required"),
])
def test_act_mode_refuses_like_api_task(client, monkeypatch, outcome, status, code):
    _guard(monkeypatch, "act")
    monkeypatch.setattr("app.security.hop_guard.decide", _fake_decide(outcome, "test-policy", []))

    with patch("app.api.endpoints.task.task_manager.register_pending"):
        task_reply = client.post("/api/task", json={"query": "some request"})
    with patch("app.api.endpoints.frontend.task_manager") as tm:
        chat_reply = client.post(URL, json={"messages": [_user()]})

    assert chat_reply.status_code == task_reply.status_code == status
    chat_body, task_body = chat_reply.json(), task_reply.json()
    assert chat_body["code"] == code and chat_body["policy_id"] == "test-policy"
    assert set(chat_body) == set(task_body)
    tm.register_pending.assert_not_called()


def test_shadow_mode_runs_and_records_the_decision(client, monkeypatch):
    _guard(monkeypatch, "shadow")
    calls = []
    monkeypatch.setattr("app.security.hop_guard.decide", _fake_decide("decline", "test-policy", calls))

    with patch("app.api.endpoints.frontend.task_manager") as tm:
        reply = client.post(URL, json={"messages": [_user()]})

    assert reply.status_code == 200
    assert calls == [True]
    tm.register_pending.assert_called_once()


def test_missing_user_message_stays_400(client):
    with patch("app.api.endpoints.frontend.task_manager"):
        reply = client.post(URL, json={"messages": [{"role": "assistant", "content": "hi"}]})

    assert reply.status_code == 400


def test_system_messages_are_not_passed_to_the_agents(client):
    messages = [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "answer"},
        _user("second"),
    ]
    with patch("app.api.endpoints.frontend.task_manager") as tm:
        reply = client.post(URL, json={"messages": messages})

    assert reply.status_code == 200
    _task_id, message, context = tm.start.call_args.args
    assert [m["role"] for m in context["previous_messages"]] == ["user", "assistant"]
    assert message == "second"
    assert tm.start.call_args.kwargs["direct"] is True


@pytest.mark.parametrize("messages,status", [
    ([_user()] * 8, 200),
    ([_user()] * 9, 200),
    ([_user()] * 10, 422),
    ([_user("x" * 19999)], 200),
    ([_user("x" * 20000)], 200),
    ([_user("x" * 20001)], 422),
    ([{"role": "tool", "content": "q"}], 422),
])
def test_message_limits(client, monkeypatch, messages, status):
    monkeypatch.setattr(settings, "SECURITY_GUARD", settings.SECURITY_GUARD.model_copy(update={"enabled": False}))
    with patch("app.api.endpoints.frontend.task_manager"):
        reply = client.post(URL, json={"messages": messages})

    assert reply.status_code == status
