"""One CORS setup from YAML, and startup/shutdown through the lifespan handler (IMP-024)."""
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

import main
from app.core.config import settings

BACKEND = Path(__file__).resolve().parents[1]


def test_cors_middleware_is_added_once():
    cors = [m for m in main.app.user_middleware if m.cls is CORSMiddleware]

    assert len(cors) == 1


def test_cors_settings_come_from_yaml():
    data = yaml.safe_load((BACKEND / "config" / f"{settings.WEBAGENT_ENV}.yaml").read_text(encoding="utf-8"))
    cors = data["cors"]

    assert settings.CORS_ALLOW_METHODS == cors["allow_methods"]
    assert settings.CORS_ALLOW_HEADERS == cors["allow_headers"]
    assert settings.CORS_ALLOW_CREDENTIALS == cors["allow_credentials"]


@pytest.mark.parametrize("env", ["uat", "prod"])
def test_uat_and_prod_allow_the_api_key_header(env):
    data = yaml.safe_load((BACKEND / "config" / f"{env}.yaml").read_text(encoding="utf-8"))
    headers = [h.lower() for h in data["cors"]["allow_headers"]]

    assert "*" in headers or "x-api-key" in headers


def test_preflight_from_allowed_origin_gets_cors_headers(monkeypatch):
    monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", False)
    origin = settings.CORS_ORIGINS[0]

    reply = TestClient(main.app).options(
        "/api/task",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type,x-api-key",
        },
    )

    assert reply.status_code == 200
    assert reply.headers["access-control-allow-origin"] == origin
    assert "x-api-key" in reply.headers["access-control-allow-headers"].lower()


def test_preflight_from_unknown_origin_is_refused(monkeypatch):
    monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", False)

    reply = TestClient(main.app).options(
        "/api/task",
        headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"},
    )

    assert "access-control-allow-origin" not in reply.headers


def test_lifespan_starts_cleanups_and_stops_them_with_metrics(monkeypatch):
    monkeypatch.setattr(settings, "TRIAGE", settings.TRIAGE.model_copy(update={"enabled": False}))
    monkeypatch.setattr(
        settings, "SECURITY_GUARD", settings.SECURITY_GUARD.model_copy(update={"enabled": False})
    )
    with patch.object(main, "check_production_settings") as checks, \
            patch.object(main, "check_task_timeouts") as timeouts, \
            patch.object(main, "mark_worker_dead") as dead:
        with TestClient(main.app):
            triage_task = main.app.state.triage_cleanup_task
            security_task = main.app.state.security_cleanup_task
            assert not triage_task.done() and not security_task.done()
            dead.assert_not_called()

    checks.assert_called_once()
    timeouts.assert_called_once()
    dead.assert_called_once()
    assert triage_task.cancelled() or triage_task.done()
    assert security_task.cancelled() or security_task.done()
