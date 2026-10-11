"""Health probes and per-caller rate limiting behind the web app's proxy."""
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.core import security
from app.core.config import settings
from main import app


def _request(host: str, forwarded_for: str = ""):
    headers = {"x-forwarded-for": forwarded_for} if forwarded_for else {}
    return SimpleNamespace(client=SimpleNamespace(host=host), headers=headers)


@pytest.fixture(autouse=True)
def _fresh_rate_limits(monkeypatch):
    monkeypatch.setattr(security, "rate_limit_data", {})
    monkeypatch.setattr(security, "_last_rate_limit_sweep", 0.0)


def test_health_probe_needs_no_api_key(monkeypatch):
    monkeypatch.setattr(settings, "API_KEY_ENABLED", True)

    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health_probe_is_never_rate_limited(monkeypatch):
    monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", True)
    monkeypatch.setattr(settings, "RATE_LIMIT_MAX_REQUESTS", 2)
    client = TestClient(app)

    statuses = {client.get("/health").status_code for _ in range(5)}

    assert statuses == {200}


def test_metrics_are_served_where_prometheus_scrapes_and_never_throttled(monkeypatch):
    # prometheus.yml and the Kubernetes annotations scrape /metrics
    monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", True)
    monkeypatch.setattr(settings, "RATE_LIMIT_MAX_REQUESTS", 1)
    client = TestClient(app)

    responses = [client.get(path) for path in ("/metrics", "/metrics", "/api/v1/metrics")]

    assert [r.status_code for r in responses] == [200, 200, 200]
    assert "webagent" in responses[0].text


def test_rate_limited_response_says_when_to_retry(monkeypatch):
    monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", True)
    monkeypatch.setattr(settings, "RATE_LIMIT_MAX_REQUESTS", 1)
    client = TestClient(app)

    client.get("/api/task/00000000-0000-0000-0000-000000000000")
    limited = client.get("/api/task/00000000-0000-0000-0000-000000000000")

    assert limited.status_code == 429
    assert limited.headers["Retry-After"] == str(settings.RATE_LIMIT_WINDOW_SIZE)


def test_caller_behind_trusted_proxy_is_identified_by_forwarded_address():
    assert security.client_identifier(_request("127.0.0.1", "203.0.113.7, 10.0.0.1")) == "203.0.113.7"


def test_forwarded_header_from_untrusted_client_is_ignored():
    # Anyone can send X-Forwarded-For; only the proxy's value is believed.
    assert security.client_identifier(_request("198.51.100.4", "203.0.113.7")) == "198.51.100.4"


def test_users_behind_the_proxy_have_separate_budgets(monkeypatch):
    monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", True)
    monkeypatch.setattr(settings, "RATE_LIMIT_MAX_REQUESTS", 1)

    assert security.apply_rate_limit(security.client_identifier(_request("127.0.0.1", "203.0.113.1")))
    assert security.apply_rate_limit(security.client_identifier(_request("127.0.0.1", "203.0.113.2")))
    assert not security.apply_rate_limit(security.client_identifier(_request("127.0.0.1", "203.0.113.1")))


def test_idle_callers_are_dropped_from_the_table(monkeypatch):
    monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", True)
    security.rate_limit_data["old-caller"] = {"requests": [1.0], "last_reset": 1.0}

    security.apply_rate_limit("new-caller")

    assert "old-caller" not in security.rate_limit_data
    assert "new-caller" in security.rate_limit_data
