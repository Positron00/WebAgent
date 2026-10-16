"""Observability alerts (OBS-012): each fires on its condition, stays, clears; none says "retrain"."""
import json
from datetime import datetime, timedelta

import pytest

from app.core import drift_alerts, drift_store
from app.core.config import settings

NOW = datetime(2026, 10, 15, 12, 0, 0)


def write(rows):
    folder = drift_store.drift_dir()
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / f"samples-{NOW:%Y-%m-%d}.jsonl", "a") as handle:
        for ts, detector, value in rows:
            handle.write(json.dumps({"ts": ts.isoformat(), "d": detector, "v": value}) + "\n")


def outcomes(ok=0, error=0, rate_limited=0, minutes_ago=10):
    ts = NOW - timedelta(minutes=minutes_ago)
    return [(ts, "llm_outcome", "ok")] * ok + [(ts, "llm_outcome", "error")] * error + \
        [(ts, "llm_outcome", "rate_limited")] * rate_limited


def active(now=NOW):
    return {a["name"]: a for a in drift_store.compute_once(now)["alerts"]}


def test_model_errors_fires_above_5_percent_with_enough_requests_and_ignores_rate_limits():
    write(outcomes(ok=18, error=2, rate_limited=30))  # 2 of 20 counted = 10%
    alerts = active()
    assert alerts["model_errors"]["severity"] == "critical" and alerts["model_errors"]["service"] == "llm"
    assert "rate_limited" in alerts  # 30 of 50 = 60%


def test_model_errors_needs_20_requests():
    write(outcomes(ok=10, error=9))
    assert "model_errors" not in active()


def test_alert_stays_with_its_since_and_clears_when_the_window_passes():
    write(outcomes(ok=18, error=2))
    first = active()["model_errors"]["since"]
    assert active(NOW + timedelta(minutes=5))["model_errors"]["since"] == first
    assert "model_errors" not in active(NOW + timedelta(hours=2))


def test_token_usage_is_off_by_default_and_fires_over_the_limit(monkeypatch):
    write([(NOW - timedelta(hours=1), "llm_tokens:supervisor", 6000)] * 2)
    assert "token_usage" not in active()
    monkeypatch.setattr(settings, "TOKEN_ALERT_DAILY", 10000)
    assert active()["token_usage"]["severity"] == "warning"


def test_document_errors_fires_above_10_percent_with_10_loads():
    write([(NOW - timedelta(hours=3), "doc_outcome", "ok")] * 8 + [(NOW - timedelta(hours=3), "doc_outcome", "error")] * 2)
    alerts = active()
    assert alerts["document_errors"]["service"] == "document_processing"


@pytest.mark.parametrize("detector,alert", [("llm_tokens:a", "model_drift"), ("doc_size", "document_drift")])
def test_drift_alerts_name_the_detector(detector, alert):
    detectors = {detector: {"drift": True}, "llm_speed:b": {"drift": False}}
    figs = drift_alerts.figures([], NOW)
    fired = {a["name"]: a for a in drift_alerts.evaluate(figs, detectors, 0)}
    assert set(fired) == {alert} and detector in fired[alert]["message"]


def test_no_alert_message_says_retrain(monkeypatch):
    figs = {"llm": {"requests": 100, "errors": 50, "rate_limited": 40, "tokens_24h": 10**9, "by_label": {}},
            "doc": {"loads": 100, "errors": 50}}
    detectors = {"llm_speed:a": {"drift": True}, "doc_type": {"drift": True}}
    alerts = drift_alerts.evaluate(figs, detectors, 1)
    assert {a["name"] for a in alerts} == set(drift_alerts.ALERTS)
    assert not any("retrain" in a["message"].lower() for a in alerts)
