"""WEBAGENT_EMAIL_* override notifications.email from YAML, like WEBAGENT_SCHEDULER_ENABLED."""
from types import SimpleNamespace

from app.core import config as config_module


def _attached(monkeypatch, **env):
    for key in ("WEBAGENT_EMAIL_ENABLED", "WEBAGENT_EMAIL_TRANSPORT", "WEBAGENT_EMAIL_MODE", "WEBAGENT_SCHEDULER_ENABLED"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return config_module._attach_schedule(SimpleNamespace())


def test_yaml_values_without_overrides(monkeypatch):
    email = _attached(monkeypatch).NOTIFICATIONS.email
    assert (email.enabled, email.transport, email.mode) == (False, "log", "notice")


def test_overrides_apply_and_invalid_values_keep_yaml(monkeypatch):
    email = _attached(monkeypatch, WEBAGENT_EMAIL_ENABLED="true", WEBAGENT_EMAIL_TRANSPORT="SMTP",
                      WEBAGENT_EMAIL_MODE="loud").NOTIFICATIONS.email
    assert (email.enabled, email.transport, email.mode) == (True, "smtp", "notice")
