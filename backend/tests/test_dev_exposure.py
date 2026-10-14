"""Dev backend stays local (IMP-014) and the stale-task timeout warning (IMP-041)."""
import logging
from pathlib import Path

import pytest
import yaml

from app.core import startup_checks
from app.core.config import settings

BACKEND = Path(__file__).resolve().parents[1]


@pytest.fixture
def dev(monkeypatch):
    monkeypatch.setattr(settings, "WEBAGENT_ENV", "dev")
    monkeypatch.setattr(settings, "API_KEY_ENABLED", False)
    monkeypatch.setattr(settings, "ALLOW_OPEN_DEV", False)


@pytest.mark.parametrize("host", ["127.0.0.1", "::1", "localhost"])
def test_loopback_hosts_are_fine(dev, host):
    startup_checks.check_dev_exposure(host)


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.5"])
def test_open_host_in_dev_without_keys_refuses_and_says_how_to_fix(dev, host):
    with pytest.raises(SystemExit) as err:
        startup_checks.check_dev_exposure(host)

    message = str(err.value)
    assert host in message
    assert "127.0.0.1" in message and "WEBAGENT_ALLOW_OPEN_DEV=true" in message


@pytest.mark.parametrize("setting,value", [
    ("ALLOW_OPEN_DEV", True),
    ("API_KEY_ENABLED", True),
    ("WEBAGENT_ENV", "prod"),
    ("WEBAGENT_ENV", "uat"),
])
def test_open_host_allowed_when(dev, monkeypatch, setting, value):
    monkeypatch.setattr(settings, setting, value)

    startup_checks.check_dev_exposure("0.0.0.0")


def test_dev_yaml_listens_on_loopback():
    data = yaml.safe_load((BACKEND / "config" / "dev.yaml").read_text(encoding="utf-8"))

    assert data["api"]["host"] == "127.0.0.1"


def test_compose_publishes_ports_on_loopback_only():
    data = yaml.safe_load((BACKEND / "docker-compose.yml").read_text(encoding="utf-8"))

    ports = [p for service in data["services"].values() for p in service.get("ports", [])]
    assert len(ports) == 3
    assert all(str(p).startswith("127.0.0.1:") for p in ports)
    assert "127.0.0.1:${GRAFANA_HOST_PORT:-3001}:3000" in ports


@pytest.mark.parametrize("minutes,max_seconds,warns", [
    ("30", 420, False),
    ("9", 420, False),
    ("8", 420, True),
    ("5", 420, True),
])
def test_task_timeout_warning(monkeypatch, caplog, minutes, max_seconds, warns):
    monkeypatch.setenv("WEBAGENT_TASK_TIMEOUT_MINUTES", minutes)
    monkeypatch.setattr(settings, "TASK_MAX_SECONDS", max_seconds)

    with caplog.at_level(logging.WARNING, logger=startup_checks.__name__):
        startup_checks.check_task_timeouts()

    assert any("WEBAGENT_TASK_TIMEOUT_MINUTES" in r.message for r in caplog.records) == warns
