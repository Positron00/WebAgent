"""uat/prod must refuse to start with unsafe settings; dev must never be blocked."""
import subprocess
import sys
from pathlib import Path

import pytest

from app.core import startup_checks
from app.core.config import settings
from app.core.security import INSECURE_SECRET_DEFAULT

BACKEND_DIR = Path(__file__).resolve().parents[1]


@pytest.fixture
def prod(monkeypatch):
    monkeypatch.setattr(settings, "WEBAGENT_ENV", "prod")
    monkeypatch.setattr(settings, "API_KEY_ENABLED", True)
    monkeypatch.setattr(settings, "SECRET_KEY", "a-real-secret")
    monkeypatch.setenv("WEBAGENT_API_KEY", "client-key")
    monkeypatch.setenv("WEBAGENT_CODE_SANDBOX", "local")


def test_good_prod_settings_pass(prod):
    startup_checks.check_production_settings(code_execution_enabled=False)


def test_default_secret_is_rejected(prod, monkeypatch):
    monkeypatch.setattr(settings, "SECRET_KEY", INSECURE_SECRET_DEFAULT)
    with pytest.raises(RuntimeError, match="WEBAGENT_SECURITY_SECRET_KEY"):
        startup_checks.check_production_settings(code_execution_enabled=False)


def test_missing_api_key_is_rejected(prod, monkeypatch):
    monkeypatch.delenv("WEBAGENT_API_KEY")
    with pytest.raises(RuntimeError, match="WEBAGENT_API_KEY"):
        startup_checks.check_production_settings(code_execution_enabled=False)


def test_code_execution_without_container_sandbox_is_rejected(prod):
    with pytest.raises(RuntimeError, match="WEBAGENT_CODE_SANDBOX=docker"):
        startup_checks.check_production_settings(code_execution_enabled=True)


def test_dev_is_never_blocked(monkeypatch):
    monkeypatch.setattr(settings, "WEBAGENT_ENV", "dev")
    monkeypatch.setattr(settings, "SECRET_KEY", INSECURE_SECRET_DEFAULT)
    startup_checks.check_production_settings(code_execution_enabled=True)


def test_command_line_check_exits_with_a_message():
    # The Docker image runs this before uvicorn, so a bad config stops the container.
    # Code execution without the container sandbox fails whatever .env.local provides.
    env = {
        "PATH": "/usr/bin:/bin",
        "WEBAGENT_ENV": "prod",
        "WEBAGENT_SECURITY_SECRET_KEY": "a-real-secret",
        "WEBAGENT_API_KEY": "client-key",
        "WEBAGENT_ENABLE_CODE_EXECUTION": "true",
        "WEBAGENT_CODE_SANDBOX": "local",
        "LANGSMITH_TRACING_ENABLED": "false",
    }
    out = subprocess.run(
        [sys.executable, "-m", "app.core.startup_checks"],
        cwd=BACKEND_DIR, env=env, capture_output=True, text=True, timeout=120,
    )
    assert out.returncode == 1
    assert "Startup check failed" in out.stderr
    assert "WEBAGENT_CODE_SANDBOX=docker" in out.stderr
