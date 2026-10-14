"""uat and prod don't publish the OpenAPI docs; dev does (IMP-063)."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]

PROBE = (
    "import main; "
    "print(main.app.openapi_url, main.app.docs_url, main.app.redoc_url)"
)


def _doc_urls(env_name: str) -> str:
    env = {
        **os.environ,
        "WEBAGENT_ENV": env_name,
        "WEBAGENT_SECURITY_SECRET_KEY": "x" * 48,
        "WEBAGENT_API_KEY": "test-key",
        "LANGSMITH_TRACING_ENABLED": "false",
    }
    result = subprocess.run(
        [sys.executable, "-c", PROBE],
        cwd=BACKEND_DIR, env=env, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    return result.stdout.strip().splitlines()[-1]


@pytest.mark.parametrize("env_name", ["uat", "prod"])
def test_docs_are_off_in_uat_and_prod(env_name):
    assert _doc_urls(env_name) == "None None None"


def test_docs_are_on_in_dev():
    assert _doc_urls("dev") == "/api/v1/openapi.json /api/v1/docs /api/v1/redoc"
