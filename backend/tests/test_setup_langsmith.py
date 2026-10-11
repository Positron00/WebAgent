"""Tests that LangSmith settings from the app win over ones exported in the user's shell."""
import pytest

from app.core import setup_langsmith as module
from app.core.config import settings

VARS = [
    "LANGSMITH_TRACING", "LANGSMITH_PROJECT", "LANGSMITH_API_KEY", "LANGSMITH_ENDPOINT",
    "LANGCHAIN_TRACING_V2", "LANGCHAIN_PROJECT", "LANGCHAIN_API_KEY", "LANGCHAIN_ENDPOINT",
]


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch):
    for name in VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(module, "Client", lambda **kwargs: object())
    monkeypatch.setattr(module, "LangChainTracer", lambda **kwargs: object())


def test_app_project_overrides_shell_langsmith_project(monkeypatch):
    monkeypatch.setenv("LANGSMITH_PROJECT", "some-other-project")
    monkeypatch.setattr(settings, "LANGSMITH_API_KEY", "test-key")
    monkeypatch.setattr(settings, "LANGSMITH_TRACING_ENABLED", True)
    monkeypatch.setattr(settings, "LANGSMITH_PROJECT_NAME", "webagent-research")

    assert module.setup_langsmith() is True

    import os
    assert os.environ["LANGSMITH_PROJECT"] == "webagent-research"
    assert os.environ["LANGCHAIN_PROJECT"] == "webagent-research"
    assert os.environ["LANGSMITH_TRACING"] == "true"
    assert os.environ["LANGCHAIN_TRACING_V2"] == "true"
    assert os.environ["LANGSMITH_API_KEY"] == "test-key"


def test_tracing_off_in_both_styles_without_a_key(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setattr(settings, "LANGSMITH_API_KEY", None)

    assert module.setup_langsmith() is False

    import os
    assert os.environ["LANGSMITH_TRACING"] == "false"
    assert os.environ["LANGCHAIN_TRACING_V2"] == "false"
