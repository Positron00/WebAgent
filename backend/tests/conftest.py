"""Shared test setup: import paths, no tracing, no network, no writes to tracked files."""
import os
import socket
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_DIR.parent

for path in (str(REPO_ROOT), str(BACKEND_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

# Set before any app module loads: config reads these at import time.
# .env.local supplies a real LangSmith key, so tracing must be switched off explicitly.
for _name in ("LANGSMITH_TRACING_ENABLED", "LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2"):
    os.environ[_name] = "false"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ["WEBAGENT_NLTK_AUTO_DOWNLOAD"] = "false"
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
# Placeholder provider keys, so agents can be built without a .env.local (CI has none). They
# never reach a provider: outbound connections are blocked below. A real .env.local still wins.
for _name in ("TOGETHER_API_KEY", "OPENAI_API_KEY", "TAVILY_API_KEY"):
    os.environ.setdefault(_name, "test-placeholder-key-never-sent-0000000000")

_LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost", "0.0.0.0"}
_real_connect = socket.socket.connect
_real_connect_ex = socket.socket.connect_ex


def _address_is_local(sock: socket.socket, address) -> bool:
    if sock.family == getattr(socket, "AF_UNIX", None):
        return True
    host = address[0] if isinstance(address, tuple) else address
    return host in _LOCAL_HOSTS


def _guarded_connect(self, address):
    if not _address_is_local(self, address):
        raise RuntimeError(
            f"Test tried to open a network connection to {address!r}. "
            "Fake the external service instead (see docs/TESTING.md)."
        )
    return _real_connect(self, address)


def _guarded_connect_ex(self, address):
    if not _address_is_local(self, address):
        raise RuntimeError(
            f"Test tried to open a network connection to {address!r}. "
            "Fake the external service instead (see docs/TESTING.md)."
        )
    return _real_connect_ex(self, address)


if os.getenv("WEBAGENT_TESTS_ALLOW_NETWORK", "false").lower() != "true":
    socket.socket.connect = _guarded_connect
    socket.socket.connect_ex = _guarded_connect_ex

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _isolate_saved_results(tmp_path, monkeypatch):
    """Keep tests from writing task results into the real backend/data/results folder."""
    from app.services import result_store

    monkeypatch.setattr(result_store, "RESULTS_DIR", tmp_path / "results")


@pytest.fixture(autouse=True)
def _isolate_triage(request, tmp_path, monkeypatch):
    """Tests never write collected questions into the real data folder or load the real classifier.

    Triage runs in `act` mode with collection off and no model (so model decisions fall back to
    agentic). Tests that want shadow mode, collection or a fake model set them explicitly; tests
    marked `real_triage_model` load the committed classifier.
    """
    from app.core.config import settings
    from app.services import triage_model

    monkeypatch.setattr(settings, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(settings, "TRIAGE", settings.TRIAGE.model_copy(update={"mode": "act", "collect": False}))
    if request.node.get_closest_marker("real_triage_model") is None:
        monkeypatch.setattr(triage_model, "load_model", lambda *a, **k: None)
    triage_model.set_model_for_tests(None)
    yield
    triage_model.set_model_for_tests(None)


@pytest.fixture(autouse=True)
def _isolate_rate_limit(monkeypatch):
    """Each test starts with an empty rate-limit bucket, so suites don't 429 each other by order."""
    from app.core import security

    monkeypatch.setattr(security, "rate_limit_data", {})
    monkeypatch.setattr(security, "_last_rate_limit_sweep", 0.0)


@pytest.fixture(autouse=True)
def _isolate_context_defender(monkeypatch):
    from app.security import context_defender

    monkeypatch.setattr(context_defender, "load_model", lambda *a, **k: None)
    context_defender.set_model_for_tests(None)
    yield
    context_defender.set_model_for_tests(None)


@pytest.fixture(autouse=True)
def _isolate_model_registry(tmp_path, monkeypatch):
    """Keep tests from writing the model registry into the real data folder."""
    monkeypatch.setenv("WEBAGENT_MODEL_REGISTRY_PATH", str(tmp_path / "models_registry.json"))
    for module_name in ("backend.models.registry.model_registry", "models.registry.model_registry"):
        module = sys.modules.get(module_name)
        if module is not None:
            monkeypatch.setattr(module, "_registry_instance", None)
