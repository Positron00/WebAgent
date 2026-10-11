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
def _isolate_model_registry(tmp_path, monkeypatch):
    """Keep tests from writing the model registry into the real data folder."""
    monkeypatch.setenv("WEBAGENT_MODEL_REGISTRY_PATH", str(tmp_path / "models_registry.json"))
    for module_name in ("backend.models.registry.model_registry", "models.registry.model_registry"):
        module = sys.modules.get(module_name)
        if module is not None:
            monkeypatch.setattr(module, "_registry_instance", None)
