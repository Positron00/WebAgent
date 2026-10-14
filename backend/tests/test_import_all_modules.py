"""Every module under backend/app imports without error (IMP-024)."""
import importlib
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parents[1] / "app"

# Example scripts start training or MLflow runs when imported; they aren't part of the server.
SKIPPED_PACKAGES = ("app.services.examples",)

def _module_names():
    for path in sorted(APP_DIR.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        relative = path.relative_to(APP_DIR.parent).with_suffix("")
        parts = list(relative.parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        name = ".".join(parts)
        if not name.startswith(SKIPPED_PACKAGES):
            yield name


@pytest.mark.parametrize("module_name", list(_module_names()))
def test_module_imports(module_name):
    importlib.import_module(module_name)
