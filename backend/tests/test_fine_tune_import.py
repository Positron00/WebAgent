"""fine_tune_service imports without error (IMP-024)."""
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def test_fine_tune_service_imports():
    module = importlib.import_module("backend.models.services.fine_tune_service")

    assert hasattr(module, "FineTuneService")
