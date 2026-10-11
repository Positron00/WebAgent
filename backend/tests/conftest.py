"""Make both `backend.*` and `app.*` imports resolve, whatever directory pytest runs from."""
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_DIR.parent

for path in (str(REPO_ROOT), str(BACKEND_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _isolate_saved_results(tmp_path, monkeypatch):
    """Keep tests from writing task results into the real backend/data/results folder."""
    from app.services import result_store

    monkeypatch.setattr(result_store, "RESULTS_DIR", tmp_path / "results")
