"""Shadow mode (REQ-040): label workflow requests without changing the route; never fail the task."""
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.services import triage_model
from main import app
from tests.fakes.triage import FakeTriageModel
from tests.fakes.workflow import StreamingFakeWorkflow

client = TestClient(app)


class FakeWorkflow(StreamingFakeWorkflow):
    async def ainvoke(self, state):
        return {"query": state.query, "context": state.context, "final_report": {"title": "R", "content": "done"}}


@pytest.fixture
def shadow(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "TRIAGE", settings.TRIAGE.model_copy(update={"mode": "shadow", "collect": True}))
    triage_model.set_model_for_tests(FakeTriageModel(p=0.1, category="coding help"))
    return tmp_path / "triage" / "collected"


def _run(body: dict) -> dict:
    with patch("app.services.task_manager.get_agent_workflow", return_value=FakeWorkflow()), patch(
        "app.services.task_manager.SupervisorAgent"
    ) as sup:
        sup.return_value.answer_directly = AsyncMock(return_value="direct")
        task_id = client.post("/api/task", json=body).json()["task_id"]
        for _ in range(30):
            polled = client.get(f"/api/task/{task_id}").json()
            if polled["status"] != "processing":
                break
    return polled


def test_shadow_records_but_still_runs_the_workflow(shadow):
    polled = _run({"query": "explain closures", "agentic": True})
    assert polled["status"] == "completed"
    assert polled["result"]["route"] == "agentic"  # the path that ran
    assert polled["result"]["triage"]["shadow"] is True
    assert polled["result"]["triage"]["route"] == "direct"  # what the classifier would have done
    rows = list(shadow.glob("*.jsonl"))
    assert rows and '"kind": "shadow"' in rows[0].read_text()


def test_shadow_skips_auto_and_off(shadow):
    _run({"query": "explain closures", "mode": "auto"})
    _run({"query": "explain closures", "mode": "off"})
    assert not shadow.exists() or not list(shadow.glob("*.jsonl"))


def test_shadow_skips_overrides(shadow):
    polled = _run({"query": "explain closures", "agentic": True, "triage_override": True})
    assert "shadow" not in polled["result"]["triage"]


def test_act_mode_records_nothing(shadow, monkeypatch):
    monkeypatch.setattr(settings, "TRIAGE", settings.TRIAGE.model_copy(update={"mode": "act"}))
    polled = _run({"query": "explain closures", "agentic": True})
    assert "shadow" not in polled["result"]["triage"]
    assert not shadow.exists()


def test_triage_error_never_fails_the_task(shadow, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("app.api.endpoints.task.classify", boom)
    polled = _run({"query": "research climate policy", "mode": "always"})
    assert polled["status"] == "completed"
    assert "shadow" not in polled["result"]["triage"]
