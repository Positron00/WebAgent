"""Backend modes (REQ-070-072): resolve_mode, dispatch to the direct path or the workflow, route fields."""
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.core import metrics
from app.core.config import settings
from app.services import triage_model
from app.services.routing import resolve_mode
from main import app
from tests.fakes.triage import FakeTriageModel
from tests.fakes.workflow import StreamingFakeWorkflow

client = TestClient(app)


@pytest.mark.parametrize(
    "mode,legacy,expected",
    [
        (None, None, "off"),
        (None, True, "always"),
        (None, False, "off"),
        ("auto", None, "auto"),
        ("auto", True, "auto"),
        ("always", False, "always"),
        ("off", True, "off"),
    ],
)
def test_resolve_mode_table(mode, legacy, expected):
    assert resolve_mode(mode, legacy) == expected


class _Workflow(StreamingFakeWorkflow):
    def __init__(self):
        self.called = False

    async def ainvoke(self, state):
        self.called = True
        return {"query": state.query, "context": state.context, "final_report": {"title": "R", "content": "ok"}}


def _run(body: dict, p_agentic: float = 0.05):
    """Post a task with a fake classifier, workflow and direct LLM; return (final poll body, workflow)."""
    triage_model.set_model_for_tests(FakeTriageModel(p=p_agentic, category="coding help"))
    workflow = _Workflow()
    with patch("app.services.task_manager.get_agent_workflow", return_value=workflow), patch(
        "app.services.task_manager.SupervisorAgent"
    ) as sup:
        sup.return_value.answer_directly = AsyncMock(return_value="direct answer")
        task_id = client.post("/api/task", json=body).json()["task_id"]
        for _ in range(30):
            polled = client.get(f"/api/task/{task_id}").json()
            if polled["status"] != "processing":
                break
    return polled, workflow


@pytest.mark.parametrize(
    "body,route,decided_by",
    [
        ({"query": "explain list comprehensions"}, "direct", "mode"),  # neither field: off
        ({"query": "explain list comprehensions", "agentic": False}, "direct", "mode"),
        ({"query": "explain list comprehensions", "agentic": True}, "agentic", "mode"),
        ({"query": "explain list comprehensions", "mode": "off", "agentic": True}, "direct", "mode"),
        ({"query": "explain list comprehensions", "mode": "always"}, "agentic", "mode"),
        ({"query": "explain list comprehensions", "mode": "auto"}, "direct", "model"),
    ],
)
def test_task_mode_combinations(body, route, decided_by):
    polled, workflow = _run(body)
    assert polled["status"] == "completed"
    assert polled["result"]["route"] == route
    assert polled["result"]["decided_by"] == decided_by
    assert workflow.called == (route == "agentic")


def test_auto_agentic_runs_workflow_and_reports_p():
    polled, workflow = _run({"query": "compare CRM pricing this year", "mode": "auto"}, p_agentic=0.9)
    assert workflow.called
    assert polled["result"]["route"] == "agentic"
    assert polled["result"]["p_agentic"] == 0.9


def test_image_in_auto_goes_direct_and_is_ignored():
    polled, workflow = _run({"query": "describe this", "mode": "auto", "image": "data:image/png;base64,abc"})
    assert not workflow.called
    assert polled["result"]["route"] == "direct"
    assert polled["result"]["image_ignored"] is True


def test_image_on_workflow_path_is_also_flagged():
    polled, _ = _run({"query": "describe this", "mode": "always", "image": "data:image/png;base64,abc"})
    assert polled["result"]["image_ignored"] is True


def test_auto_and_triage_endpoint_agree():
    triage_model.set_model_for_tests(FakeTriageModel(p=0.2, category="coding help"))
    triage = client.post("/api/triage", json={"query": "explain recursion"}).json()
    polled, _ = _run({"query": "explain recursion", "mode": "auto"}, p_agentic=0.2)
    assert polled["result"]["route"] == triage["route"]
    assert polled["result"]["decided_by"] == triage["decided_by"]


def test_override_routes_agentic_counts_and_collects(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "TRIAGE", settings.TRIAGE.model_copy(update={"collect": True}))
    sample = ("webagent_triage_decisions_total", {"route": "agentic", "decided_by": "override"})
    before = metrics.WEBAGENT_REGISTRY.get_sample_value(*sample) or 0.0
    overrides_before = metrics.WEBAGENT_REGISTRY.get_sample_value("webagent_triage_overrides_total") or 0.0
    polled, workflow = _run({"query": "explain recursion", "mode": "auto", "triage_override": True})
    assert workflow.called
    assert polled["result"]["decided_by"] == "override"
    assert metrics.WEBAGENT_REGISTRY.get_sample_value(*sample) == before + 1
    assert metrics.WEBAGENT_REGISTRY.get_sample_value("webagent_triage_overrides_total") == overrides_before + 1
    collected = list((tmp_path / "triage" / "collected").glob("*.jsonl"))
    assert collected and '"kind": "override"' in collected[0].read_text()


def test_chat_completions_modes_choose_the_path():
    """Neither field means off (direct); `mode` and the old `agentic_toggle` still pick the workflow."""
    with patch("app.api.endpoints.frontend.task_manager") as tm:
        r = client.post("/api/v1/chat/completions", json={"messages": [{"role": "user", "content": "q"}]})
        assert r.status_code == 200
        client.post(
            "/api/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "q"}], "mode": "always"},
        )
        client.post(
            "/api/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "q"}], "agentic_toggle": True},
        )
    assert [c.kwargs["direct"] for c in tm.start.call_args_list] == [True, False, False]


def test_deprecated_endpoints_marked_in_openapi():
    spec = client.get(f"{settings.API_V1_STR}/openapi.json").json()
    assert spec["paths"][f"{settings.API_V1_STR}/chat/"]["post"]["deprecated"] is True
    assert spec["paths"][f"{settings.API_V1_STR}/tasks/create"]["post"]["deprecated"] is True
