"""Scheduler fields on POST /api/task (REC-107) and the effects of a scheduled run (REC-108)."""
import json
from uuid import uuid4

import numpy as np
import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch

from app.api.endpoints import task as task_endpoint
from app.core import drift_store
from app.core.config import settings
from app.core.metrics import WEBAGENT_REGISTRY
from app.core.scheduler_auth import require_scheduler
from app.security import records
from app.security.types import Decision
from app.services import result_store, run_origin, task_budget, triage_model
from app.services.drift_embeddings import DriftRecordingEmbeddings
from main import app
from tests.fakes.triage import FakeTriageModel
from tests.fakes.workflow import StreamingFakeWorkflow

client = TestClient(app)

TOKEN = "scheduler-token-for-tests-0123456789"
HEADER = {"X-WebAgent-Scheduler": TOKEN}
TOPIC = str(uuid4())


@pytest.fixture
def token(monkeypatch):
    monkeypatch.setattr(settings, "SCHEDULER_TOKEN", TOKEN)


def scheduled(**fields):
    body = {"query": "EU AI rules news", "mode": "always", "origin": "schedule", "topic_id": TOPIC,
            "token_budget": 5000}
    body.update(fields)
    return {k: v for k, v in body.items() if v is not None}


class _Embeddings:
    def embed_documents(self, texts):
        return [[1.0, 0.0, 2.0] for _ in texts]

    def embed_query(self, text):
        return [1.0, 0.0, 2.0]


class _Workflow(StreamingFakeWorkflow):
    """Records what the run sees: budget, topic_run, config; embeds and collects like a real run."""

    def __init__(self):
        self.seen = {}

    async def ainvoke(self, state):
        self.seen.update(limit=task_budget.current().limit, topic_run=run_origin.topic_run.get(),
                         previous=state.context.get("previous_messages"))
        emb = DriftRecordingEmbeddings(_Embeddings(), "local")
        for _ in range(drift_store.MIN_SAMPLES):
            emb.embed_query("q")
        records.record_collection(
            checkpoint=2, action="tool::web_search", decision=Decision("allow", (), "ok", True, 1.0),
            state_context=state.context, args={},
        )
        return {"query": state.query, "context": state.context, "final_report": {"title": "R", "content": "ok"}}

    async def astream(self, state, stream_mode=None, config=None):
        self.seen["config"] = config
        async for item in super().astream(state, stream_mode=stream_mode):
            yield item


@pytest.fixture
def collecting(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "SECURITY_GUARD",
                        settings.SECURITY_GUARD.model_copy(update={"collect": True}))
    return tmp_path / "security" / "collected"


def _post(body, headers=None):
    return client.post("/api/task", json=body, headers=headers or {})


def _run(body, headers=None):
    workflow = _Workflow()
    with patch("app.services.task_manager.get_agent_workflow", return_value=workflow), patch(
        "app.services.task_manager.SupervisorAgent"
    ) as sup:
        sup.return_value.answer_directly = AsyncMock(return_value="direct")
        reply = _post(body, headers)
        assert reply.status_code == 200, reply.text
        task_id = reply.json()["task_id"]
        for _ in range(30):
            polled = client.get(f"/api/task/{task_id}").json()
            if polled["status"] != "processing":
                break
    return task_id, polled, workflow


# --- the credential ---

@pytest.mark.parametrize("headers", [{}, {"X-WebAgent-Scheduler": "wrong"}, {"X-WebAgent-Scheduler": ""}])
def test_missing_or_wrong_header_is_422(token, headers):
    reply = _post(scheduled(), headers)
    assert reply.status_code == 422
    assert reply.json()["code"] == "scheduler_fields_need_token"


def test_api_key_alone_is_not_enough(token, monkeypatch):
    monkeypatch.setattr(settings, "API_KEY_ENABLED", True)
    monkeypatch.setenv("WEBAGENT_API_KEY", "api-key-for-tests")
    reply = _post(scheduled(), {"X-API-Key": "api-key-for-tests"})
    assert (reply.status_code, reply.json()["code"]) == (422, "scheduler_fields_need_token")


def test_token_unset_refuses_every_field_even_with_a_header(monkeypatch):
    monkeypatch.setattr(settings, "SCHEDULER_TOKEN", None)
    reply = _post(scheduled(), HEADER)
    assert (reply.status_code, reply.json()["code"]) == (422, "scheduler_fields_need_token")


@pytest.mark.parametrize("field, value", [("origin", "schedule"), ("topic_id", TOPIC), ("task_id", str(uuid4())),
                                          ("token_budget", 100)])
def test_each_field_alone_needs_the_token(token, field, value):
    reply = _post({"query": "q", "mode": "always", field: value})
    assert (reply.status_code, reply.json()["code"]) == (422, "scheduler_fields_need_token")


def test_right_header_runs_the_task(token):
    _, polled, _ = _run(scheduled(), HEADER)
    assert polled["status"] == "completed"


def test_header_without_fields_is_an_ordinary_task(token):
    _, polled, workflow = _run({"query": "q", "mode": "always"}, HEADER)
    assert polled["status"] == "completed"
    assert "origin" not in polled["result"] and workflow.seen["topic_run"] is None


def test_the_secret_is_not_in_the_reply_or_logs(token, caplog):
    reply = _post(scheduled(), {"X-WebAgent-Scheduler": "wrong"})
    assert TOKEN not in reply.text and TOKEN not in caplog.text


def test_require_scheduler_dependency(monkeypatch):
    mini = FastAPI()

    @mini.get("/only-scheduler", dependencies=[Depends(require_scheduler)])
    def route():
        return {"ok": True}

    mini_client = TestClient(mini)
    monkeypatch.setattr(settings, "SCHEDULER_TOKEN", None)
    assert mini_client.get("/only-scheduler", headers=HEADER).status_code == 404
    monkeypatch.setattr(settings, "SCHEDULER_TOKEN", TOKEN)
    assert mini_client.get("/only-scheduler").status_code == 403
    assert mini_client.get("/only-scheduler", headers={"X-WebAgent-Scheduler": "wrong"}).status_code == 403
    assert mini_client.get("/only-scheduler", headers=HEADER).status_code == 200


# --- the field rules ---

@pytest.mark.parametrize("body", [
    scheduled(origin=None),                       # fields without origin
    scheduled(mode="auto"),
    scheduled(mode=None),
    scheduled(topic_id="not-a-uuid"),
    scheduled(topic_id=None),
    scheduled(task_id="../../etc/passwd"),
    scheduled(token_budget=None),
    scheduled(token_budget=0),
    scheduled(token_budget=-5),
])
def test_invalid_scheduler_fields_are_422(token, body):
    assert _post(body, HEADER).status_code == 422


def test_token_budget_upper_bound(token):
    cap = settings.SCHEDULE.max_token_budget
    assert _post(scheduled(token_budget=cap + 1), HEADER).status_code == 422
    _, polled, workflow = _run(scheduled(token_budget=cap), HEADER)
    assert polled["status"] == "completed" and workflow.seen["limit"] == cap


def test_caller_chosen_task_id_is_used_and_a_duplicate_is_409(token):
    chosen = str(uuid4())
    task_id, polled, _ = _run(scheduled(task_id=chosen), HEADER)
    assert task_id == chosen and polled["status"] == "completed"
    reply = _post(scheduled(task_id=chosen), HEADER)
    assert (reply.status_code, reply.json()["code"]) == (409, "task_exists")

    other = str(uuid4())
    result_store.save_result(other, {"task_id": other, "status": "completed", "result": {"content": "x"}})
    assert _post(scheduled(task_id=other), HEADER).status_code == 409


# --- the effects of a scheduled run ---

def test_triage_is_neither_run_nor_collected(token, monkeypatch):
    monkeypatch.setattr(settings, "TRIAGE", settings.TRIAGE.model_copy(update={"mode": "shadow", "collect": True}))
    triage_model.set_model_for_tests(FakeTriageModel(p=0.1, category="coding help"))
    calls = []
    monkeypatch.setattr(task_endpoint, "classify", lambda *a, **k: calls.append("classify"))
    monkeypatch.setattr(task_endpoint, "choose_route", lambda *a, **k: calls.append("choose_route"))
    monkeypatch.setattr(task_endpoint.triage_collect, "record", lambda *a, **k: calls.append("record"))

    _, polled, _ = _run(scheduled(), HEADER)
    _, polled_override, _ = _run(scheduled(triage_override=True), HEADER)
    assert polled["status"] == polled_override["status"] == "completed"
    assert calls == []


def test_budget_origin_topic_and_metadata_of_a_scheduled_run(token):
    task_id, polled, workflow = _run(scheduled(token_budget=1234, previous_messages=[
        {"role": "user", "content": "earlier"}, {"role": "assistant", "content": "answer"}]), HEADER)
    assert polled["status"] == "completed"
    assert workflow.seen["limit"] == 1234
    assert workflow.seen["topic_run"] == TOPIC
    assert workflow.seen["previous"] == []  # no history
    assert workflow.seen["config"] == {"metadata": {"origin": "schedule", "topic_id": TOPIC}}
    assert (polled["result"]["origin"], polled["result"]["topic_id"]) == ("schedule", TOPIC)
    saved = result_store.load_result(task_id)
    assert (saved["origin"], saved["topic_id"]) == ("schedule", TOPIC)
    assert (saved["route"], saved["decided_by"]) == ("agentic", "schedule")
    assert (polled["result"]["route"], polled["result"]["decided_by"]) == ("agentic", "schedule")
    assert run_origin.topic_run.get() is None  # reset after the run


def test_interactive_run_passes_no_config_and_no_topic():
    _, polled, workflow = _run({"query": "q", "mode": "always"})
    assert polled["status"] == "completed"
    assert workflow.seen["config"] is None and workflow.seen["topic_run"] is None


def test_collection_rows_carry_origin_and_topic(token, collecting, monkeypatch):
    def intake(query, history, has_image, consent_token, task_id):
        records.record_collection(checkpoint=1, action="intake", decision=Decision("allow", (), "ok", True, 1.0),
                                  state_context={"task_id": task_id}, args={})
        return False

    monkeypatch.setattr(task_endpoint, "run_intake_guard", intake)
    _run(scheduled(), HEADER)
    _run({"query": "q", "mode": "always"})
    rows = [json.loads(line) for f in collecting.glob("*.jsonl") for line in f.read_text().splitlines()]
    scheduled_rows = [r for r in rows if r.get("origin") == "schedule"]
    assert {r["action"] for r in scheduled_rows} == {"intake", "tool::web_search"}
    assert all(r["topic_id"] == TOPIC for r in scheduled_rows)
    interactive_rows = [r for r in rows if "origin" not in r]
    assert len(interactive_rows) == 2 and all("topic_id" not in r for r in interactive_rows)


def test_drift_is_not_observed_during_a_topic_run(token):
    _run(scheduled(), HEADER)
    assert not [s for s in drift_store.read_samples() if s.detector.startswith("embeddings:")]
    _run({"query": "q", "mode": "always"})
    assert [s for s in drift_store.read_samples() if s.detector.startswith("embeddings:")]


def test_drift_wrapper_skips_while_topic_run_is_set():
    emb = DriftRecordingEmbeddings(_Embeddings(), "local")
    reset = run_origin.topic_run.set(TOPIC)
    try:
        for _ in range(drift_store.MIN_SAMPLES):
            assert np.allclose(emb.embed_query("q"), [1.0, 0.0, 2.0])  # still passes the vector through
    finally:
        run_origin.topic_run.reset(reset)
    assert not drift_store.read_samples()


def _count(status, origin):
    return WEBAGENT_REGISTRY.get_sample_value("webagent_tasks_total", {"status": status, "origin": origin}) or 0.0


def _durations(status, origin):
    return WEBAGENT_REGISTRY.get_sample_value(
        "webagent_task_duration_seconds_count", {"status": status, "origin": origin}) or 0.0


def test_metrics_have_an_origin_label(token):
    before = {o: (_count("started", o), _count("completed", o), _durations("completed", o))
              for o in ("schedule", "interactive")}
    _run(scheduled(), HEADER)
    after_schedule = {o: (_count("started", o), _count("completed", o), _durations("completed", o))
                      for o in ("schedule", "interactive")}
    assert after_schedule["schedule"] == tuple(v + 1 for v in before["schedule"])
    assert after_schedule["interactive"] == before["interactive"]

    _run({"query": "q", "mode": "always"})
    assert _count("completed", "interactive") == before["interactive"][1] + 1
    assert _count("completed", "schedule") == after_schedule["schedule"][1]


def test_topic_creation_rows_carry_their_origin(collecting):
    """The intake check when a topic is created is tagged origin topic_create (REC-103)."""
    records.record_collection(checkpoint=1, action="intake", decision=Decision("allow", (), "ok", True, 1.0),
                              state_context={"task_id": "topic-x", "origin": "topic_create", "topic_id": TOPIC},
                              args={})
    rows = [json.loads(line) for f in collecting.glob("*.jsonl") for line in f.read_text().splitlines()]
    assert rows[-1]["origin"] == "topic_create" and rows[-1]["topic_id"] == TOPIC
