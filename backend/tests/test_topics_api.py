"""Topic API (recurring-research spec REC-100 – REC-103, REC-114 API side)."""
import json
import logging
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.services import result_store, topic_files as files, topics
from main import app

SCHEDULE = {"every": "week", "weekday": "mon", "time": "07:30", "tz": "Europe/Paris"}
SECRET_QUERY = "What changed in the EU AI Act implementing acts this week?"
SECRET_NAME = "EU AI Act watch"


def _guard(monkeypatch, enabled=True, mode="act"):
    monkeypatch.setattr(
        settings,
        "SECURITY_GUARD",
        settings.SECURITY_GUARD.model_copy(update={"enabled": enabled, "input_mode": mode}),
    )


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(settings, "API_KEY_ENABLED", False)
    _guard(monkeypatch, enabled=False)
    return TestClient(app)


def _create(client, **overrides):
    body = {"name": SECRET_NAME, "query": SECRET_QUERY, "schedule": SCHEDULE, **overrides}
    return client.post("/api/topics", json=body)


def _created(client, **overrides):
    response = _create(client, **overrides)
    assert response.status_code == 201, response.text
    return response.json()


def _write_run(topic_id, started, new_sources=2, status="completed"):
    run_id = str(uuid.uuid4())
    task_id = files.task_id_for_run(run_id)
    record = {
        "run_id": run_id,
        "task_id": task_id,
        "status": status,
        "error_code": None,
        "started_at": started.isoformat(),
        "completed_at": (started + timedelta(minutes=3)).isoformat(),
        "usage": {"input_tokens": 10, "output_tokens": 5, "searches": 1, "tavily_credits": 1},
        "report": "# Report",
        "sources": [{"url": f"https://example.com/{i}"} for i in range(new_sources)],
        "change": {
            "first": False,
            "new_sources": [f"https://example.com/{i}" for i in range(new_sources)],
            "dropped_sources": [],
            "changed": new_sources > 0,
        },
    }
    files.write_json_atomic(files.run_path(topic_id, run_id), record)
    return record


# --- create and validation (REC-100) -------------------------------------------------------------

def test_create_returns_definition_and_writes_file(client):
    task = str(uuid.uuid4())
    body = _created(client, notify={"when": "always"}, created_from_task=task)
    assert files.is_uuid(body["topic_id"])
    assert body["status"] == "active" and body["notify"]["when"] == "always"
    assert body["created_from_task"] == task and body["run_requests"] == 0 and body["resumes"] == 0
    saved = json.loads(files.definition_path(body["topic_id"]).read_text())
    assert saved["query"] == SECRET_QUERY and saved["schedule"]["weekday"] == "mon"


@pytest.mark.parametrize(
    "overrides, field",
    [
        ({"name": ""}, "name"),
        ({"name": "x" * 81}, "name"),
        ({"name": "bad\x07name"}, "name"),
        ({"query": "q" * 2001}, "query"),
        ({"query": "line\x00break"}, "query"),
        ({"schedule": {**SCHEDULE, "tz": "Mars/Olympus"}}, "schedule"),
        ({"schedule": {"every": "week", "time": "07:30", "tz": "UTC"}}, "schedule"),
        ({"schedule": {**SCHEDULE, "time": "24:00"}}, "schedule"),
        ({"schedule": {**SCHEDULE, "every": "hour"}}, "schedule"),
        ({"notify": {"when": "sometimes"}}, "notify"),
        ({"notify": {"when": "changes", "subscribers": [{"email": "a@b.co"}]}}, "notify"),
        ({"created_from_task": "not-a-uuid"}, "created_from_task"),
        ({"status": "paused"}, "status"),
    ],
)
def test_create_validation_names_the_field(client, overrides, field):
    response = _create(client, **overrides)
    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "validation_error"
    assert any(field in [str(p) for p in err["loc"]] for err in body["detail"])
    assert files.list_topic_ids() == []


def test_daily_schedule_drops_weekday(client):
    body = _created(client, schedule={"every": "day", "weekday": "fri", "time": "23:59", "tz": "UTC"})
    assert body["schedule"]["weekday"] is None


# --- the cap (REC-102) ---------------------------------------------------------------------------

def test_create_beyond_cap_is_409(client, monkeypatch):
    monkeypatch.setattr(settings, "SCHEDULE", settings.SCHEDULE.model_copy(update={"max_active_topics": 2}))
    _created(client)
    paused = _created(client)
    response = _create(client)
    assert response.status_code == 409
    assert response.json()["code"] == "topic_limit"
    assert response.json()["error"]["code"] == "topic_limit"

    # A paused topic doesn't count, and re-activating it checks the cap.
    assert client.patch(f"/api/topics/{paused['topic_id']}", json={"status": "paused"}).status_code == 200
    third = _created(client)
    response = client.patch(f"/api/topics/{paused['topic_id']}", json={"status": "active"})
    assert response.status_code == 409 and response.json()["code"] == "topic_limit"
    response = client.post(f"/api/topics/{paused['topic_id']}/resume")
    assert response.status_code == 409 and response.json()["code"] == "topic_limit"
    assert topics.load_definition(paused["topic_id"]).status == "paused"

    # Deleting one frees a place.
    assert client.delete(f"/api/topics/{third['topic_id']}").status_code == 204
    assert client.post(f"/api/topics/{paused['topic_id']}/resume").status_code == 202


def test_patch_active_topic_to_active_at_cap_is_allowed(client, monkeypatch):
    monkeypatch.setattr(settings, "SCHEDULE", settings.SCHEDULE.model_copy(update={"max_active_topics": 1}))
    topic = _created(client)
    response = client.patch(f"/api/topics/{topic['topic_id']}", json={"status": "active"})
    assert response.status_code == 200
    assert client.post(f"/api/topics/{topic['topic_id']}/resume").status_code == 202


# --- list, get, patch -----------------------------------------------------------------------------

def test_list_shows_effective_state_and_last_run(client):
    a = _created(client)
    b = _created(client)
    c = _created(client)
    deleted = _created(client)
    client.delete(f"/api/topics/{deleted['topic_id']}")
    client.patch(f"/api/topics/{c['topic_id']}", json={"status": "paused"})
    files.write_json_atomic(files.state_path(a["topic_id"]), {
        "next_due": "2026-10-19T05:30:00+00:00",
        "hold": "failures",
        "skipped": {"reason": "daily_cap", "at": "2026-10-18T05:30:00+00:00"},
    })
    files.write_json_atomic(files.state_path(b["topic_id"]), {"hold": "budget"})
    now = datetime(2026, 10, 18, 5, 30, tzinfo=timezone.utc)
    _write_run(a["topic_id"], now - timedelta(days=7), new_sources=5)
    latest = _write_run(a["topic_id"], now, new_sources=3)

    rows = {row["topic_id"]: row for row in client.get("/api/topics").json()["topics"]}
    assert set(rows) == {a["topic_id"], b["topic_id"], c["topic_id"]}
    assert rows[a["topic_id"]]["state"] == "held_failures"
    assert rows[b["topic_id"]]["state"] == "held_budget"
    assert rows[c["topic_id"]]["state"] == "paused"
    row = rows[a["topic_id"]]
    assert row["next_due"] == "2026-10-19T05:30:00+00:00"
    assert row["skipped"]["reason"] == "daily_cap"
    assert row["last_run"]["run_id"] == latest["run_id"]
    assert row["last_new_sources"] == 3
    assert row["subscriber_count"] == 0 and row["notify"] == {"when": "changes"}
    assert row["query"] == SECRET_QUERY
    assert rows[b["topic_id"]]["last_run"] is None and rows[b["topic_id"]]["last_new_sources"] is None


def test_get_and_patch(client):
    topic = _created(client)
    tid = topic["topic_id"]
    got = client.get(f"/api/topics/{tid}").json()
    assert got["name"] == SECRET_NAME and got["state"] == "active" and got["notify"]["subscribers"] == []

    new_schedule = {"every": "day", "time": "06:00", "tz": "America/New_York"}
    response = client.patch(f"/api/topics/{tid}", json={
        "name": "Renamed", "schedule": new_schedule, "notify": {"when": "always"}, "status": "paused"})
    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "Renamed" and body["schedule"]["tz"] == "America/New_York"
    assert body["notify"]["when"] == "always" and body["status"] == "paused" and body["state"] == "paused"
    saved = topics.load_definition(tid)
    assert saved.name == "Renamed" and saved.status == "paused" and saved.query == SECRET_QUERY


@pytest.mark.parametrize("body", [
    {"status": "deleted"},
    {"name": ""},
    {"query": "can't change the question"},
    {"schedule": {"every": "week", "time": "07:00", "tz": "UTC"}},
    {"add_subscriber": "a@b.co"},
])
def test_patch_validation(client, body):
    tid = _created(client)["topic_id"]
    assert client.patch(f"/api/topics/{tid}", json=body).status_code == 422
    assert topics.load_definition(tid).status == "active"


def test_run_now_and_resume_counters(client):
    tid = _created(client)["topic_id"]
    assert client.post(f"/api/topics/{tid}/run").json() == {"run_requests": 1}
    response = client.post(f"/api/topics/{tid}/run")
    assert response.status_code == 202 and response.json() == {"run_requests": 2}

    client.patch(f"/api/topics/{tid}", json={"status": "paused"})
    response = client.post(f"/api/topics/{tid}/resume")
    assert response.status_code == 202 and response.json() == {"status": "active", "resumes": 1}
    saved = topics.load_definition(tid)
    assert saved.run_requests == 2 and saved.resumes == 1 and saved.status == "active"


# --- runs -----------------------------------------------------------------------------------------

def test_runs_newest_first_capped(client, monkeypatch):
    monkeypatch.setattr(settings, "SCHEDULE", settings.SCHEDULE.model_copy(update={"history_runs_kept": 3}))
    tid = _created(client)["topic_id"]
    start = datetime(2026, 10, 1, tzinfo=timezone.utc)
    records = [_write_run(tid, start + timedelta(days=i), new_sources=i) for i in range(5)]
    runs = client.get(f"/api/topics/{tid}/runs").json()["runs"]
    assert [r["run_id"] for r in runs] == [records[4]["run_id"], records[3]["run_id"], records[2]["run_id"]]
    assert runs[0]["change"] == {"first": False, "new_sources": 4, "changed": True}
    assert "report" not in runs[0]

    detail = client.get(f"/api/topics/{tid}/runs/{records[1]['run_id']}")
    assert detail.status_code == 200
    assert detail.json()["report"] == "# Report" and len(detail.json()["sources"]) == 1


def test_runs_empty(client):
    tid = _created(client)["topic_id"]
    assert client.get(f"/api/topics/{tid}/runs").json() == {"runs": []}


# --- 404s -----------------------------------------------------------------------------------------

@pytest.mark.parametrize("method, suffix", [
    ("get", ""), ("patch", ""), ("delete", ""), ("post", "/run"), ("post", "/resume"), ("get", "/runs"),
])
def test_unknown_bad_and_deleted_ids_are_404(client, method, suffix):
    deleted = _created(client)["topic_id"]
    assert client.delete(f"/api/topics/{deleted}").status_code == 204
    for topic_id in (str(uuid.uuid4()), "not-a-uuid", "..%2F..%2Fetc", deleted):
        kwargs = {"json": {"name": "x"}} if method == "patch" else {}
        response = getattr(client, method)(f"/api/topics/{topic_id}{suffix}", **kwargs)
        assert response.status_code == 404, (topic_id, response.text)


def test_unknown_run_is_404(client):
    tid = _created(client)["topic_id"]
    assert client.get(f"/api/topics/{tid}/runs/{uuid.uuid4()}").status_code == 404
    assert client.get(f"/api/topics/{tid}/runs/not-a-uuid").status_code == 404


def test_api_key_required(client, monkeypatch):
    monkeypatch.setattr(settings, "API_KEY_ENABLED", True)
    assert client.get("/api/topics").status_code == 401
    assert _create(client).status_code == 401


# --- delete (REC-114) -----------------------------------------------------------------------------

def test_delete_without_scheduler_removes_everything(client):
    tid = _created(client)["topic_id"]
    record = _write_run(tid, datetime(2026, 10, 1, tzinfo=timezone.utc))
    files.write_json_atomic(files.state_path(tid), {"next_due": "2026-10-19T05:30:00+00:00"})
    result_store.save_result(record["task_id"], {"task_id": record["task_id"], "status": "completed"})

    assert client.delete(f"/api/topics/{tid}").status_code == 204
    assert not files.topic_dir(tid).exists()
    assert not files.definition_path(tid).exists()
    assert result_store.load_result(record["task_id"]) is None
    assert not files.scheduler_lock_path().exists()
    assert client.get(f"/api/topics/{tid}").status_code == 404


def test_delete_with_live_scheduler_leaves_tombstone(client):
    tid = _created(client)["topic_id"]
    record = _write_run(tid, datetime(2026, 10, 1, tzinfo=timezone.utc))
    result_store.save_result(record["task_id"], {"task_id": record["task_id"], "status": "completed"})
    lock = {"pid": 1, "heartbeat": files.utcnow().isoformat(), "enabled": True, "owner": "scheduler"}
    files.write_json_atomic(files.scheduler_lock_path(), lock)

    assert client.delete(f"/api/topics/{tid}").status_code == 204
    assert topics.load_definition(tid).status == "deleted"
    assert files.topic_dir(tid).exists() and result_store.load_result(record["task_id"]) is not None
    assert files.read_json(files.scheduler_lock_path()) == lock
    assert client.get(f"/api/topics/{tid}").status_code == 404
    assert client.get("/api/topics").json() == {"topics": []}


# --- intake guard (REC-103) -----------------------------------------------------------------------

def _fake_decide(outcome, policy="test-policy"):
    from app.security.types import Decision

    def decide(principal, action, resource, context, shadow=None):
        if context.get("source") == "user_turn":
            return Decision(outcome, (policy,), "Test reason", bool(shadow), 1.0)
        return Decision("allow", ("baseline-permit",), "ok", bool(shadow), 1.0)

    return decide


@pytest.mark.parametrize("outcome, code", [
    ("decline", "declined"), ("escalate", "escalated"), ("consent", "consent_required"),
])
def test_intake_act_mode_refuses(client, monkeypatch, outcome, code):
    _guard(monkeypatch, enabled=True, mode="act")
    monkeypatch.setattr("app.security.hop_guard.decide", _fake_decide(outcome, f"user-{outcome}"))
    monkeypatch.setattr("app.security.records.record_escalation", lambda **kw: None)
    response = _create(client)
    assert response.status_code == 403
    body = response.json()
    assert body["code"] == code and body["policy_id"] == f"user-{outcome}"
    assert "consent_token" not in body
    assert files.list_topic_ids() == []


def test_intake_shadow_mode_records_and_creates(client, monkeypatch):
    _guard(monkeypatch, enabled=True, mode="shadow")
    seen = []

    def decide(principal, action, resource, context, shadow=None):
        seen.append((principal, context.get("source"), shadow))
        from app.security.types import Decision
        return Decision("decline", ("user-harm-decline",), "Test reason", bool(shadow), 1.0)

    monkeypatch.setattr("app.security.hop_guard.decide", decide)
    response = _create(client)
    assert response.status_code == 201
    assert seen == [("topic_create", "user_turn", True)]
    assert len(files.list_topic_ids()) == 1


def test_intake_act_mode_allows(client, monkeypatch):
    _guard(monkeypatch, enabled=True, mode="act")
    monkeypatch.setattr("app.security.hop_guard.decide", _fake_decide("allow"))
    assert _create(client).status_code == 201


# --- privacy (RNFR-03) ----------------------------------------------------------------------------

def test_logs_carry_no_query_or_name(client, caplog):
    with caplog.at_level(logging.DEBUG):
        tid = _created(client)["topic_id"]
        client.patch(f"/api/topics/{tid}", json={"name": "Another secret name"})
        client.post(f"/api/topics/{tid}/run")
        client.delete(f"/api/topics/{tid}")
    text = caplog.text
    assert tid in text
    for secret in (SECRET_QUERY, SECRET_NAME, "Another secret name"):
        assert secret not in text


# --- startup log (REC-118) ------------------------------------------------------------------------

def test_scheduling_off_logged_once_without_token(monkeypatch, caplog):
    from app.core import startup_checks

    monkeypatch.setattr(settings, "SCHEDULER_TOKEN", None)
    monkeypatch.setattr(startup_checks, "_scheduler_token_logged", False)
    with caplog.at_level(logging.INFO, logger="app.core.startup_checks"):
        startup_checks.check_scheduler_token()
        startup_checks.check_scheduler_token()
    assert caplog.text.count("scheduling off: no WEBAGENT_SCHEDULER_TOKEN") == 1

    caplog.clear()
    monkeypatch.setattr(settings, "SCHEDULER_TOKEN", "s3cret")
    monkeypatch.setattr(startup_checks, "_scheduler_token_logged", False)
    with caplog.at_level(logging.INFO, logger="app.core.startup_checks"):
        startup_checks.check_scheduler_token()
    assert "scheduling off" not in caplog.text



def test_delete_leaves_no_question_name_or_address_on_disk(client):
    """The tombstone is scrubbed at once, even while a live scheduler still has to remove the files."""
    tid = _created(client)["topic_id"]
    files.write_json_atomic(files.scheduler_lock_path(), {
        "pid": 1, "heartbeat": files.utcnow().isoformat(), "enabled": True, "owner": "scheduler"})
    files.write_json_atomic(files.state_path(tid), {"next_due": None})
    assert client.delete(f"/api/topics/{tid}").status_code == 204
    raw = files.definition_path(tid).read_text()
    assert SECRET_QUERY not in raw and SECRET_NAME not in raw and '"deleted"' in raw
