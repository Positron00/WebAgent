"""The scheduler (recurring-research spec REC-104 – REC-106, REC-111 – REC-115, RNFR-01, RNFR-04).

Offline: a fake clock, a fake backend behind `httpx.MockTransport`, a temporary data folder, and the
scheduler in unthreaded mode, so ticks and polls happen exactly when a test says.
"""
import json
import os
import uuid
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from app import scheduler as sched_mod
from app.core.config import settings
from app.core.config_def import ScheduleConfig
from app.scheduler import Scheduler, SchedulerLock
from app.services import result_store, topic_runs, topics
from app.services import topic_files as files

T0 = datetime(2026, 10, 19, 6, 0, tzinfo=timezone.utc)      # a Monday, 06:00 UTC
TOKEN = "scheduler-test-token"


class Clock:
    def __init__(self, now=T0):
        self.t = now

    def __call__(self):
        return self.t

    def advance(self, **delta):
        self.t += timedelta(**delta)
        return self.t


class FakeBackend:
    """The backend contract: POST /api/task, GET /api/task/{id}, DELETE /api/topics/{id}, /health,
    and phase B's POST /api/topics/{id}/summarize and /notify (design §12a)."""

    def __init__(self, clock, duration=timedelta(minutes=3)):
        self.clock = clock
        self.duration = duration
        self.tasks = {}
        self.posts = []
        self.gets = []
        self.deletes = []
        self.post_reply = None        # (status, body) or an exception to raise
        self.final = {}               # task_id -> (status, error_code), instead of completing
        self.usage = {"input_tokens": 1000, "output_tokens": 500, "searches": 3, "tavily_credits": 4}
        self.sources = [{"url": "https://a.org/x", "title": "A"}, {"url": "https://b.org", "title": "B"}]
        self.security = {}
        self.summaries = []           # (topic_id, body, headers)
        self.summary_reply = (200, {"summary": "- Something new [1]", "changed": True,
                                    "tokens": {"input": 2000, "output": 100}})
        self.notifies = []            # (topic_id, body, headers)
        self.notify_reply = (200, {"sent": 1, "skipped": 0})

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.method == "GET" and path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.method == "POST" and path == "/api/task":
            body = json.loads(request.content)
            self.posts.append((dict(request.headers), body))
            if isinstance(self.post_reply, Exception):
                raise self.post_reply
            if self.post_reply is not None:
                status, reply = self.post_reply
                return httpx.Response(status, json=reply)
            if body["task_id"] in self.tasks:
                return httpx.Response(409, json={"code": "task_exists"})
            self.tasks[body["task_id"]] = {"created": self.clock(), "topic_id": body["topic_id"]}
            return httpx.Response(202, json={"task_id": body["task_id"], "status": "processing"})
        if request.method == "GET" and path.startswith("/api/task/"):
            task_id = path.rsplit("/", 1)[1]
            self.gets.append(task_id)
            task = self.tasks.get(task_id)
            if task is None:
                return httpx.Response(404, json={"error": {"code": "not_found"}})
            if task_id in self.final:
                status, code = self.final[task_id]
                return httpx.Response(200, json={"status": status, "error_code": code,
                                                 "result": {"usage": self.usage, "error_code": code}})
            if self.clock() - task["created"] < self.duration:
                return httpx.Response(200, json={"status": "processing", "result": None})
            result = {"title": "Report", "content": "report text", "sources": list(self.sources),
                      "usage": dict(self.usage), "security": dict(self.security)}
            result_store.save_result(task_id, {"task_id": task_id, "status": "completed", "result": result})
            return httpx.Response(200, json={"status": "completed", "result": result})
        if request.method == "POST" and path.startswith("/api/topics/") and path.endswith("/summarize"):
            self.summaries.append((path.split("/")[3], json.loads(request.content), dict(request.headers)))
            if isinstance(self.summary_reply, Exception):
                raise self.summary_reply
            status, reply = self.summary_reply
            return httpx.Response(status, json=reply)
        if request.method == "POST" and path.startswith("/api/topics/") and path.endswith("/notify"):
            self.notifies.append((path.split("/")[3], json.loads(request.content), dict(request.headers)))
            if isinstance(self.notify_reply, Exception):
                raise self.notify_reply
            status, reply = self.notify_reply
            return httpx.Response(status, json=reply)
        if request.method == "DELETE" and path.startswith("/api/topics/"):
            self.deletes.append((path.rsplit("/", 1)[1], dict(request.headers)))
            return httpx.Response(204)
        return httpx.Response(404)


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(result_store, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(settings, "SCHEDULER_TOKEN", TOKEN)
    monkeypatch.setenv("WEBAGENT_API_KEY", "api-key-test")
    clock = Clock()
    backend = FakeBackend(clock)
    client = httpx.Client(base_url="http://backend", transport=httpx.MockTransport(backend.handler))

    def make(**cfg):
        config = ScheduleConfig(**{"enabled": True, **cfg})
        return Scheduler(client=client, now=clock, config=config, threaded=False,
                         max_follow_up_questions=3, search_cap=5, price_input=0.35, price_output=1.5)

    class Env:
        pass

    e = Env()
    e.clock, e.backend, e.client, e.make, e.dir = clock, backend, client, make, tmp_path
    return e


def add_topic(every="day", at="06:00", tz="UTC", weekday=None, status="active", **extra):
    defn = topics.TopicDefinition(
        topic_id=str(uuid.uuid4()), name="name", query="what is new in x", status=status,
        schedule={"every": every, "time": at, "tz": tz, "weekday": weekday},
        created_at=T0 - timedelta(days=30), **extra)
    topics.save_new_definition(defn)
    return defn.topic_id


def edit(topic_id, **changes):
    def change(defn):
        for key, value in changes.items():
            setattr(defn, key, value)
    topics.update_definition(topic_id, change)


def set_state(topic_id, **changes):
    state = topics.load_state(topic_id)
    state.update(changes)
    files.write_json_atomic(files.state_path(topic_id), state)


def runs(topic_id):
    return topic_runs.list_records(topic_id)


def drive(s, clock, minutes, step=15):
    """The main loop on a fake clock: polls every 15 s, a tick each minute or when a run frees a slot."""
    end = clock() + timedelta(minutes=minutes)
    last_tick = None
    while clock() < end:
        finished = s.pump(clock())
        if finished or last_tick is None or clock() - last_tick >= timedelta(seconds=60):
            s.tick(clock())
            last_tick = clock()
        clock.advance(seconds=step)


def counters():
    return files.read_json(files.counters_path()) or {}


# --- due selection, no backfill, ordering, concurrency (REC-104) ----------------------------------

def test_a_new_topic_gets_state_with_the_next_slot_and_runs_when_due(env):
    env.clock.t = T0 - timedelta(hours=1)
    topic_id = add_topic(at="06:00")
    s = env.make()
    s.tick()
    state = topics.load_state(topic_id)
    assert files.parse_time(state["next_due"]) == T0
    assert env.backend.posts == []
    env.clock.t = T0
    s.tick()
    assert len(env.backend.posts) == 1
    assert files.parse_time(topics.load_state(topic_id)["next_due"]) == T0 + timedelta(days=1)


def test_the_post_carries_the_scheduler_fields_and_headers(env):
    topic_id = add_topic()
    s = env.make()
    s.tick()
    set_state(topic_id, next_due=T0.isoformat())
    s.tick()
    headers, body = env.backend.posts[0]
    [record] = runs(topic_id)
    assert body == {"query": "what is new in x", "mode": "always", "origin": "schedule", "topic_id": topic_id,
                    "task_id": files.task_id_for_run(record["run_id"]), "token_budget": 60000}
    assert headers["x-webagent-scheduler"] == TOKEN
    assert headers["x-webagent-client"] == f"schedule:{topic_id}"
    assert headers["x-api-key"] == "api-key-test"
    assert record["status"] == "running" and record["task_id"] == body["task_id"]


def test_missed_slots_are_not_backfilled_but_counted(env):
    topic_id = add_topic()
    s = env.make()
    s.tick()
    set_state(topic_id, next_due=(T0 - timedelta(days=3, hours=1)).isoformat())
    s.tick()
    assert len(env.backend.posts) == 1
    assert files.parse_time(topics.load_state(topic_id)["next_due"]) == T0 + timedelta(days=1)
    assert counters()["missed_slots"] == 3
    s.tick()
    assert len(env.backend.posts) == 1


def test_oldest_due_runs_first(env):
    ids = [add_topic() for _ in range(3)]
    s = env.make(max_concurrent_runs=1)
    s.tick()
    for topic_id, hours in zip(ids, (1, 5, 3)):
        set_state(topic_id, next_due=(T0 - timedelta(hours=hours)).isoformat())
    s.tick()
    assert [b["topic_id"] for _, b in env.backend.posts] == [ids[1]]
    drive(s, env.clock, 10)
    assert [b["topic_id"] for _, b in env.backend.posts] == [ids[1], ids[2], ids[0]]


def test_at_most_max_concurrent_runs(env):
    ids = [add_topic() for _ in range(5)]
    s = env.make(max_concurrent_runs=2)
    s.tick()
    for topic_id in ids:
        set_state(topic_id, next_due=T0.isoformat())
    s.tick()
    s.tick()
    assert len(env.backend.posts) == 2 and len(s.jobs) == 2


def test_run_now_starts_a_run_without_moving_next_due(env):
    env.clock.t = T0 - timedelta(hours=3)
    topic_id = add_topic()
    s = env.make()
    s.tick()
    edit(topic_id, run_requests=1)
    s.tick()
    assert len(env.backend.posts) == 1
    state = topics.load_state(topic_id)
    assert state["run_requests_seen"] == 1 and files.parse_time(state["next_due"]) == T0
    assert runs(topic_id)[0]["trigger"] == "request"


def test_a_schedule_change_recomputes_next_due(env):
    topic_id = add_topic(at="06:00")
    s = env.make()
    s.tick()
    edit(topic_id, schedule=topics.Schedule(every="week", weekday="wed", time="09:30", tz="Europe/Paris"))
    s.tick()
    due = files.parse_time(topics.load_state(topic_id)["next_due"])
    assert due == datetime(2026, 10, 21, 7, 30, tzinfo=timezone.utc)   # 09:30 CEST
    assert env.backend.posts == []


def test_a_completed_run_records_report_sources_usage_and_change(env):
    topic_id = add_topic()
    s = env.make()
    s.tick()
    set_state(topic_id, next_due=T0.isoformat())
    s.tick()
    drive(s, env.clock, 5)
    [record] = runs(topic_id)
    assert record["status"] == "completed" and record["error_code"] is None
    assert record["report"] == "report text" and record["title"] == "Report"
    assert [x["url"] for x in record["sources"]] == ["https://a.org/x", "https://b.org"]
    assert record["usage"] == env.backend.usage
    assert record["change"]["first"] is True and len(record["change"]["new_sources"]) == 2
    usage = files.read_json(files.usage_path("2026-10"))
    assert usage["credits"] == 4
    assert usage["dollars"] == pytest.approx(1000 * 0.35 / 1e6 + 500 * 1.5 / 1e6)
    assert counters()["runs"]["completed"] == 1
    assert counters()["run_seconds"]["count"] == 1
    state = topics.load_state(topic_id)
    assert state["last_run"]["run_id"] == record["run_id"] and state["current_run"] is None

    # The next run with one new source counts a change.
    env.backend.sources = [{"url": "https://www.a.org/x/"}, {"url": "https://c.org"}]
    edit(topic_id, run_requests=1)
    drive(s, env.clock, 5)
    newest = runs(topic_id)[0]
    assert {k: newest["change"][k] for k in ("first", "new_sources", "dropped_sources", "changed")} == {
        "first": False, "new_sources": ["https://c.org"], "dropped_sources": ["https://b.org"], "changed": True}
    assert counters()["changes"]["new_sources"] == 1


# --- the lock (REC-105) ---------------------------------------------------------------------------

def test_a_second_instance_exits(env):
    first = SchedulerLock(now=env.clock)
    first.acquire(enabled=True)
    second = SchedulerLock(now=env.clock)
    second.host = "another-host"
    with pytest.raises(sched_mod.LockHeld):
        second.acquire(enabled=True)
    assert files.read_json(files.scheduler_lock_path())["instance"] == first.instance


def test_a_stale_lock_is_taken_over(env):
    files.write_json_atomic(files.scheduler_lock_path(), {
        "pid": 1, "heartbeat": (T0 - timedelta(seconds=181)).isoformat(), "enabled": True,
        "owner": "scheduler", "host": "elsewhere", "instance": "old"})
    lock = SchedulerLock(now=env.clock)
    lock.acquire(enabled=True)
    data = files.read_json(files.scheduler_lock_path())
    assert data["instance"] == lock.instance and data["pid"] == os.getpid()
    assert files.parse_time(data["heartbeat"]) == T0


def test_our_own_previous_life_is_taken_over(env):
    lock = SchedulerLock(now=env.clock)
    files.write_json_atomic(files.scheduler_lock_path(), {
        "pid": os.getpid(), "heartbeat": T0.isoformat(), "enabled": True, "owner": "scheduler",
        "host": lock.host, "instance": "previous"})
    lock.acquire(enabled=True)
    assert files.read_json(files.scheduler_lock_path())["instance"] == lock.instance


def test_a_backend_held_lock_is_waited_for_then_given_up(env):
    files.write_json_atomic(files.scheduler_lock_path(), {
        "pid": 1, "heartbeat": T0.isoformat(), "enabled": False, "owner": "backend"})
    waits = []
    lock = SchedulerLock(now=env.clock, wait_for_backend=10, sleep=waits.append)
    with pytest.raises(sched_mod.LockHeld):
        lock.acquire(enabled=True)
    assert waits == [5, 5]


def test_disabled_scheduler_beats_with_enabled_false_and_runs_nothing(env):
    topic_id = add_topic()
    lock = SchedulerLock(now=env.clock)
    lock.acquire(enabled=False)
    s = env.make(enabled=False)
    calls = []

    def beat(enabled):
        calls.append(enabled)
        files.write_json_atomic(files.scheduler_lock_path(), {**lock._mine(enabled)})
        s.stop()
        return True

    lock.beat = beat
    s.serve(lock)
    assert calls == [False]
    assert files.read_json(files.scheduler_lock_path())["enabled"] is False
    assert env.backend.posts == [] and topics.load_state(topic_id) == {}


def test_beat_fails_when_someone_else_took_the_lock(env):
    lock = SchedulerLock(now=env.clock)
    lock.acquire(enabled=True)
    assert lock.beat(enabled=True)
    files.write_json_atomic(files.scheduler_lock_path(), {"heartbeat": T0.isoformat(), "instance": "x"})
    assert lock.beat(enabled=True) is False


def test_release_keeps_the_heartbeat_but_lets_the_next_one_start(env):
    first = SchedulerLock(now=env.clock)
    first.acquire(enabled=True)
    first.release()
    data = files.read_json(files.scheduler_lock_path())
    assert data["released"] is True and files.parse_time(data["heartbeat"]) == T0
    second = SchedulerLock(now=env.clock)
    second.host = "another-host"
    second.acquire(enabled=True)
    assert files.read_json(files.scheduler_lock_path())["instance"] == second.instance


@pytest.mark.parametrize("lock, code", [
    ({"heartbeat": (T0 - timedelta(seconds=30)).isoformat(), "enabled": True}, 0),
    ({"heartbeat": (T0 - timedelta(seconds=30)).isoformat(), "enabled": False}, 0),
    ({"heartbeat": (T0 - timedelta(seconds=200)).isoformat(), "enabled": True}, 1),
    ({"enabled": True}, 1),
    (None, 1),
], ids=["fresh", "fresh-disabled", "stale", "no-heartbeat", "missing"])
def test_health_flag(env, capsys, lock, code):
    if lock is not None:
        files.write_json_atomic(files.scheduler_lock_path(), lock)
    assert sched_mod.health(now=T0) == code
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1 and out[0].startswith("ok" if code == 0 else "unhealthy")


def test_health_flag_through_main_takes_no_lock(env, monkeypatch):
    files.write_json_atomic(files.scheduler_lock_path(), {"heartbeat": files.utcnow().isoformat()})
    before = files.read_json(files.scheduler_lock_path())
    assert sched_mod.main(["--health"]) == 0
    assert files.read_json(files.scheduler_lock_path()) == before
    files.scheduler_lock_path().unlink()
    assert sched_mod.main(["--health"]) == 1
    assert not files.scheduler_lock_path().exists()


def test_startup_checks(env, monkeypatch):
    monkeypatch.setattr(settings, "SCHEDULE", ScheduleConfig(enabled=True))
    sched_mod.checks(env.client, sleep=lambda s: None)          # token set, /health answers
    monkeypatch.setattr(settings, "SCHEDULER_TOKEN", None)
    with pytest.raises(SystemExit) as no_token:
        sched_mod.checks(env.client, sleep=lambda s: None)
    assert no_token.value.code != 0
    monkeypatch.setattr(settings, "SCHEDULER_TOKEN", TOKEN)
    tries = []

    def down(request):
        tries.append(1)
        raise httpx.ConnectError("refused")

    dead = httpx.Client(base_url="http://backend", transport=httpx.MockTransport(down))
    with pytest.raises(SystemExit) as unreachable:
        sched_mod.checks(dead, sleep=lambda s: None)
    assert unreachable.value.code != 0 and len(tries) == 3
    monkeypatch.setattr(settings, "SCHEDULE", ScheduleConfig(enabled=False))
    sched_mod.checks(dead, sleep=lambda s: None)                # disabled: nothing to check


# --- crash safety (REC-106) -----------------------------------------------------------------------

def _open_record(topic_id, status):
    run_id = str(uuid.uuid4())
    record = topic_runs.new_record(topic_id, run_id, T0 - timedelta(minutes=2), "slot")
    record["status"] = status
    topic_runs.save(record)
    return record


def test_crash_after_the_record_marks_the_run_not_started(env):
    topic_id = add_topic()
    env.make().tick()
    record = _open_record(topic_id, "starting")
    s = env.make()
    assert s.reconcile() == 1
    after = topic_runs.load(topic_id, record["run_id"])
    assert after["status"] == "failed" and after["error_code"] == "not_started"
    assert topics.load_state(topic_id)["failures_in_a_row"] == 1
    assert env.backend.posts == []


@pytest.mark.parametrize("status", ["starting", "running"])
def test_crash_after_the_post_polls_the_known_task_to_the_end(env, status):
    topic_id = add_topic()
    env.make().tick()
    record = _open_record(topic_id, status)
    env.backend.tasks[record["task_id"]] = {"created": T0 - timedelta(minutes=2), "topic_id": topic_id}
    s = env.make()
    assert s.reconcile() == 1
    assert len(s.jobs) == 1
    drive(s, env.clock, 3)
    after = topic_runs.load(topic_id, record["run_id"])
    assert after["status"] == "completed" and after["usage"] == env.backend.usage
    assert files.read_json(files.usage_path("2026-10"))["credits"] == 4
    assert env.backend.posts == []          # never started twice


def test_crash_between_state_and_record_is_a_failed_run_not_a_second_one(env):
    topic_id = add_topic()
    env.make().tick()
    lost = str(uuid.uuid4())
    set_state(topic_id, current_run=lost, next_due=(T0 + timedelta(days=1)).isoformat())
    s = env.make()
    s.reconcile()
    record = topic_runs.load(topic_id, lost)
    assert record["status"] == "failed" and record["error_code"] == "not_started"
    s.tick()
    assert env.backend.posts == []


def test_a_post_that_times_out_is_polled_and_a_post_with_task_exists_is_followed(env):
    topic_id = add_topic()
    s = env.make()
    s.tick()
    env.backend.post_reply = httpx.ReadTimeout("slow")
    set_state(topic_id, next_due=T0.isoformat())
    s.tick()
    drive(s, env.clock, 1)
    [record] = runs(topic_id)
    assert record["error_code"] == "not_started"     # the task never appeared


# --- failures and holds (REC-112) -----------------------------------------------------------------

@pytest.mark.parametrize("reply, code", [
    ((403, {"code": "declined"}), "intake_refused"),
    ((409, {"error": {"code": "consent_required"}}), "intake_refused"),
    ((429, {"error": {"code": "rate_limited"}}), "rate_limited"),
    ((503, {}), "backend_unreachable"),
    (httpx.ConnectError("refused"), "backend_unreachable"),
    ((422, {"code": "validation_error"}), "rejected"),
])
def test_start_failures(env, reply, code):
    topic_id = add_topic()
    s = env.make()
    s.tick()
    env.backend.post_reply = reply
    set_state(topic_id, next_due=T0.isoformat())
    s.tick()
    [record] = runs(topic_id)
    assert record["status"] == "failed" and record["error_code"] == code
    assert topics.load_state(topic_id)["failures_in_a_row"] == 1
    assert counters()["runs"]["failed"] == 1


@pytest.mark.parametrize("code", ["timeout", "budget", "cancelled", "escalated"])
def test_task_failures_keep_the_backend_error_code_and_count_usage(env, code):
    topic_id = add_topic()
    s = env.make()
    s.tick()
    set_state(topic_id, next_due=T0.isoformat())
    s.tick()
    [record] = runs(topic_id)
    env.backend.final[record["task_id"]] = ("failed", code)
    drive(s, env.clock, 1)
    record = runs(topic_id)[0]
    assert record["status"] == "failed" and record["error_code"] == code
    assert files.read_json(files.usage_path("2026-10"))["credits"] == 4


def test_poll_timeout_after_ten_minutes_then_one_late_poll_counts_usage(env):
    topic_id = add_topic()
    env.backend.duration = timedelta(minutes=11)
    s = env.make()
    s.tick()
    set_state(topic_id, next_due=T0.isoformat())
    s.tick()
    drive(s, env.clock, 10.5)
    record = runs(topic_id)[0]
    assert record["status"] == "failed" and record["error_code"] == "poll_timeout"
    assert files.read_json(files.usage_path("2026-10"))["credits"] == 0
    assert s.reserved() == 1                         # its worst case stays reserved
    drive(s, env.clock, 2)
    record = runs(topic_id)[0]
    assert record["error_code"] == "poll_timeout" and record["usage"] == env.backend.usage
    assert files.read_json(files.usage_path("2026-10"))["credits"] == 4
    assert s.reserved() == 0


def test_three_failures_hold_the_topic_and_resume_clears_it(env):
    topic_id = add_topic()
    s = env.make(pause_after_failures=3)
    s.tick()
    env.backend.post_reply = (403, {"code": "declined"})
    for i in range(3):
        edit(topic_id, run_requests=i + 1)
        s.tick()
    state = topics.load_state(topic_id)
    assert state["failures_in_a_row"] == 3 and state["hold"] == "failures"
    assert counters()["holds"]["failures"] == 1
    edit(topic_id, run_requests=4)
    s.tick()
    assert len(env.backend.posts) == 3                # held: no fourth start
    assert topics.effective_state(topics.load_definition(topic_id), topics.load_state(topic_id)) == "held_failures"

    env.backend.post_reply = None
    edit(topic_id, resumes=1)
    s.tick()
    state = topics.load_state(topic_id)
    assert state["hold"] is None and state["resumes_seen"] == 1
    assert len(env.backend.posts) == 4                # the pending request runs after the resume
    drive(s, env.clock, 4)
    assert topics.load_state(topic_id)["failures_in_a_row"] == 0


# --- daily cap and allowance (REC-113, RNFR-04) ---------------------------------------------------

def test_the_daily_cap_skips_and_advances(env):
    a, b = add_topic(), add_topic()
    s = env.make(max_runs_per_day=1)
    s.tick()
    set_state(a, next_due=(T0 - timedelta(minutes=2)).isoformat())
    set_state(b, next_due=(T0 - timedelta(minutes=1)).isoformat())
    s.tick()
    assert [body["topic_id"] for _, body in env.backend.posts] == [a]
    state = topics.load_state(b)
    assert state["skipped"] == {"reason": "daily_cap", "at": T0.isoformat()}
    assert files.parse_time(state["next_due"]) == T0 + timedelta(days=1)
    assert counters()["runs"]["skipped"] == 1
    # The next UTC day the cap resets: one more run starts, the other topic is skipped again.
    env.clock.t = T0 + timedelta(days=1)
    drive(s, env.clock, 1)
    assert len(env.backend.posts) == 2
    assert counters()["runs"]["skipped"] == 2


def test_the_allowance_is_never_passed_with_two_concurrent_runs(env):
    # Worst case: 3 passes x 3 = 9 credits a run. 20 credits cover two runs at once, not three.
    ids = [add_topic() for _ in range(4)]
    env.backend.usage = dict(env.backend.usage, tavily_credits=9)
    s = env.make(max_concurrent_runs=3, monthly_tavily_credits=20)
    s.tick()
    for topic_id in ids:
        set_state(topic_id, next_due=T0.isoformat())
    s.tick()
    assert len(env.backend.posts) == 2
    for topic_id in ids:
        assert topics.load_state(topic_id)["hold"] == "budget"
    assert counters()["holds"]["budget"] == 4
    for _ in range(20):
        drive(s, env.clock, 1)
        used = files.read_json(files.usage_path("2026-10"))["credits"]
        assert used + 9 * s.reserved() <= 20
    assert used == 18 and len(env.backend.posts) == 2
    assert all(topics.load_state(t)["hold"] == "budget" for t in ids)

    # Next month the hold clears and the topics run again, from their next slot.
    env.clock.t = datetime(2026, 11, 1, 6, 0, tzinfo=timezone.utc)
    s.tick()
    assert all(topics.load_state(t)["hold"] is None for t in ids)
    assert len(env.backend.posts) == 2
    env.clock.t = datetime(2026, 11, 2, 6, 0, tzinfo=timezone.utc)
    s.tick()
    assert len(env.backend.posts) == 4
    assert files.read_json(files.usage_path("2026-11"))["runs"] == 2


def test_a_raised_allowance_clears_the_budget_hold(env):
    topic_id = add_topic()
    s = env.make(monthly_tavily_credits=5)
    s.tick()
    set_state(topic_id, next_due=T0.isoformat())
    s.tick()
    assert topics.load_state(topic_id)["hold"] == "budget" and env.backend.posts == []
    s.cfg = s.cfg.model_copy(update={"monthly_tavily_credits": 400})
    s.tick()
    assert topics.load_state(topic_id)["hold"] is None


def test_the_dollar_allowance_counts_too(env):
    topic_id = add_topic()
    # (60000 + 8000) x 1.5 / 1e6 = 0.102 dollars a run.
    s = env.make(monthly_together_usd=0.1)
    s.tick()
    set_state(topic_id, next_due=T0.isoformat())
    s.tick()
    assert env.backend.posts == [] and topics.load_state(topic_id)["hold"] == "budget"
    assert s.worst_case() == {"credits": 9.0, "dollars": pytest.approx(0.102)}


# --- deletion and retention (REC-114, REC-115) ----------------------------------------------------

def test_a_tombstone_removes_the_topic_folder_and_its_results(env):
    topic_id = add_topic()
    other = add_topic()
    s = env.make()
    s.tick()
    for t in (topic_id, other):
        set_state(t, next_due=T0.isoformat())
    s.tick()
    drive(s, env.clock, 4)
    task_ids = {t: runs(t)[0]["task_id"] for t in (topic_id, other)}
    assert result_store.load_result(task_ids[topic_id]) is not None
    edit(topic_id, status="deleted")
    s.tick()
    assert not files.topic_dir(topic_id).exists()
    assert result_store.load_result(task_ids[topic_id]) is None
    assert files.definition_path(topic_id).exists()          # the backend removes the definition
    assert result_store.load_result(task_ids[other]) is not None and runs(other)


def test_a_run_finishing_after_its_topic_was_deleted_leaves_nothing_but_counts_usage(env):
    topic_id = add_topic()
    s = env.make()
    s.tick()
    set_state(topic_id, next_due=T0.isoformat())
    s.tick()
    edit(topic_id, status="deleted")
    s.tick()
    assert not files.topic_dir(topic_id).exists()
    task_id = next(iter(env.backend.tasks))
    drive(s, env.clock, 4)
    assert not files.topic_dir(topic_id).exists()
    assert result_store.load_result(task_id) is None
    assert files.read_json(files.usage_path("2026-10"))["credits"] == 4


def test_pruning_keeps_the_last_runs_and_removes_their_results(env):
    topic_id = add_topic()
    s = env.make(history_runs_kept=3)
    s.tick()
    for i in range(5):
        edit(topic_id, run_requests=i + 1)
        drive(s, env.clock, 4)
    kept = runs(topic_id)
    assert len(kept) == 3
    kept_tasks = {r["task_id"] for r in kept}
    for task_id in env.backend.tasks:
        assert (result_store.load_result(task_id) is not None) == (task_id in kept_tasks)
    assert counters()["runs"]["completed"] == 5                # counters never shrink


def test_paused_since_and_the_90_day_deletion(env):
    topic_id = add_topic()
    s = env.make(paused_topic_days=90)
    s.tick()
    edit(topic_id, status="paused")
    s.tick()
    assert files.parse_time(topics.load_state(topic_id)["paused_since"]) == T0
    env.clock.t = T0 + timedelta(days=89)
    s.tick()
    assert env.backend.deletes == []
    env.clock.t = T0 + timedelta(days=90)
    s.tick()
    [(deleted, headers)] = env.backend.deletes
    assert deleted == topic_id
    assert headers["x-webagent-scheduler"] == TOKEN and headers["x-api-key"] == "api-key-test"
    s.tick()
    assert len(env.backend.deletes) == 1                         # not asked again at once


def test_unpausing_clears_paused_since_and_waits_for_the_next_slot(env):
    topic_id = add_topic()
    s = env.make()
    s.tick()
    edit(topic_id, status="paused")
    s.tick()
    env.clock.t = T0 + timedelta(days=3, hours=1)
    edit(topic_id, status="active")
    s.tick()
    state = topics.load_state(topic_id)
    assert state["paused_since"] is None
    assert files.parse_time(state["next_due"]) == T0 + timedelta(days=4)
    assert env.backend.posts == []


# --- timeliness (RNFR-01) and the threaded path --------------------------------------------------

def test_ten_topics_due_at_once_all_start_within_15_minutes(env):
    env.clock.t = T0 - timedelta(minutes=1)
    ids = [add_topic(at="06:00") for _ in range(10)]
    s = env.make(max_concurrent_runs=2)
    s.tick()
    env.clock.t = T0
    starts = {}
    end = T0 + timedelta(minutes=30)
    last_tick = None
    while env.clock() < end:
        finished = s.pump(env.clock())
        if finished or last_tick is None or env.clock() - last_tick >= timedelta(seconds=60):
            s.tick(env.clock())
            last_tick = env.clock()
        for _, body in env.backend.posts:
            starts.setdefault(body["topic_id"], env.clock())
        env.clock.advance(seconds=15)
    assert set(starts) == set(ids)
    assert max(starts.values()) - T0 <= timedelta(minutes=15)


def test_threaded_runs_finish_and_wake_the_loop(env):
    topic_id = add_topic()
    env.backend.duration = timedelta(0)
    s = Scheduler(client=env.client, now=env.clock, config=ScheduleConfig(enabled=True), threaded=True,
                  sleep=lambda seconds: None, max_follow_up_questions=3, search_cap=5)
    s.tick()
    set_state(topic_id, next_due=T0.isoformat())
    s.tick()
    assert s._wake.wait(timeout=5)
    for _ in range(100):
        if not s.jobs:
            break
        s._wake.wait(timeout=0.05)
    assert runs(topic_id)[0]["status"] == "completed"


def test_logs_carry_ids_not_the_query(env, caplog):
    topic_id = add_topic()
    s = env.make()
    s.tick()
    set_state(topic_id, next_due=T0.isoformat())
    with caplog.at_level("DEBUG", logger="app.scheduler"):
        s.tick()
        drive(s, env.clock, 4)
    text = caplog.text
    assert topic_id in text and "what is new in x" not in text and "report text" not in text


def test_a_fresh_lock_is_waited_for_until_it_goes_stale(env):
    """After a crash or a redeploy the new scheduler waits for the old heartbeat to expire (no crash loop)."""
    files.write_json_atomic(files.scheduler_lock_path(), {
        "pid": 1, "heartbeat": env.clock().isoformat(), "enabled": True, "owner": "scheduler",
        "host": "another-host", "instance": "old"})
    start = env.clock()
    slept = []

    def sleep(seconds):
        slept.append(seconds)
        env.clock.advance(seconds=seconds)

    lock = SchedulerLock(now=env.clock, sleep=sleep, wait_for_holder=240)
    lock.acquire(True)
    assert lock.held and sum(slept) >= 180 and (env.clock() - start).total_seconds() < 240
    assert files.read_json(files.scheduler_lock_path())["instance"] == lock.instance


def test_a_live_holder_is_not_taken_over_while_waiting(env):
    def sleep(seconds):
        env.clock.advance(seconds=seconds)
        # the live holder keeps beating
        files.write_json_atomic(files.scheduler_lock_path(), {
            "pid": 1, "heartbeat": env.clock().isoformat(), "enabled": True, "owner": "scheduler",
            "host": "another-host", "instance": "live"})

    sleep(0)
    lock = SchedulerLock(now=env.clock, sleep=sleep, wait_for_holder=240)
    with pytest.raises(sched_mod.LockHeld):
        lock.acquire(True)
    assert files.read_json(files.scheduler_lock_path())["instance"] == "live"


# --- phase B: change summary, notify decision, notices (REC-130, REC-131, REC-138) ----------------

def subscribed(when="changes", confirmed=True):
    sub = {"email": "reader@example.org", "added_at": (T0 - timedelta(days=1)).isoformat(),
           "confirmed_at": (T0 - timedelta(days=1)).isoformat() if confirmed else None}
    return {"when": when, "subscribers": [sub]}


def run_once(s, env, topic_id, minutes=5):
    defn = topics.load_definition(topic_id)
    edit(topic_id, run_requests=defn.run_requests + 1)
    drive(s, env.clock, minutes)
    return runs(topic_id)[0]


def run_kinds(env):
    return [(t, b["kind"], b.get("run_id")) for t, b, _ in env.backend.notifies]


def test_the_first_run_is_not_summarised_and_the_second_is_stored(env):
    topic_id = add_topic()
    s = env.make()
    s.tick()
    first = run_once(s, env, topic_id)
    assert env.backend.summaries == []
    assert first["change"]["summary"] is None and "summary_error" not in first["change"]

    env.backend.sources = list(env.backend.sources)   # same sources: no new ones
    second = run_once(s, env, topic_id)
    [(tid, body, headers)] = env.backend.summaries
    assert tid == topic_id and body == {"run_id": second["run_id"], "previous_run_id": first["run_id"]}
    assert headers["x-webagent-scheduler"] == TOKEN and headers["x-api-key"] == "api-key-test"
    assert second["change"]["summary"] == "- Something new [1]"
    assert second["change"]["new_sources"] == [] and second["change"]["changed"] is True
    assert counters()["changes"]["summary_changed"] == 1
    # The summary's tokens count toward the month (two runs plus one summary).
    usage = files.read_json(files.usage_path("2026-10"))
    run_dollars = 1000 * 0.35 / 1e6 + 500 * 1.5 / 1e6
    assert usage["dollars"] == pytest.approx(2 * run_dollars + 2000 * 0.35 / 1e6 + 100 * 1.5 / 1e6)
    assert usage["credits"] == 8


def test_the_worst_case_reserves_the_summary(env):
    s = env.make(token_budget_default=60000)
    assert s.worst_case()["dollars"] == pytest.approx((60000 + 8000) * 1.5 / 1e6)


def test_no_material_change_leaves_changed_false_without_new_sources(env):
    topic_id = add_topic()
    s = env.make()
    s.tick()
    run_once(s, env, topic_id)
    env.backend.summary_reply = (200, {"summary": "No material change", "changed": False,
                                       "tokens": {"input": 10, "output": 3}})
    second = run_once(s, env, topic_id)
    assert second["change"]["summary"] == "No material change"
    assert second["change"]["changed"] is False
    assert "summary_changed" not in counters().get("changes", {})


def test_new_sources_keep_changed_true_whatever_the_summary(env):
    topic_id = add_topic()
    s = env.make()
    s.tick()
    run_once(s, env, topic_id)
    env.backend.sources = [{"url": "https://c.org"}]
    env.backend.summary_reply = (200, {"summary": "No material change", "changed": False,
                                       "tokens": {"input": 10, "output": 3}})
    second = run_once(s, env, topic_id)
    assert second["change"]["changed"] is True and second["change"]["new_sources"] == ["https://c.org"]


@pytest.mark.parametrize("reply, code", [
    ((200, {"summary": None, "changed": False, "tokens": {"input": 50, "output": 0}, "error": "model_failed"}),
     "model_failed"),
    ((500, {"detail": "boom"}), "http_500"),
    (httpx.ConnectError("down"), "backend_unreachable"),
])
def test_a_failed_summary_keeps_the_diff_and_records_the_error(env, reply, code):
    topic_id = add_topic()
    s = env.make()
    s.tick()
    run_once(s, env, topic_id)
    env.backend.summary_reply = reply
    second = run_once(s, env, topic_id)
    assert second["status"] == "completed"
    assert second["change"]["summary"] is None and second["change"]["summary_error"] == code
    assert second["change"]["changed"] is False and second["change"]["first"] is False
    state = topics.load_state(topic_id)
    assert state["failures_in_a_row"] == 0 and state["hold"] is None


def test_flagged_refs_come_from_dropped_and_shadow_flagged(env):
    topic_id = add_topic()
    env.backend.security = {
        "dropped": [{"source": "web_result", "ref": "https://a.org/x", "policy_id": "p"}],
        "shadow_flagged": [{"source": "web_result", "ref": "https://b.org", "policy_id": "p"},
                           {"source": "web_result", "ref": "https://a.org/x", "policy_id": "p"},
                           {"source": "retrieved_doc", "ref": None, "policy_id": "p"}],
    }
    s = env.make()
    s.tick()
    record = run_once(s, env, topic_id)
    assert record["flagged_refs"] == ["https://a.org/x", "https://b.org"]


@pytest.mark.parametrize("status, change, when, expected", [
    ("completed", {"first": True, "changed": True}, "changes", True),
    ("completed", {"first": True, "changed": True}, "always", True),
    ("completed", {"first": False, "changed": True}, "changes", True),
    ("completed", {"first": False, "changed": False}, "changes", False),
    ("completed", {"first": False, "changed": False}, "always", True),
    ("failed", None, "always", False),
    ("failed", {"first": True, "changed": True}, "always", False),
])
def test_the_notify_decision_table(status, change, when, expected):
    assert sched_mod.should_notify_run(status, change, when) is expected


def test_notify_runs_for_changes_only_with_a_confirmed_subscriber(env):
    topic_id = add_topic(notify=subscribed("changes"))
    s = env.make()
    s.tick()
    first = run_once(s, env, topic_id)
    assert run_kinds(env) == [(topic_id, "run", first["run_id"])]          # first run notifies
    _, _, headers = env.backend.notifies[0]
    assert headers["x-webagent-scheduler"] == TOKEN and headers["x-api-key"] == "api-key-test"

    env.backend.summary_reply = (200, {"summary": "No material change", "changed": False,
                                       "tokens": {"input": 1, "output": 1}})
    run_once(s, env, topic_id)
    assert len(env.backend.notifies) == 1                                   # nothing changed

    env.backend.summary_reply = (200, {"summary": "- New [1]", "changed": True,
                                       "tokens": {"input": 1, "output": 1}})
    third = run_once(s, env, topic_id)
    assert run_kinds(env)[-1] == (topic_id, "run", third["run_id"])        # a material summary


def test_notify_always_after_every_completed_run(env):
    topic_id = add_topic(notify=subscribed("always"))
    env.backend.summary_reply = (200, {"summary": "No material change", "changed": False,
                                       "tokens": {"input": 1, "output": 1}})
    s = env.make()
    s.tick()
    run_once(s, env, topic_id)
    run_once(s, env, topic_id)
    assert [k for _, k, _ in run_kinds(env)] == ["run", "run"]


@pytest.mark.parametrize("notify", [None, "unconfirmed"])
def test_no_confirmed_subscriber_means_no_notify_call(env, notify):
    extra = {} if notify is None else {"notify": subscribed("always", confirmed=False)}
    topic_id = add_topic(**extra)
    s = env.make(pause_after_failures=1)
    s.tick()
    run_once(s, env, topic_id)
    env.backend.post_reply = (403, {"code": "declined"})
    edit(topic_id, run_requests=topics.load_definition(topic_id).run_requests + 1)
    s.tick()
    assert topics.load_state(topic_id)["hold"] == "failures"
    assert env.backend.notifies == []


def test_failed_runs_never_notify_and_the_failures_hold_sends_one_notice(env):
    topic_id = add_topic(notify=subscribed("always"))
    s = env.make(pause_after_failures=3)
    s.tick()
    env.backend.post_reply = (403, {"code": "declined"})
    for i in range(3):
        edit(topic_id, run_requests=i + 1)
        s.tick()
    assert topics.load_state(topic_id)["hold"] == "failures"
    assert [(t, b) for t, b, _ in env.backend.notifies] == [(topic_id, {"kind": "hold"})]
    edit(topic_id, run_requests=4)
    s.tick()
    assert len(env.backend.notifies) == 1


def test_the_deletion_notice_goes_out_once_seven_days_before(env):
    topic_id = add_topic(notify=subscribed())
    s = env.make(paused_topic_days=90)
    s.tick()
    edit(topic_id, status="paused")
    s.tick()
    env.clock.t = T0 + timedelta(days=82, hours=23)
    s.tick()
    assert env.backend.notifies == []
    env.clock.t = T0 + timedelta(days=83)
    s.tick()
    assert [(t, b) for t, b, _ in env.backend.notifies] == [(topic_id, {"kind": "deletion"})]
    assert files.parse_time(topics.load_state(topic_id)["deletion_notice_at"]) == env.clock.t
    env.clock.t = T0 + timedelta(days=85)
    s.tick()
    env.clock.t = T0 + timedelta(days=90)
    s.tick()
    assert len(env.backend.notifies) == 1 and len(env.backend.deletes) == 1


def test_resuming_resets_the_deletion_notice(env):
    topic_id = add_topic(notify=subscribed())
    s = env.make(paused_topic_days=90)
    s.tick()
    edit(topic_id, status="paused")
    s.tick()
    env.clock.t = T0 + timedelta(days=84)
    s.tick()
    edit(topic_id, status="active")
    s.tick()
    assert "deletion_notice_at" not in topics.load_state(topic_id)
    edit(topic_id, status="paused")
    s.tick()
    env.clock.t = T0 + timedelta(days=84 + 83)
    s.tick()
    assert [b["kind"] for _, b, _ in env.backend.notifies] == ["deletion", "deletion"]


def test_a_long_paused_topic_without_subscribers_gets_no_notice(env):
    topic_id = add_topic()
    s = env.make(paused_topic_days=90)
    s.tick()
    edit(topic_id, status="paused")
    s.tick()
    env.clock.t = T0 + timedelta(days=85)
    s.tick()
    assert env.backend.notifies == [] and "deletion_notice_at" not in topics.load_state(topic_id)


@pytest.mark.parametrize("reply", [httpx.ConnectError("down"), (500, {"detail": "x"}), (200, {"sent": 0, "skipped": 1, "reason": "email_off"})])
def test_a_notify_failure_is_harmless(env, reply, caplog):
    topic_id = add_topic(notify=subscribed("always"))
    env.backend.notify_reply = reply
    s = env.make()
    s.tick()
    with caplog.at_level("INFO", logger="app.scheduler"):
        record = run_once(s, env, topic_id)
    assert record["status"] == "completed"
    state = topics.load_state(topic_id)
    assert state["failures_in_a_row"] == 0 and state["hold"] is None and state["current_run"] is None
    assert len(env.backend.notifies) == 1
    assert "reader@example.org" not in caplog.text and "report text" not in caplog.text
    second = run_once(s, env, topic_id)                     # the schedule carries on
    assert second["status"] == "completed" and len(env.backend.notifies) == 2


def test_summary_text_is_not_logged(env, caplog):
    topic_id = add_topic()
    s = env.make()
    s.tick()
    run_once(s, env, topic_id)
    with caplog.at_level("DEBUG", logger="app.scheduler"):
        run_once(s, env, topic_id)
    assert "Something new" not in caplog.text
