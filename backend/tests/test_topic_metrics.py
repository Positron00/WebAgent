"""The topic collector (recurring-research spec REC-117), from fixture files."""
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from prometheus_client import CollectorRegistry, generate_latest

from app.core import topic_metrics
from app.core.config import settings
from app.core.config_def import ScheduleConfig
from app.core.topic_metrics import TopicCollector, register_topic_collector
from app.services import topic_files as files
from app.services import topic_runs, topics

NOW = datetime.now(timezone.utc)


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(settings, "SCHEDULE", ScheduleConfig(monthly_tavily_credits=400, monthly_together_usd=5.0))
    return tmp_path


def _topic(status="active", hold=None):
    topic_id = str(uuid.uuid4())
    files.write_json_atomic(files.definition_path(topic_id), {
        "topic_id": topic_id, "name": "n", "query": "q", "status": status, "created_at": NOW.isoformat(),
        "schedule": {"every": "day", "time": "06:00", "tz": "UTC"}})
    files.write_json_atomic(files.state_path(topic_id), {"hold": hold})
    return topic_id


def _fixtures():
    _topic()
    _topic()
    _topic(status="paused")
    _topic(hold="failures")
    _topic(hold="budget")
    _topic(status="deleted")
    files.write_json_atomic(files.counters_path(), {
        "runs": {"completed": 7, "failed": 2, "skipped": 1},
        "changes": {"new_sources": 3},
        "holds": {"failures": 1, "budget": 4},
        "missed_slots": 5,
        "run_seconds": {"buckets": {"60": 2, "180": 4, "+Inf": 1}, "sum": 1500.0, "count": 7},
    })
    files.write_json_atomic(files.usage_path(NOW.strftime("%Y-%m")), {"credits": 36, "dollars": 0.42})
    files.write_json_atomic(files.scheduler_lock_path(), {
        "pid": 1, "heartbeat": NOW.isoformat(), "enabled": True, "owner": "scheduler"})


def _registry():
    registry = CollectorRegistry()
    collector = register_topic_collector(registry)
    return registry, collector


def test_the_collector_exports_every_metric():
    _fixtures()
    registry, _ = _registry()
    value = registry.get_sample_value
    assert value("webagent_topic_runs_total", {"outcome": "completed"}) == 7
    assert value("webagent_topic_runs_total", {"outcome": "failed"}) == 2
    assert value("webagent_topic_runs_total", {"outcome": "skipped"}) == 1
    assert value("webagent_topic_changes_total", {"kind": "new_sources"}) == 3
    assert value("webagent_topic_changes_total", {"kind": "summary_changed"}) == 0
    assert value("webagent_topic_held_total", {"reason": "budget"}) == 4
    assert value("webagent_scheduler_missed_slots_total") == 5
    assert value("webagent_topics", {"state": "active"}) == 2
    assert value("webagent_topics", {"state": "paused"}) == 1
    assert value("webagent_topics", {"state": "held_failures"}) == 1
    assert value("webagent_topics", {"state": "held_budget"}) == 1
    assert value("webagent_topic_budget_used", {"kind": "credits"}) == 36
    assert value("webagent_topic_budget_used", {"kind": "dollars"}) == pytest.approx(0.42)
    assert value("webagent_topic_budget_limit", {"kind": "credits"}) == 400
    assert value("webagent_topic_budget_limit", {"kind": "dollars"}) == 5.0
    assert value("webagent_scheduler_heartbeat_timestamp") == pytest.approx(NOW.timestamp())
    assert value("webagent_scheduler_enabled") == 1
    assert value("webagent_topic_run_seconds_bucket", {"le": "30.0"}) == 0
    assert value("webagent_topic_run_seconds_bucket", {"le": "60.0"}) == 2
    assert value("webagent_topic_run_seconds_bucket", {"le": "300.0"}) == 6
    assert value("webagent_topic_run_seconds_bucket", {"le": "+Inf"}) == 7
    assert value("webagent_topic_run_seconds_count") == 7
    assert value("webagent_topic_run_seconds_sum") == 1500.0
    assert b"webagent_topic_runs_total" in generate_latest(registry)


def test_no_files_still_exports_the_families_without_failing():
    registry, _ = _registry()
    text = generate_latest(registry).decode()
    for name in ("webagent_topic_runs", "webagent_topics", "webagent_topic_budget_limit",
                 "webagent_scheduler_heartbeat_timestamp", "webagent_topic_run_seconds"):
        assert f"# TYPE {name}" in text
    assert registry.get_sample_value("webagent_topic_runs_total", {"outcome": "completed"}) == 0
    assert registry.get_sample_value("webagent_scheduler_heartbeat_timestamp") is None


def test_reads_are_cached_for_15_seconds(monkeypatch):
    _fixtures()
    moment = [100.0]
    collector = TopicCollector(clock=lambda: moment[0])
    calls = []
    real = topic_metrics._snapshot
    monkeypatch.setattr(topic_metrics, "_snapshot", lambda now: calls.append(1) or real(now))
    for _ in range(5):
        list(collector.collect())
    moment[0] += 14
    list(collector.collect())
    assert len(calls) == 1
    moment[0] += 2
    list(collector.collect())
    assert len(calls) == 2


def test_a_backend_held_lock_is_not_a_scheduler_heartbeat():
    files.write_json_atomic(files.scheduler_lock_path(), {"heartbeat": NOW.isoformat(), "owner": "backend"})
    registry, _ = _registry()
    assert registry.get_sample_value("webagent_scheduler_heartbeat_timestamp") is None


def test_counters_never_decrease_after_pruning(monkeypatch, tmp_path):
    """Drive real runs through the scheduler, prune, and check every counter only grows."""
    import httpx

    from app.scheduler import Scheduler
    from app.services import result_store

    monkeypatch.setattr(result_store, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(settings, "SCHEDULER_TOKEN", "t")
    clock = [datetime(2026, 10, 19, 6, 0, tzinfo=timezone.utc)]

    def handler(request):
        if request.method == "POST":
            return httpx.Response(202, json={"status": "processing"})
        n = len(handler.seen)
        handler.seen.append(1)
        return httpx.Response(200, json={"status": "completed", "result": {
            "content": "r", "sources": [{"url": f"https://s{n}.org"}],
            "usage": {"input_tokens": 1, "output_tokens": 1, "searches": 1, "tavily_credits": 1}}})
    handler.seen = []

    topic_id = str(uuid.uuid4())
    topics.save_new_definition(topics.TopicDefinition(
        topic_id=topic_id, name="n", query="q", created_at=clock[0],
        schedule={"every": "day", "time": "06:00", "tz": "UTC"}))
    s = Scheduler(client=httpx.Client(base_url="http://b", transport=httpx.MockTransport(handler)),
                  now=lambda: clock[0], config=ScheduleConfig(enabled=True, history_runs_kept=2),
                  threaded=False, max_follow_up_questions=3, search_cap=5)
    s.tick()
    previous = None
    for i in range(6):
        topics.update_definition(topic_id, lambda d, i=i: setattr(d, "run_requests", i + 1))
        s.tick()
        clock[0] += timedelta(seconds=15)
        s.pump()
        assert len(topic_runs.list_records(topic_id)) <= 2
        collector = TopicCollector(cache_seconds=0)
        samples = {(m.name, sample.name, tuple(sorted(sample.labels.items()))): sample.value
                   for m in collector.collect() if m.type in ("counter", "histogram")
                   for sample in m.samples}
        if previous is not None:
            for key, value in previous.items():
                assert samples.get(key, 0) >= value, key
        previous = samples
    assert previous[("webagent_topic_runs", "webagent_topic_runs_total", (("outcome", "completed"),))] == 6
    assert previous[("webagent_topic_changes", "webagent_topic_changes_total", (("kind", "new_sources"),))] == 5
