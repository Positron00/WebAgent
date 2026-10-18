"""Topic files: atomic writes, UUID paths, flock across processes, deletion takeover (REC-101, REC-114)."""
import json
import multiprocessing
import os
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from app.core.config import settings
from app.services import result_store, topic_files as files, topics

SCHEDULE = topics.Schedule(every="day", time="07:00", tz="UTC")


def _definition(status="active"):
    return topics.TopicDefinition(
        topic_id=topics.new_topic_id(),
        name="Topic",
        query="A question",
        schedule=SCHEDULE,
        status=status,
        created_at=files.utcnow(),
    )


# --- atomic writes and paths ----------------------------------------------------------------------

def test_write_json_atomic_replaces_whole_file_and_leaves_no_temp(tmp_path):
    path = tmp_path / "topics" / "x.json"
    files.write_json_atomic(path, {"a": 1})
    files.write_json_atomic(path, {"b": 2})
    assert json.loads(path.read_text()) == {"b": 2}
    assert [p.name for p in path.parent.iterdir()] == ["x.json"]


def test_failed_write_keeps_old_file(tmp_path, monkeypatch):
    path = tmp_path / "x.json"
    files.write_json_atomic(path, {"a": 1})

    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(files.os, "replace", boom)
    with pytest.raises(OSError):
        files.write_json_atomic(path, {"b": 2})
    assert json.loads(path.read_text()) == {"a": 1}


def test_read_json_tolerates_missing_and_broken(tmp_path):
    assert files.read_json(tmp_path / "missing.json") is None
    broken = tmp_path / "broken.json"
    broken.write_text("{not json")
    assert files.read_json(broken) is None
    listed = tmp_path / "list.json"
    listed.write_text("[1, 2]")
    assert files.read_json(listed) is None


@pytest.mark.parametrize("bad", ["../etc/passwd", "..", "", "abc", "1234", None, str(uuid.uuid4()) + "/x"])
def test_paths_reject_non_uuids(bad):
    for fn in (files.definition_path, files.topic_dir, files.state_path, files.runs_dir):
        with pytest.raises(ValueError):
            fn(bad)
    with pytest.raises(ValueError):
        files.run_path(str(uuid.uuid4()), bad)
    with pytest.raises(ValueError):
        files.task_id_for_run(bad)


def test_paths_stay_in_topics_dir():
    tid = str(uuid.uuid4())
    root = files.topics_dir()
    assert files.definition_path(tid) == root / f"{tid}.json"
    assert files.state_path(tid).parent == root / tid
    assert str(files.definition_path(tid.upper())).endswith(f"{tid}.json")


# --- flock across two processes (REC-101) ---------------------------------------------------------

ROUNDS = 40


def _bump(data_dir, topic_id, field, start_event):
    """Child process: many read-modify-writes of one field through update_definition."""
    from app.core.config import settings as child_settings
    from app.services import topics as child_topics

    child_settings.DATA_DIR = data_dir
    start_event.wait(10)

    def change(defn):
        setattr(defn, field, getattr(defn, field) + 1)

    for _ in range(ROUNDS):
        child_topics.update_definition(topic_id, change)


def test_two_processes_editing_one_definition_keep_both_changes():
    defn = _definition()
    topics.save_new_definition(defn)
    ctx = multiprocessing.get_context("spawn")
    start = ctx.Event()
    procs = [
        ctx.Process(target=_bump, args=(settings.DATA_DIR, defn.topic_id, field, start))
        for field in ("run_requests", "resumes")
    ]
    for proc in procs:
        proc.start()
    start.set()
    for proc in procs:
        proc.join(60)
        assert proc.exitcode == 0
    saved = topics.load_definition(defn.topic_id)
    assert saved.run_requests == ROUNDS and saved.resumes == ROUNDS


def test_update_definition_missing_returns_none():
    assert topics.update_definition(str(uuid.uuid4()), lambda d: None) is None


# --- deletion takeover (REC-114) ------------------------------------------------------------------

def _tombstoned_with_runs(n_runs=2):
    defn = _definition(status="deleted")
    topics.save_new_definition(defn)
    task_ids = []
    for i in range(n_runs):
        run_id = str(uuid.uuid4())
        task_id = files.task_id_for_run(run_id)
        record = {"run_id": run_id, "task_id": task_id, "status": "completed",
                  "started_at": f"2026-10-0{i + 1}T07:00:00+00:00"}
        files.write_json_atomic(files.run_path(defn.topic_id, run_id), record)
        result_store.save_result(task_id, {"task_id": task_id, "status": "completed"})
        task_ids.append(task_id)
    files.write_json_atomic(files.state_path(defn.topic_id), {"next_due": "2026-10-19T07:00:00+00:00"})
    return defn.topic_id, task_ids


def _nothing_left(topic_id, task_ids):
    assert not files.topic_dir(topic_id).exists()
    assert not files.definition_path(topic_id).exists()
    lock = files.definition_path(topic_id)
    assert not lock.with_name(lock.name + ".lock").exists()
    for task_id in task_ids:
        assert result_store.load_result(task_id) is None
        assert not (result_store.RESULTS_DIR / f"{task_id}.json").exists()


def _lock(heartbeat, owner="scheduler"):
    data = {"pid": 12345, "heartbeat": heartbeat.isoformat(), "enabled": True, "owner": owner}
    files.write_json_atomic(files.scheduler_lock_path(), data)
    return data


def test_cleanup_without_scheduler_takes_over_and_removes_results():
    topic_id, task_ids = _tombstoned_with_runs()
    keep = _definition()
    topics.save_new_definition(keep)
    other_task = str(uuid.uuid4())
    result_store.save_result(other_task, {"task_id": other_task})

    seen_lock = {}
    real_remove = topics._remove_topic_files

    def spy(tid):
        seen_lock.update(files.read_json(files.scheduler_lock_path()) or {})
        real_remove(tid)

    topics._remove_topic_files = spy
    try:
        assert topics.cleanup_deleted() == 1
    finally:
        topics._remove_topic_files = real_remove

    assert seen_lock["owner"] == "backend" and seen_lock["pid"] == os.getpid()
    _nothing_left(topic_id, task_ids)
    assert not files.scheduler_lock_path().exists()   # released
    assert topics.load_definition(keep.topic_id) is not None
    assert result_store.load_result(other_task) is not None


def test_cleanup_with_stale_heartbeat_takes_over():
    topic_id, task_ids = _tombstoned_with_runs()
    _lock(files.utcnow() - timedelta(seconds=181))
    assert topics.cleanup_deleted() == 1
    _nothing_left(topic_id, task_ids)
    assert not files.scheduler_lock_path().exists()


def test_cleanup_with_fresh_heartbeat_leaves_files_to_scheduler():
    topic_id, task_ids = _tombstoned_with_runs()
    lock = _lock(files.utcnow() - timedelta(seconds=30))
    assert topics.cleanup_deleted() == 0
    assert files.topic_dir(topic_id).exists()
    assert files.definition_path(topic_id).exists()
    assert all(result_store.load_result(t) is not None for t in task_ids)
    assert files.read_json(files.scheduler_lock_path()) == lock

    # The scheduler removes its part; the next cleanup removes the definition.
    for task_id in task_ids:
        result_store.delete_result(task_id)
    import shutil
    shutil.rmtree(files.topic_dir(topic_id))
    assert topics.cleanup_deleted() == 1
    _nothing_left(topic_id, task_ids)
    assert files.read_json(files.scheduler_lock_path()) == lock


def test_cleanup_ignores_live_topics_and_keeps_a_scheduler_lock_it_did_not_write():
    live = _definition()
    topics.save_new_definition(live)
    files.write_json_atomic(files.state_path(live.topic_id), {})
    assert topics.cleanup_deleted() == 0
    assert files.topic_dir(live.topic_id).exists()
    assert not files.scheduler_lock_path().exists()

    # A scheduler that takes the lock during the takeover keeps it.
    topic_id, _ = _tombstoned_with_runs(1)
    real_remove = topics._remove_topic_files

    def scheduler_starts(tid):
        real_remove(tid)
        _lock(files.utcnow())

    topics._remove_topic_files = scheduler_starts
    try:
        topics.cleanup_deleted()
    finally:
        topics._remove_topic_files = real_remove
    assert files.read_json(files.scheduler_lock_path())["owner"] == "scheduler"


def test_cleanup_falls_back_to_run_id_when_task_id_missing():
    defn = _definition(status="deleted")
    topics.save_new_definition(defn)
    run_id = str(uuid.uuid4())
    task_id = files.task_id_for_run(run_id)
    files.write_json_atomic(files.run_path(defn.topic_id, run_id), {"run_id": run_id, "status": "starting"})
    result_store.save_result(task_id, {"task_id": task_id})
    assert topics.cleanup_deleted() == 1
    _nothing_left(defn.topic_id, [task_id])


def test_scheduler_alive_reads_heartbeat():
    now = datetime(2026, 10, 18, 12, 0, tzinfo=timezone.utc)
    assert topics.scheduler_alive(now) is False
    _lock(now - timedelta(seconds=179))
    assert topics.scheduler_alive(now) is True
    _lock(now - timedelta(seconds=180))
    assert topics.scheduler_alive(now) is False
    files.scheduler_lock_path().write_text("garbage")
    assert topics.scheduler_alive(now) is False


@pytest.mark.asyncio
async def test_cleanup_daily_runs_once_then_sleeps(monkeypatch):
    import asyncio

    calls = []
    monkeypatch.setattr(topics, "cleanup_deleted", lambda *a, **k: calls.append(1) or 0)

    async def stop(_):
        raise asyncio.CancelledError

    monkeypatch.setattr(topics.asyncio, "sleep", stop)
    with pytest.raises(asyncio.CancelledError):
        await topics.cleanup_daily()
    assert calls == [1]
