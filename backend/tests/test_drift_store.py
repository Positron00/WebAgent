"""Shared drift store (OBS-005 – OBS-011). conftest points settings.DATA_DIR at a temp folder."""
import asyncio
import json
import logging
from datetime import datetime, timedelta

import numpy as np
import pytest

from app.core import drift_store
from app.core.config import settings

T0 = datetime(2026, 10, 1, 9, 0, 0)


def write(detector, values, start=T0, step=timedelta(minutes=1)):
    """Append samples with chosen timestamps (one file per day, like the store)."""
    folder = drift_store.drift_dir()
    folder.mkdir(parents=True, exist_ok=True)
    for i, value in enumerate(values):
        ts = start + i * step
        with open(folder / f"samples-{ts:%Y-%m-%d}.jsonl", "a") as handle:
            handle.write(json.dumps({"ts": ts.isoformat(), "d": detector, "v": value}) + "\n")
    return start + len(values) * step


def test_add_writes_one_line_and_respects_the_switch(monkeypatch):
    drift_store.add("llm_speed:supervisor", 4.2)
    lines = [json.loads(line) for path in drift_store.drift_dir().glob("samples-*.jsonl")
             for line in path.read_text().splitlines()]
    assert len(lines) == 1 and lines[0]["d"] == "llm_speed:supervisor" and lines[0]["v"] == 4.2
    monkeypatch.setattr(settings, "DRIFT_ENABLED", False)
    drift_store.add("llm_speed:supervisor", 4.2)
    assert sum(1 for p in drift_store.drift_dir().glob("*.jsonl") for _ in p.read_text().splitlines()) == 1


def test_not_enough_data_until_a_frozen_reference_and_30_current():
    rng = np.random.default_rng(1)
    end = write("llm_speed:a", list(rng.normal(5, 1, 49)))
    det = drift_store.compute_once(end)["detectors"]["llm_speed:a"]
    assert not det["enough_data"] and det["score"] is None and det["reference_count"] == 49
    end = write("llm_speed:a", list(rng.normal(5, 1, 30)), start=end)  # reference done (50) + 29 current
    det = drift_store.compute_once(end)["detectors"]["llm_speed:a"]
    assert not det["enough_data"] and det["current_count"] == 29
    end = write("llm_speed:a", [5.0], start=end)
    det = drift_store.compute_once(end)["detectors"]["llm_speed:a"]
    assert det["enough_data"] and det["current_count"] == 30 and det["score"] is not None


def test_shifted_samples_are_drift_and_alert_once(caplog):
    rng = np.random.default_rng(2)
    end = write("llm_speed:a", list(rng.normal(5, 1, 50)))
    end = write("llm_speed:a", list(rng.normal(12, 1, 30)), start=end)
    with caplog.at_level(logging.WARNING):
        state = drift_store.compute_once(end)
        drift_store.compute_once(end + timedelta(minutes=1))
    det = state["detectors"]["llm_speed:a"]
    assert det["drift"] and det["score"] > 0.95 and det["last_drift_at"]
    assert [a["name"] for a in state["alerts"]] == ["model_drift"]
    assert sum("Drift detected in llm_speed:a" in r.message for r in caplog.records) == 1
    assert sum("Alert model_drift" in r.message for r in caplog.records) == 1


def test_frozen_reference_survives_retention():
    rng = np.random.default_rng(3)
    end = write("doc_size", list(rng.normal(1000, 50, 50)))
    reference = drift_store.compute_once(end)["detectors"]["doc_size"]["reference"]
    later = end + timedelta(days=60)
    assert drift_store.cleanup(settings.RESULTS_RETENTION_DAYS, later) >= 1
    end = write("doc_size", list(rng.normal(1000, 50, 30)), start=later)
    det = drift_store.compute_once(end)["detectors"]["doc_size"]
    assert det["reference"] == reference and det["enough_data"] and not det["drift"]


def test_reset_starts_a_new_reference_for_its_group_only():
    end = write("llm_tokens:a", [100] * 50)
    end = write("doc_type", ["pdf"] * 50, start=end)
    drift_store.compute_once(end)
    drift_store.reset("llm", now=end)
    state = drift_store.load_state()
    assert "llm_tokens:a" not in state["detectors"] and "doc_type" in state["detectors"]
    det = drift_store.compute_once(end + timedelta(minutes=1))["detectors"]["llm_tokens:a"]
    assert det["reference_count"] == 0 and "reference" not in det


def test_every_worker_and_a_restart_read_the_same_state(monkeypatch):
    end = write("llm_tokens:a", [100] * 50)
    computed = drift_store.compute_once(end)
    monkeypatch.setattr(drift_store, "_last_state", {})  # another worker / a restart: no memory
    assert drift_store.load_state() == computed


def test_unreadable_state_keeps_the_last_good_copy():
    end = write("llm_tokens:a", [100] * 50)
    good = drift_store.compute_once(end)
    (drift_store.drift_dir() / "state.json").write_text("{not json")
    assert drift_store.load_state() == good


def test_bad_lines_are_skipped():
    end = write("llm_tokens:a", [100] * 50)
    with open(next(drift_store.drift_dir().glob("samples-*.jsonl")), "a") as handle:
        handle.write("{torn line\n")
    assert drift_store.compute_once(end)["detectors"]["llm_tokens:a"]["reference_count"] == 50


@pytest.mark.asyncio
async def test_two_loops_compute_once_per_tick(monkeypatch):
    calls = []

    def fake_compute():
        calls.append(1)
        import time

        time.sleep(0.3)  # hold the lock while the other loop tries
        return drift_store._empty_state()

    monkeypatch.setattr(drift_store, "compute_once", fake_compute)
    loops = [asyncio.create_task(drift_store.compute_loop(interval_sec=3600)) for _ in range(2)]
    await asyncio.sleep(0.6)
    for loop in loops:
        loop.cancel()
    await asyncio.gather(*loops, return_exceptions=True)
    assert len(calls) == 1


def test_cleanup_deletes_only_old_files():
    write("doc_size", [1.0], start=T0)
    write("doc_size", [1.0], start=T0 + timedelta(days=40))
    assert drift_store.cleanup(30, now=T0 + timedelta(days=41)) == 1
    assert len(list(drift_store.drift_dir().glob("samples-*.jsonl"))) == 1


def test_an_unreadable_state_is_logged_once(caplog):
    end = write("llm_tokens:a", [100] * 50)
    drift_store.compute_once(end)
    (drift_store.drift_dir() / "state.json").write_text("{not json")
    with caplog.at_level(logging.WARNING):
        for _ in range(3):
            drift_store.load_state()
    assert sum("Drift state unreadable" in r.message for r in caplog.records) == 1
