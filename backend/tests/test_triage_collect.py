"""Collection (REQ-030-032): opt-in, written only under data/triage/collected/, 30-day cleanup."""
import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from app.core.config import settings
from app.services import triage_collect
from app.services.triage import TriageDecision


@pytest.fixture
def collecting(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "TRIAGE", settings.TRIAGE.model_copy(update={"collect": True}))
    return tmp_path


def test_nothing_written_when_collect_off(tmp_path):
    triage_collect.record("shadow", "hello", None, None)
    assert not (tmp_path / "triage").exists()


def test_record_writes_one_line_only_under_collected(collecting):
    decision = TriageDecision("direct", 0.2, "model", "Looks like: coding help", "v1", 1.0)
    triage_collect.record("shadow", "x" * 3000, decision, "previous question")
    files = [p for p in collecting.rglob("*") if p.is_file()]
    assert len(files) == 1
    assert files[0].parent == collecting / "triage" / "collected"
    assert files[0].name == f"{datetime.now(timezone.utc).date().isoformat()}.jsonl"
    line = files[0].read_text()
    assert '"route": "direct"' in line and '"classifier_version": "v1"' in line
    assert '"' + "x" * 2000 + '"' in line  # text cut to 2,000 characters


def test_write_failure_is_swallowed(collecting, monkeypatch):
    def broken_open(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(triage_collect.os, "open", broken_open)
    triage_collect.record("override", "q", None, None)  # must not raise


def test_cleanup_deletes_only_old_files(collecting):
    d = collecting / "triage" / "collected"
    d.mkdir(parents=True)
    now = datetime(2026, 10, 11, tzinfo=timezone.utc)
    old = d / f"{(now - timedelta(days=31)).date().isoformat()}.jsonl"
    recent = d / f"{(now - timedelta(days=5)).date().isoformat()}.jsonl"
    other = d / "notes.jsonl"
    for p in (old, recent, other):
        p.write_text("{}\n")
    assert triage_collect.cleanup(now=now) == 1
    assert not old.exists() and recent.exists() and other.exists()


async def test_daily_cleanup_runs_repeatedly(monkeypatch):
    calls = []
    monkeypatch.setattr(triage_collect, "cleanup", lambda: calls.append(1))
    task = asyncio.create_task(triage_collect.cleanup_daily(interval_sec=0.01))
    await asyncio.sleep(0.05)
    task.cancel()
    assert len(calls) >= 2
