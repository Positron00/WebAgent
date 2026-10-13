"""Security collection and escalation records (SEC-060, SEC-064)."""
import json
from datetime import datetime, timedelta, timezone

import pytest

from app.core.config import settings
from app.security import records
from app.security.types import Decision


@pytest.fixture
def collecting(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(
        settings,
        "SECURITY_GUARD",
        settings.SECURITY_GUARD.model_copy(update={"collect": True, "collect_retention_days": 30}),
    )
    return tmp_path


def test_nothing_written_when_collect_off(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(
        settings,
        "SECURITY_GUARD",
        settings.SECURITY_GUARD.model_copy(update={"collect": False}),
    )
    decision = Decision("decline", ("p",), "no", False, 1.0)
    records.record_collection(
        checkpoint=2,
        action="tool::web_search",
        decision=decision,
        state_context={"task_id": "t"},
        args={"query": "x"},
    )
    assert not list(tmp_path.rglob("*.jsonl"))


def test_collection_writes_under_security_collected(collecting):
    decision = Decision("decline", ("search-secret-decline",), "no", True, 2.0)
    records.record_collection(
        checkpoint=2,
        action="tool::web_search",
        decision=decision,
        state_context={"task_id": "t", "query": "secret"},
        args={"query": "api_key=abc"},
    )
    files = list(collecting.rglob("*.jsonl"))
    assert len(files) == 1
    assert "security/collected" in str(files[0])
    row = json.loads(files[0].read_text(encoding="utf-8").strip())
    assert row["outcome"] == "decline"
    assert row["shadow"] is True


def test_escalation_appends_line(collecting, monkeypatch):
    monkeypatch.setattr(settings, "WEBAGENT_ENV", "prod")
    decision = Decision("escalate", ("repeated-declines-escalate",), "review", False, 1.0)
    records.record_escalation(
        hop="web_research",
        action="tool::web_search",
        decision=decision,
        state_context={"task_id": "t", "query": "hello"},
        args={"query": "hello"},
    )
    path = collecting / "security" / "escalations" / f"{datetime.now(timezone.utc).date().isoformat()}.jsonl"
    assert path.is_file()
    row = json.loads(path.read_text(encoding="utf-8").strip())
    assert row["text_sha256"]
    assert "text" not in row


def test_cleanup_deletes_old_files(collecting):
    d = collecting / "security" / "collected"
    d.mkdir(parents=True)
    old = (datetime.now(timezone.utc) - timedelta(days=40)).date().isoformat()
    (d / f"{old}.jsonl").write_text("{}\n", encoding="utf-8")
    today = datetime.now(timezone.utc).date().isoformat()
    (d / f"{today}.jsonl").write_text("{}\n", encoding="utf-8")
    deleted = records.cleanup(now=datetime.now(timezone.utc))
    assert deleted >= 1
    assert (d / f"{today}.jsonl").exists()
    assert not (d / f"{old}.jsonl").exists()
