"""Shadow-week instrumentation: collected item details, timing metrics, exported registry."""
import json

import pytest

from app.core import metrics
from app.core.config import settings
from app.models.task import WorkflowState
from app.security import hop_guard
from app.security.context_defender import set_model_for_tests
from app.security.errors import GuardStop
from app.security.types import ContextItem, Decision


@pytest.fixture(autouse=True)
def _reset():
    set_model_for_tests(None)
    yield
    set_model_for_tests(None)


def _guard(monkeypatch, tmp_path, env="uat", mode="shadow"):
    monkeypatch.setattr(settings, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(settings, "WEBAGENT_ENV", env)
    monkeypatch.setattr(settings, "SECURITY_GUARD", settings.SECURITY_GUARD.model_copy(
        update={"enabled": True, "input_mode": mode, "collect": True}))


def _rows(tmp_path):
    return [json.loads(line) for f in sorted(tmp_path.rglob("collected/*.jsonl"))
            for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_checkpoint1_rows_carry_item_details_but_no_text_outside_dev(monkeypatch, tmp_path):
    _guard(monkeypatch, tmp_path)
    state = WorkflowState(query="q", context={"task_id": "t-1"})
    hop_guard.check_hop("web_research", [
        ContextItem("web_result", "the page body text", ref="https://example.com/a"),
        ContextItem("document", "an uploaded document body", ref="Q3 board minutes.pdf"),
    ], state)
    rows = [r for r in _rows(tmp_path) if r.get("item")]
    assert {r["item"]["source"] for r in rows} == {"web_result", "document"}
    web = next(r for r in rows if r["item"]["source"] == "web_result")
    doc = next(r for r in rows if r["item"]["source"] == "document")
    assert web["item"]["ref"] == "https://example.com/a"   # public URL, for re-checking the page
    assert "ref" not in doc["item"]                         # document titles may be private
    assert set(web["item"]["axes"]) == {"injection", "harm", "sensitive_data", "jailbreak"}
    assert not any("text" in r["item"] for r in rows)
    assert "page body" not in json.dumps(rows)


def test_checkpoint1_rows_include_text_in_dev(monkeypatch, tmp_path):
    _guard(monkeypatch, tmp_path, env="dev")
    hop_guard.check_hop("web_research", [ContextItem("web_result", "dev page text", ref="https://x")],
                        WorkflowState(query="q", context={"task_id": "t-2"}))
    assert any(r.get("item", {}).get("text") == "dev page text" for r in _rows(tmp_path))


def test_hop_time_is_recorded_even_when_the_hop_stops_the_task(monkeypatch, tmp_path):
    _guard(monkeypatch, tmp_path, mode="act")

    def refuse(principal, action, resource, context, shadow=None):
        return Decision("decline", ("user-harm-decline",), "no", False, 1.0)

    monkeypatch.setattr("app.security.hop_guard.decide", refuse)
    hist = metrics.GUARD_HOP_SECONDS.labels(agent="intake")
    before = hist._sum.get()
    count_before = sum(b.get() for b in hist._buckets)
    with pytest.raises(GuardStop):
        hop_guard.check_hop("intake", [ContextItem("user_turn", "a request")],
                            WorkflowState(query="q", context={"task_id": "t-3"}))
    assert sum(b.get() for b in hist._buckets) == count_before + 1
    assert hist._sum.get() >= before


def test_new_guard_metrics_are_exported():
    from prometheus_client import generate_latest

    metrics.GUARD_ALIGNMENT.labels(tainted="false").observe(42)
    metrics.GUARD_SHADOW_POLICY.labels(policy_id="off-intent-search-verify", action="tool::web_search").inc()
    metrics.GUARD_TOOL_SECONDS.labels(action="tool::web_search").observe(0.01)
    exported = generate_latest(metrics.WEBAGENT_REGISTRY).decode()
    for name in ("webagent_guard_search_alignment", "webagent_guard_shadow_policy_total",
                 "webagent_guard_hop_seconds", "webagent_guard_tool_seconds"):
        assert name in exported, name


def test_checkpoint2_rows_carry_alignment_and_taint(monkeypatch, tmp_path):
    import asyncio

    from app.security import signals, tool_guard

    _guard(monkeypatch, tmp_path)
    monkeypatch.setattr(settings, "SECURITY_GUARD", settings.SECURITY_GUARD.model_copy(
        update={"enabled": True, "tool_mode": "shadow", "collect": True}))
    monkeypatch.setattr(signals, "score_text", lambda state, text, **_: signals.TextSignals(
        intent_alignment=12, task_alignment=12, harm=0, available=True))
    state = WorkflowState(query="what's the weather in Paris", context={"task_id": "t-4"})
    assert asyncio.run(tool_guard.guard_tool("tool::web_search", {"query": "unlock the front door"}, state))
    row = next(r for r in _rows(tmp_path) if r["checkpoint"] == 2)
    assert row["context"]["alignment"] == 12 and row["context"]["tainted"] is False
    assert row["context"]["intent_alignment"] == 12 and row["context"]["signals_available"] is True
    assert row["shadow_policy_ids"] == ["off-intent-search-verify"]
