"""Intent record and plan check (checkpoint2-intent SEC-100 – SEC-102, SEC-107)."""
import asyncio
import json
import logging

import pytest

from app.models.task import WorkflowState
from app.security import alignment, intent, ledger, signals
from app.security.errors import GuardStop
from app.security.types import Decision
from app.services.task_manager import TaskManager
from tests.intent_fakes import WordEncoder, guard_on, keyword_harm


@pytest.fixture(autouse=True)
def fakes(monkeypatch):
    alignment.set_encoder_for_tests(WordEncoder())
    monkeypatch.setattr(signals, "harm_of", keyword_harm)
    yield
    alignment.set_encoder_for_tests(None)


def _state(query, previous=None, plan=None):
    ctx = {"task_id": "t-int", "previous_messages": previous or [],
           "research_plan": plan or {"standalone_query": "", "questions": []}}
    state = WorkflowState(query=query, context=ctx)
    intent.record_intent(state)
    return state


def _decision(outcome):
    return Decision(outcome=outcome, policy_ids=(f"test-{outcome}",), reason="r", shadow=False, latency_ms=0.1)


def test_intent_is_the_users_turns_only_and_capped(monkeypatch, tmp_path):
    guard_on(monkeypatch, tmp_path)
    previous = [{"role": "user", "content": f"turn {i}"} for i in range(5)]
    previous.insert(2, {"role": "assistant", "content": "an assistant reply"})
    state = _state("final question", previous)
    text = intent.intent_text(state)
    assert "assistant reply" not in text and text.splitlines() == ["turn 2", "turn 3", "turn 4", "final question"]
    long_state = _state("x" * 5000)
    assert len(intent.intent_text(long_state)) == intent.INTENT_MAX_CHARS
    state.context["research_plan"]["standalone_query"] = "something else entirely"
    assert intent.intent_text(state).endswith("final question")  # never built from agent output
    assert signals.intent_harm(state) == 5  # computed on first use


def test_the_intent_never_reaches_results(monkeypatch, tmp_path):
    guard_on(monkeypatch, tmp_path)
    state = _state("a private question about my health")
    signals.intent_embedding(state)
    block = TaskManager._security_block(state.context)
    assert "private question" not in json.dumps(block) and "intent" not in block


def test_shadow_mode_records_a_drifted_rewrite_and_changes_nothing(monkeypatch, tmp_path, caplog):
    guard_on(monkeypatch, tmp_path, tool_mode="shadow")
    plan = {"standalone_query": "quantum computing hardware vendors", "questions": ["exploit malware kits"]}
    state = _state("weather forecast paris tomorrow", plan=plan)
    with caplog.at_level(logging.INFO):
        asyncio.run(intent.check_plan(state))
    assert plan["standalone_query"] == "quantum computing hardware vendors" and plan["questions"]
    lines = [r.message for r in caplog.records if "guard plan" in r.message]
    assert any("rewrite-drift-verify" in line for line in lines)
    assert any("subgoal-harm-decline" in line or "subgoal-off-intent-decline" in line for line in lines)
    assert ledger.declined_actions(state) == 0


def test_act_mode_falls_back_and_removes_and_counts(monkeypatch, tmp_path):
    guard_on(monkeypatch, tmp_path, tool_mode="act")
    monkeypatch.setattr(intent, "decide", lambda *a, **k: _decision("decline"))
    plan = {"standalone_query": "a drifted rewrite", "questions": ["q one", "q two"]}
    state = _state("the user's question", plan=plan)
    asyncio.run(intent.check_plan(state))
    assert plan["standalone_query"] == "" and plan["questions"] == []
    assert ledger.declined_actions(state) == 2
    kept = asyncio.run(intent.check_follow_ups(state, ["follow up"]))
    assert kept == []
    skipped = ledger.get_ledger(state)["skipped"]
    assert [s["origin"] for s in skipped] == ["plan", "plan", "follow_up"]
    assert all("text" not in s for s in skipped)  # no text outside dev


def test_escalate_stops_the_task(monkeypatch, tmp_path):
    guard_on(monkeypatch, tmp_path, tool_mode="act")
    monkeypatch.setattr(intent, "decide", lambda *a, **k: _decision("escalate"))
    state = _state("the user's question")
    with pytest.raises(GuardStop):
        asyncio.run(intent.check_follow_ups(state, ["a follow-up"]))


def test_plan_check_fails_open(monkeypatch, tmp_path):
    guard_on(monkeypatch, tmp_path, tool_mode="act")

    def boom(*a, **k):
        raise RuntimeError("engine down")

    monkeypatch.setattr(intent, "decide", boom)
    plan = {"standalone_query": "rewrite", "questions": ["q"]}
    state = _state("the question", plan=plan)
    asyncio.run(intent.check_plan(state))
    assert plan == {"standalone_query": "rewrite", "questions": ["q"]}
    assert asyncio.run(intent.check_follow_ups(state, ["f"])) == ["f"]


def test_plan_check_switch(monkeypatch, tmp_path):
    guard_on(monkeypatch, tmp_path, tool_mode="act", plan_check=False)
    monkeypatch.setattr(intent, "decide", lambda *a, **k: _decision("decline"))
    plan = {"standalone_query": "rewrite", "questions": ["q"]}
    asyncio.run(intent.check_plan(_state("the question", plan=plan)))
    assert plan["questions"] == ["q"]


@pytest.mark.parametrize("action,plan,query,expected", [
    ("tool::web_search", {"requires_web_search": True}, "q", True),
    ("tool::internal_retrieve", {"requires_internal_knowledge": False}, "q", False),
    ("tool::document_extract", {"requires_document_extraction": True}, "q", True),
    ("tool::execute_code", {}, "plot the monthly sales as a chart", True),
    ("tool::execute_code", {}, "who won the 1998 world cup", False),
    ("tool::web_search", None, "q", True),
])
def test_declared_tools(action, plan, query, expected):
    state = WorkflowState(query=query, context={"research_plan": plan} if plan is not None else {})
    assert intent.declared_for(state, action) is expected
