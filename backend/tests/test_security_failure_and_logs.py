"""Checkpoint 2 judge failure (SEC-061) and no raw text in guard logs outside dev (SNFR-09)."""
import logging
import time
from types import SimpleNamespace

import pytest

from app.core.config import settings
from app.models.task import WorkflowState
from app.security import hop_guard, judge as guard_judge
from app.security.tool_guard import guard_tool

MARKER = "zebra-marker-7781"


@pytest.fixture
def judge_on(monkeypatch):
    monkeypatch.setattr(
        settings,
        "SECURITY_GUARD",
        settings.SECURITY_GUARD.model_copy(update={"enabled": True, "tool_mode": "act", "judge": True}),
    )
    guard_judge.clear_cache()
    yield
    guard_judge.clear_cache()


def _state(query: str = "What is the weather in Paris?") -> WorkflowState:
    return WorkflowState(query=query, context={"task_id": "task-sec061"}, inputs={})


class _SlowLLM:
    def invoke(self, _prompt):
        time.sleep(0.5)
        return "ALIGNED"


class _BrokenLLM:
    def invoke(self, _prompt):
        raise RuntimeError("provider down")


@pytest.mark.asyncio
@pytest.mark.parametrize("llm", [_SlowLLM(), _BrokenLLM()], ids=["timeout", "error"])
async def test_judge_failure_at_checkpoint_2_is_not_aligned(judge_on, monkeypatch, llm):
    """SEC-061: a checkpoint-2 judge timeout or error counts as NOT_ALIGNED, so a verify outcome declines."""
    monkeypatch.setattr(guard_judge, "JUDGE_TIMEOUT_SEC", 0.05)
    monkeypatch.setattr("app.services.llm.get_trusted_llm", lambda **kw: llm)

    assert guard_judge.is_aligned(_state(), "tool::web_search", {"query": "user@example.com please research"}) is False

    guard_judge.clear_cache()
    allowed = await guard_tool(
        "tool::web_search",
        {"query": "user@example.com please research"},
        _state(),
        principal="web_research",
    )
    assert allowed is False


@pytest.mark.asyncio
@pytest.mark.parametrize("env, text_logged", [("uat", False), ("prod", False), ("dev", True)])
async def test_checkpoint_2_log_line_carries_no_text_outside_dev(monkeypatch, caplog, env, text_logged):
    """SNFR-09: the checkpoint-2 decision log line has the query only in dev."""
    monkeypatch.setattr(settings, "WEBAGENT_ENV", env)
    monkeypatch.setattr(
        settings,
        "SECURITY_GUARD",
        settings.SECURITY_GUARD.model_copy(update={"enabled": True, "tool_mode": "shadow"}),
    )
    with caplog.at_level(logging.DEBUG):
        await guard_tool("tool::web_search", {"query": f"{MARKER} search"}, _state(f"{MARKER} question"),
                         principal="web_research")
    guard_lines = [r.getMessage() for r in caplog.records if "guard checkpoint=2" in r.getMessage()]
    assert guard_lines, "expected a checkpoint-2 decision log line"
    assert any(MARKER in line for line in guard_lines) is text_logged
    if not text_logged:
        assert MARKER not in caplog.text


@pytest.mark.parametrize("env, text_logged", [("uat", False), ("prod", False), ("dev", True)])
def test_checkpoint_1_log_line_carries_no_text_outside_dev(monkeypatch, caplog, env, text_logged):
    """SNFR-09: the checkpoint-1 decision log line has the context item's text only in dev."""
    monkeypatch.setattr(settings, "WEBAGENT_ENV", env)
    decision = SimpleNamespace(outcome="allow", policy_ids=[], shadow=True, error=False)
    item = SimpleNamespace(text=f"{MARKER} retrieved page text", source="web")
    scores = SimpleNamespace(top_axis="none", axes={})
    state = SimpleNamespace(context={"task_id": "task-snfr09"})
    with caplog.at_level(logging.DEBUG):
        hop_guard._log_decision(decision, item, scores, "web_research", state)
    lines = [r.getMessage() for r in caplog.records if "guard checkpoint=1" in r.getMessage()]
    assert len(lines) == 1
    assert (MARKER in lines[0]) is text_logged
