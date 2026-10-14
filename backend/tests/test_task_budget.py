"""Per-task token budget (IMP-042)."""
import asyncio
from uuid import uuid4

import pytest

from app.core.config import settings
from app.services import result_store, task_budget, task_manager as task_manager_module, together_llm
from app.services.task_budget import BUDGET_MESSAGE, TaskBudgetExceeded
from app.services.together_llm import TogetherWithUsage
from tests.fakes.workers import separate_task_manager
from tests.fakes.workflow import StreamingFakeWorkflow


@pytest.mark.asyncio
async def test_counter_is_shared_across_gather_and_threads():
    token = task_budget.start(1000)
    try:
        async def agent():
            task_budget.add(30)
            await asyncio.to_thread(task_budget.add, 5)

        await asyncio.gather(agent(), agent(), agent())
        assert task_budget.used() == 105
    finally:
        task_budget.reset(token)
    assert task_budget.used() == 0


def test_check_raises_only_when_a_budget_is_reached():
    token = task_budget.start(100)
    try:
        task_budget.add(99)
        task_budget.check()
        task_budget.add(1)
        with pytest.raises(TaskBudgetExceeded):
            task_budget.check()
    finally:
        task_budget.reset(token)

    token = task_budget.start(0)
    try:
        task_budget.add(10**9)
        task_budget.check()
    finally:
        task_budget.reset(token)


@pytest.mark.asyncio
async def test_an_agents_except_exception_does_not_swallow_it():
    async def careless_agent():
        try:
            task_budget.check()
        except Exception:
            return "swallowed"
        return "ran"

    token = task_budget.start(10)
    try:
        task_budget.add(10)
        results = await asyncio.gather(careless_agent(), return_exceptions=True)
    finally:
        task_budget.reset(token)
    assert isinstance(results[0], TaskBudgetExceeded)


class _Reply:
    status_code = 200
    text = ""

    def json(self):
        return {"choices": [{"text": "ok"}], "usage": {"total_tokens": 60}}


def test_together_counts_usage_and_checks_before_each_request(monkeypatch):
    calls = []
    monkeypatch.setattr(together_llm.requests, "post", lambda *a, **k: calls.append(1) or _Reply())
    llm = TogetherWithUsage(model="meta-models/Muse-Glimmer-30B", max_tokens=50, together_api_key="test-key")

    token = task_budget.start(100)
    try:
        llm.invoke("one")
        llm.invoke("two")
        assert task_budget.used() == 120
        with pytest.raises(TaskBudgetExceeded):
            llm.invoke("three")
    finally:
        task_budget.reset(token)
    assert len(calls) == 2


class _SpendingWorkflow(StreamingFakeWorkflow):
    """Uses tokens like an agent and then tries to make another model call."""

    def __init__(self, tokens: int):
        self.tokens = tokens

    async def ainvoke(self, state):
        task_budget.add(self.tokens)
        try:
            task_budget.check()
        except Exception:
            pass
        state.mark_completed({"title": "R", "content": "done"})
        return state


@pytest.mark.asyncio
@pytest.mark.parametrize("tokens, status, code", [(60, "error", "budget"), (40, "completed", None)])
async def test_run_ends_with_budget_error_and_records_tokens(monkeypatch, tokens, status, code):
    monkeypatch.setattr(settings, "TASK_TOKEN_BUDGET", 50)
    monkeypatch.setattr(task_manager_module, "get_agent_workflow", lambda: _SpendingWorkflow(tokens))
    manager = separate_task_manager()
    task_id = str(uuid4())
    manager.register_pending(task_id, "q")

    await manager.run_workflow(task_id, "q", {})

    saved = result_store.load_result(task_id)
    assert (saved["status"], saved.get("error_code"), saved["tokens_used"]) == (status, code, tokens)
    if code == "budget":
        assert saved["result"]["error"] == BUDGET_MESSAGE
        assert manager.get_task_status(task_id)[1]["error_code"] == "budget"
