"""Two tasks running at once in one worker must not share tracking runs or agent state."""
import asyncio
from types import SimpleNamespace

import pytest

from app.core import task_tracking


class _FakeClient:
    def __init__(self):
        self.runs, self.params, self.ended = [], {}, []

    def get_experiment_by_name(self, name):
        return SimpleNamespace(experiment_id="1")

    def create_run(self, experiment_id, run_name=None):
        run_id = f"run-{len(self.runs)}"
        self.runs.append(run_id)
        self.params[run_id] = {}
        return SimpleNamespace(info=SimpleNamespace(run_id=run_id))

    def log_param(self, run_id, key, value):
        if key in self.params[run_id] and self.params[run_id][key] != value:
            raise RuntimeError("param changed")
        self.params[run_id][key] = value

    def log_metric(self, run_id, key, value):
        self.params[run_id][f"metric:{key}"] = value

    def set_terminated(self, run_id, status="FINISHED"):
        self.ended.append(run_id)


@pytest.fixture
def fake_mlflow(monkeypatch):
    client = _FakeClient()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "file:///tmp/unused")
    monkeypatch.setattr(task_tracking, "_client", lambda: client)
    monkeypatch.setattr(task_tracking, "_experiments", {})
    return client


def test_concurrent_tasks_get_their_own_runs(fake_mlflow):
    async def task(query):
        task_tracking.start_run("rag_agent_run", {"query": query})
        await asyncio.sleep(0.01)  # the other task runs in between
        task_tracking.log_param("status", "completed")
        task_tracking.log_metric("total_time", len(query))
        task_tracking.end_run()

    async def both():
        await asyncio.gather(task("first question"), task("second, longer question"))

    asyncio.run(both())
    assert len(fake_mlflow.runs) == 2 and sorted(fake_mlflow.ended) == sorted(fake_mlflow.runs)
    queries = {p["query"] for p in fake_mlflow.params.values()}
    assert queries == {"first question", "second, longer question"}
    for p in fake_mlflow.params.values():
        assert p["metric:total_time"] == len(p["query"])


def test_tracking_errors_never_fail_a_task(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "file:///tmp/unused")
    monkeypatch.setattr(task_tracking, "_experiments", {})

    def broken():
        raise RuntimeError("store unavailable")

    monkeypatch.setattr(task_tracking, "_client", broken)
    assert task_tracking.start_run("rag_agent_run", {"query": "q"}) is None
    task_tracking.log_metric("x", 1.0)  # no run: nothing happens
    task_tracking.end_run()


def test_internal_research_checks_each_tasks_own_state(monkeypatch):
    from app.agents import internal_research
    from app.models.task import WorkflowState

    seen = []

    async def fake_acheck_hop(agent, items, state):
        seen.append(state.query)
        await asyncio.sleep(0.01)
        return SimpleNamespace(items=items, dropped=[], decisions=[], tainted=False)

    monkeypatch.setattr(internal_research, "acheck_hop", fake_acheck_hop)
    agent = internal_research.InternalResearchAgent.__new__(internal_research.InternalResearchAgent)
    agent.prompt = SimpleNamespace(format_messages=lambda **kw: [])
    agent.llm = SimpleNamespace(ainvoke=lambda msgs: _done("FINDINGS"))

    async def both():
        a = WorkflowState(query="task A", context={"task_id": "A"})
        b = WorkflowState(query="task B", context={"task_id": "B"})
        await asyncio.gather(agent._generate_findings_with_retry({"context": "doc"}, a),
                             agent._generate_findings_with_retry({"context": "doc"}, b))

    asyncio.run(both())
    assert sorted(seen) == ["task A", "task B"]


async def _done(value):
    return value


class _NotThreadSafeTokenizer:
    """Raises like Hugging Face's fast tokenizer when two threads use it at once."""

    def __init__(self):
        import threading

        self._busy = threading.Lock()

    def _use(self):
        import time

        if not self._busy.acquire(blocking=False):
            raise RuntimeError("Already borrowed")
        try:
            time.sleep(0.002)
        finally:
            self._busy.release()

    def encode(self, text, add_special_tokens=False):
        self._use()
        return list(range(len(text.split())))

    def decode(self, ids, skip_special_tokens=True):
        self._use()
        return " ".join("w" for _ in ids)


class _Encoder:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def encode(self, texts, normalize_embeddings=True):
        import numpy as np

        self.tokenizer._use()  # encode() tokenizes internally
        return np.ones((len(texts), 4))


def test_scoring_from_many_threads_does_not_collide():
    from concurrent.futures import ThreadPoolExecutor

    from app.security import context_defender as cd

    tok = _NotThreadSafeTokenizer()
    enc = _Encoder(tok)
    text = " ".join(["word"] * 400)

    def work(_):
        chunks = cd.chunk_text(text, tok)
        cd.featurize(chunks, "minilm", enc, None)
        return len(chunks)

    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(work, range(40)))  # raised "Already borrowed" without the lock
    assert all(r > 1 for r in results)
