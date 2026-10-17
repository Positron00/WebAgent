"""Benign call generation (checkpoint2-intent SEC-122), with a fake supervisor chain."""
import asyncio
import importlib.util
import json
import sys
import types
from pathlib import Path

from app.core import observability

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "generate_alignment_calls.py"


def _load():
    spec = importlib.util.spec_from_file_location("generate_alignment_calls", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeChain:
    def __init__(self, prompt_tokens=1000, completion_tokens=200):
        self.calls = 0
        self.usage = (prompt_tokens, completion_tokens)

    async def ainvoke(self, payload):
        self.calls += 1
        observability.record_model_token_usage("together", *self.usage)
        return {"research_plan": {"standalone_query": f"rewrite of {payload['query']}", "questions": ["q1"]}}


def _install_fake_supervisor(monkeypatch, chain):
    fake = types.ModuleType("app.agents.supervisor")
    fake.SupervisorAgent = lambda: types.SimpleNamespace(chain=chain)
    monkeypatch.setitem(sys.modules, "app.agents.supervisor", fake)


def _questions(tmp_path, n):
    path = tmp_path / "q.jsonl"
    path.write_text("".join(json.dumps({"text": f"question {i}"}) + "\n" for i in range(n)))
    return path


def _args(module, tmp_path, n, **kw):
    return types.SimpleNamespace(questions=_questions(tmp_path, n), out=tmp_path / "out.jsonl",
                                 estimate=kw.get("estimate", 0), credit_left=kw.get("credit_left"),
                                 price_input=1.0, price_output=1.0)


def test_estimate_writes_rows_and_reports_cost(monkeypatch, tmp_path):
    module = _load()
    monkeypatch.setattr(observability, "_model_token_totals", {})
    chain = FakeChain()
    _install_fake_supervisor(monkeypatch, chain)
    report = asyncio.run(module.run(_args(module, tmp_path, 10, estimate=4)))
    rows = [json.loads(line) for line in (tmp_path / "out.jsonl").read_text().splitlines()]
    assert len(rows) == 4 and report["questions_written"] == 4 and chain.calls == 4
    assert rows[0]["origin"] == "webagent-gen" and rows[0]["questions"] == ["q1"]
    assert report["cost_usd"] == round(4 * 1200 / 1e6, 4)
    assert report["projected_full_run_usd"] == round(report["cost_usd"] / 4 * 10, 2)


def test_resume_skips_done_questions(monkeypatch, tmp_path):
    module = _load()
    monkeypatch.setattr(observability, "_model_token_totals", {})
    chain = FakeChain()
    _install_fake_supervisor(monkeypatch, chain)
    args = _args(module, tmp_path, 5, estimate=5)
    asyncio.run(module.run(args))
    asyncio.run(module.run(args))
    assert chain.calls == 5


def test_full_run_stops_before_half_the_credit(monkeypatch, tmp_path):
    module = _load()
    monkeypatch.setattr(observability, "_model_token_totals", {})
    chain = FakeChain(prompt_tokens=100_000, completion_tokens=0)  # $0.10 a question at $1/M
    _install_fake_supervisor(monkeypatch, chain)
    report = asyncio.run(module.run(_args(module, tmp_path, 50, credit_left=1.0)))
    assert report["cost_usd"] <= 0.5 and chain.calls == 5


def test_code_mode_records_code_and_explanation(monkeypatch, tmp_path):
    module = _load()
    monkeypatch.setattr(observability, "_model_token_totals", {})

    class FakeCoder:
        async def run(self, state):
            observability.record_model_token_usage("together", 100, 50)
            state.reports["coding_assistant"] = {"code": "print(1)", "explanation": "prints one"}
            return state

    fake = types.ModuleType("app.agents.coding_assistant")
    fake.CodingAssistantAgent = FakeCoder
    monkeypatch.setitem(sys.modules, "app.agents.coding_assistant", fake)
    args = _args(module, tmp_path, 3, estimate=2)
    args.code = True
    report = asyncio.run(module.run(args))
    rows = [json.loads(line) for line in (tmp_path / "out.jsonl").read_text().splitlines()]
    assert report["questions_written"] == 2 and rows[0]["code"] == "print(1)" and rows[0]["kind"] == "code"
