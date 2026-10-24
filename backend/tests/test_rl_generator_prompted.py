"""Prompted generation's selection, batches, resume, stop rule and judge candidates (rl-generator SEC-208).

Offline: a fake generator returning fixed placeholder completions, a scripted target and fake judges.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import alignment_data  # noqa: E402
from rl_generator import prompted, rewards, rows  # noqa: E402


def fake_embed(texts):
    out = []
    for t in texts:
        v = np.random.default_rng(int(hashlib.sha256(t.encode()).hexdigest(), 16) % 2**32).normal(size=384)
        out.append(v / np.linalg.norm(v))
    return np.asarray(out, dtype=np.float32)


class FakeTarget:
    version = "ac-test"

    def __init__(self, scores):
        self.scores = scores

    def score_pair(self, intent, call_text):
        return self.scores.get(call_text.split(": ", 1)[1], (0.1, 0))


class FakeTokenizer:
    def __call__(self, a, b):
        return {"input_ids": list(range(len(a.split()) + len(b.split())))}


class FakeJudge:
    def __init__(self, model="judge-model", verdicts=None, default=True, cost_per_call=0.0):
        self.model, self.verdicts, self.default = model, verdicts or {}, default
        self.judged, self.cost_per_call, self.seeds_ended = [], cost_per_call, 0

    def __call__(self, intent, call, action=""):
        self.judged.append(call)
        return self.verdicts.get(call, self.default)

    def spent(self):
        return self.cost_per_call * len(self.judged)

    def error_stop(self):
        return False

    def end_seed(self):
        self.seeds_ended += 1

    def report(self):
        return {"model": self.model, "calls": len(self.judged)}


# eight placeholder samples per seed: miss 0.9 .. 0.2; "s7" is harmful, "s6" too long
SCORES = {f"s{i}": (round(0.9 - 0.1 * i, 2), 0) for i in range(8)}
COMPLETIONS = [f"s{i}" for i in range(8)]
COMPLETIONS[6] = "s6 " + "word " * 140


def seed(i):
    return {"seed_id": f"seed{i:03d}", "part": "sample", "intent": f"placeholder intent {i}", "call": "placeholder",
            "action": "web_search", "origin": "asq:query", "groups": [f"g{i}"], "sources": ["asq"]}


def make_ctx(judge, mode="H"):
    return rewards.RewardContext(mode, FakeTarget(SCORES), np.zeros((0, 384)), FakeTokenizer(), judge,
                                 harm_verify_at=20, harm=lambda t: 90 if t == "s7" else 0, embed=fake_embed)


def generate(prompts):
    return [list(COMPLETIONS) for _ in prompts]


def prov(tmp_path, batch="h-prompted-20261024-1"):
    return prompted.Provenance("prompted", "base-model", "ac-test", "hash1", judge="judge-model",
                               batch=batch, out=tmp_path / "rl_gen_h_prompted.jsonl")


def test_pick_judges_best_two_and_keeps_best_valid():
    judge = FakeJudge(verdicts={"s0": False})
    (call, best), judged, _, scored = prompted.pick(make_ctx(judge), seed(1), COMPLETIONS)
    assert judge.judged == ["s0", "s1"]
    assert call == "s1" and best.valid and best.reward > 0
    assert len(judged) == 2 and len(scored) == 8


def test_nothing_kept_when_both_fail():
    (call, best), _, _, _ = prompted.pick(make_ctx(FakeJudge(default=False)), seed(1), COMPLETIONS)
    assert call is None and best is None


def test_writes_valid_rows_with_provenance(tmp_path):
    p = prov(tmp_path)
    report = prompted.select_and_write(make_ctx(FakeJudge()), [seed(i) for i in range(5)], generate, p, n=3)
    assert report["rows"] == 3 and report["seeds"] == 3 and report["harmful_filtered"] == 3
    written = rows.read_rows(p.out)
    assert len(written) == 3
    for r in written:
        assert alignment_data.schema_errors(r) == [] and alignment_data.rule_errors(r, committed=False) == []
        assert r["origin"] == "rl-gen-h:prompted" and r["labels"] == {"aligned": "no", "risk": "none"}
        assert r["sources"] == ["rl-gen-h", "asq"]
        meta = r["meta"]
        assert meta["batch"] == p.batch and meta["prompt"] == "p1" and meta["target"] == "ac-test"
        assert meta["adapter_step"] is None and meta["data_hash"] == "hash1" and meta["judge"] == "judge-model"


def test_resume_skips_done_seeds(tmp_path):
    p = prov(tmp_path)
    prompted.select_and_write(make_ctx(FakeJudge()), [seed(i) for i in range(5)], generate, p, n=2)
    judge = FakeJudge()
    report = prompted.select_and_write(make_ctx(judge), [seed(i) for i in range(5)], generate, p, n=4)
    assert report["resumed_rows"] == 2 and report["rows"] == 2
    assert [r["meta"]["seed_id"] for r in rows.read_rows(p.out)] == [f"seed{i:03d}" for i in range(4)]


def test_estimate_writes_no_batch(tmp_path):
    p = prov(tmp_path, batch=None)
    report = prompted.select_and_write(make_ctx(FakeJudge()), [seed(i) for i in range(4)], generate, p, n=500)
    assert report["seeds"] == 4 and report["rows"] == 4 and not p.out.exists()
    assert "projected_cost_for_n_usd" in report and "seeds_per_minute" in report


def test_credit_stop_rule(tmp_path):
    judge = FakeJudge(cost_per_call=0.1)   # 0.2 per seed; half of 1.0 is reached after 2 seeds
    report = prompted.select_and_write(make_ctx(judge), [seed(i) for i in range(10)], generate, prov(tmp_path),
                                       n=10, credit_left=1.0)
    assert report["stopped"] == "credit rule" and report["seeds"] == 2


def test_judge_candidates(tmp_path):
    a, b = FakeJudge("model-a"), FakeJudge("model-b", default=False)
    out = tmp_path / "judge-choice.jsonl"
    report = prompted.select_and_write(make_ctx(None), [seed(i) for i in range(3)], generate,
                                       prov(tmp_path, batch=None), n=500, candidates=[a, b], choice_out=out)
    assert report["rows"] == 0 and report["accepted_by_candidate"] == {"model-a": 6, "model-b": 0}
    items = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(items) == 6 and all(set(i["verdicts"]) == {"model-a", "model-b"} for i in items)
    assert a.judged == b.judged and len(a.judged) == 6


def test_args(capsys):
    base = ["--mode", "H", "--judge-model", "m"]
    with pytest.raises(SystemExit):
        prompted.parse_args(base)   # a full run needs --credit-left
    assert prompted.parse_args(base + ["--estimate", "50"]).estimate == 50
    with pytest.raises(SystemExit):
        prompted.parse_args(["--mode", "H", "--judge-candidates", "a,b", "--credit-left", "5"])
    args = prompted.parse_args(["--mode", "l", "--judge-candidates", "a,b", "--estimate", "5"])
    assert args.mode == "L" and args.candidates == ["a", "b"]
