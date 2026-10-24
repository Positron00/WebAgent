"""Reward terms and their order of evaluation (rl-generator SEC-203 – SEC-207).

Offline: a scripted target, fake tokenizer, harm function, embedder and judge; placeholder text.
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from rl_generator import rewards  # noqa: E402

SEED = {"seed_id": "s1", "intent": "placeholder intent", "action": "web_search", "call": "placeholder call",
        "groups": ["g1"], "sources": ["asq"], "origin": "asq:query"}


def fake_embed(texts):
    out = []
    for t in texts:
        v = np.random.default_rng(int(hashlib.sha256(t.encode()).hexdigest(), 16) % 2**32).normal(size=384)
        out.append(v / np.linalg.norm(v))
    return np.asarray(out, dtype=np.float32)


class FakeTarget:
    version = "ac-test"

    def __init__(self, scores=None, default=(0.8, 10)):
        self.scores, self.default, self.calls = scores or {}, default, []

    def score_pair(self, intent, call_text):
        self.calls.append(call_text)
        return self.scores.get(call_text, self.default)


class FakeTokenizer:
    def __call__(self, a, b):
        return {"input_ids": list(range(len(a.split()) + len(b.split())))}


class FakeJudge:
    def __init__(self, answer=True):
        self.answer, self.calls = answer, []

    def __call__(self, intent, call, action=""):
        self.calls.append(call)
        return self.answer


def make_ctx(mode="H", target=None, judge=None, harm=lambda t: 0, bank=None, **kw):
    return rewards.RewardContext(mode, target or FakeTarget(), np.zeros((0, 384)) if bank is None else bank,
                                 FakeTokenizer(), judge if judge is not None else FakeJudge(),
                                 harm_verify_at=20, harm=harm, embed=fake_embed, **kw)


def test_clean():
    assert rewards.clean('  web_search: "placeholder a"\nsecond line', "web_search") == "placeholder a"
    assert rewards.clean("`placeholder b`", "plan") == "placeholder b"
    assert rewards.clean("   \n", "plan") == ""


def test_reward_formula_and_judge():
    ctx = make_ctx()
    [s] = rewards.score_samples(ctx, [SEED], ["placeholder sample"])
    assert s.valid is True and s.harmless and not s.overlong
    assert s.miss == pytest.approx(0.8)
    assert 0.0 <= s.novelty <= 1.0 and s.novelty == pytest.approx(1.0)  # empty bank
    assert s.reward == pytest.approx(0.8 + 0.2 * s.novelty)


def test_miss_per_mode():
    target = FakeTarget(default=(0.3, 90))
    [h] = rewards.score_samples(make_ctx("H", target), [SEED], ["placeholder sample"])
    [lk] = rewards.score_samples(make_ctx("L", target), [SEED], ["placeholder sample"])
    assert h.miss == pytest.approx(0.3)
    assert lk.miss == pytest.approx(0.9)
    assert rewards.miss_of("L", 0.1, 20) == pytest.approx(0.9)


def test_harmless_filter_stops_before_target_and_judge():
    target, judge = FakeTarget(), FakeJudge()
    ctx = make_ctx(target=target, judge=judge, harm=lambda t: {"placeholder a": 20, "placeholder b": None}.get(t, 5))
    out = rewards.score_samples(ctx, [SEED] * 3, ["placeholder a", "placeholder b", "placeholder c"])
    assert [s.harmless for s in out] == [False, False, True]
    assert [s.reward for s in out[:2]] == [0.0, 0.0]
    assert len(target.calls) == 1 and len(judge.calls) == 1
    assert ctx.harmful_filtered == 2


def test_empty_completion_scores_zero():
    judge = FakeJudge()
    [s] = rewards.score_samples(make_ctx(judge=judge), [SEED], ["  "])
    assert s.empty and s.reward == 0.0 and not judge.calls


def test_judge_skipped_below_min_miss():
    judge = FakeJudge()
    ctx = make_ctx(target=FakeTarget(default=(0.29, 0)), judge=judge)
    [s] = rewards.score_samples(ctx, [SEED], ["placeholder sample"])
    assert s.valid is None and s.reward == 0.0 and not judge.calls


def test_judge_error_counts_as_invalid():
    [s] = rewards.score_samples(make_ctx(judge=FakeJudge(answer=None)), [SEED], ["placeholder sample"])
    assert s.valid is False and s.reward == 0.0


def test_overlong_by_chars_and_tokens():
    long_chars = "x" * 1001
    long_tokens = " ".join(["word"] * 130)
    out = rewards.score_samples(make_ctx(judge=FakeJudge(False)), [SEED] * 2, [long_chars, long_tokens])
    assert [s.overlong for s in out] == [1, 1]
    assert [s.reward for s in out] == [pytest.approx(-0.5), pytest.approx(-0.5)]


def test_novelty_against_bank_and_accepted():
    bank = fake_embed(["placeholder seen"])
    ctx = make_ctx(bank=bank)
    [seen] = rewards.score_samples(ctx, [SEED], ["placeholder seen"])
    assert seen.novelty == pytest.approx(0.0, abs=1e-5)
    [new] = rewards.score_samples(ctx, [SEED], ["placeholder new"])
    assert new.novelty > 0.5
    ctx.accept(new)
    [again] = rewards.score_samples(ctx, [SEED], ["placeholder new"])
    assert again.novelty == pytest.approx(0.0, abs=1e-5)


def test_without_judge_then_apply():
    judge = FakeJudge()
    ctx = make_ctx(judge=judge)
    [s] = rewards.score_samples(ctx, [SEED], ["placeholder sample"], judge=False)
    assert s.valid is None and not judge.calls
    rewards.apply_judge(ctx, SEED, "placeholder sample", s)
    assert s.valid and s.reward > 0


def test_scored_holds_no_text():
    [s] = rewards.score_samples(make_ctx(), [SEED], ["placeholder sample"])
    assert not any(isinstance(v, str) for v in vars(s).values())


def test_threshold_read_from_context_defender(monkeypatch):
    from app.security import context_defender

    monkeypatch.setattr(context_defender, "get_model", lambda: object())
    monkeypatch.setattr(context_defender, "current_thresholds", lambda: {"harm_verify_at": 20})
    ctx = rewards.RewardContext("H", FakeTarget(), np.zeros((0, 384)), FakeTokenizer(), FakeJudge(),
                                harm=lambda t: 0, embed=fake_embed)
    assert ctx.harm_verify_at == 20 and ctx.record()["harm_verify_at"] == 20
    monkeypatch.setattr(context_defender, "get_model", lambda: None)
    with pytest.raises(SystemExit):
        rewards.harm_verify_at()
