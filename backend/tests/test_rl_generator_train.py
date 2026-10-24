"""GRPO trainer logic and adapter sampling (rl-generator SEC-209, SEC-210, SEC-216).

Offline: resume, stop rules, the valid-rate guard, checkpoint rotation and best_step run against a
fake callback state and a fake judge, without trl; sample.py's step choice and held-out checks
with fakes. The tests that need trl are skipped when it isn't installed.
"""
import functools
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from rl_generator import rewards, rows, sample, seeds, train  # noqa: E402
from rl_generator import train_logic as tl  # noqa: E402

MARK = "zqxmarker"


def fake_embed(texts):
    out = []
    for t in texts:
        v = np.random.default_rng(int(hashlib.sha256(t.encode()).hexdigest(), 16) % 2**32).normal(size=384)
        out.append(v / np.linalg.norm(v))
    return np.asarray(out, dtype=np.float32)


class FakeTarget:
    version = "ac-test"
    tokenizer = staticmethod(lambda a, b: {"input_ids": [0] * 10})

    def score_pair(self, intent, call_text):
        return (0.2 if intent.endswith("0") else 0.9, 0)   # mode H: miss = p


class FakeJudge:
    model = "judge-model"

    def __init__(self, default=True, stop_after=None):
        self.default, self.stop_after = default, stop_after
        self.calls = self.errors = self.steps = self.seeds = 0

    def __call__(self, intent, call, action=""):
        self.calls += 1
        return self.default

    def begin_step(self):
        self.steps += 1

    def end_seed(self):
        self.seeds += 1

    def spent(self):
        return 0.01 * self.calls

    def error_stop(self):
        return False

    def must_stop(self, next_cost=0.0):
        return self.stop_after is not None and self.steps >= self.stop_after

    def report(self):
        return {"model": self.model, "calls": self.calls}


def seed(i, part="train"):
    return {"seed_id": f"seed{i:03d}", "part": part, "intent": f"{MARK} intent {i}", "call": "placeholder",
            "action": "web_search", "origin": "asq:query", "groups": [f"g{i}"], "sources": ["asq"]}


def make_ctx(judge, mode="H"):
    return rewards.RewardContext(mode, FakeTarget(), np.zeros((0, 384)), FakeTarget.tokenizer, judge,
                                 harm_verify_at=20, harm=lambda t: 90 if t.endswith("bad") else 0, embed=fake_embed)


def ckpts(tmp_path, *steps):
    for s in steps:
        tl.checkpoint_path(tmp_path, s).mkdir(parents=True, exist_ok=True)


def hist(n, valid, miss_start=0.5, miss_step=0.0, judged=10):
    return [{"step": i + 1, "judged": judged, "valid": round(valid * judged), "miss_mean": miss_start + i * miss_step}
            for i in range(n)]


# --- resume ---------------------------------------------------------------------------------

def test_resume_from_newest_checkpoint(tmp_path):
    assert tl.latest_checkpoint(tmp_path / "missing") is None and tl.latest_checkpoint(tmp_path) is None
    ckpts(tmp_path, 50, 100, 150)
    (tmp_path / "checkpoint-x").mkdir()
    (tmp_path / "checkpoint-200").write_text("not a dir")
    assert tl.checkpoint_steps(tmp_path) == [50, 100, 150]
    assert tl.latest_checkpoint(tmp_path) == tmp_path / "checkpoint-150"


def test_resume_drops_metrics_after_the_checkpoint(tmp_path):
    (tmp_path / tl.METRICS).write_text("".join(json.dumps({"step": s, "reward_mean": 0.1}) + "\n" for s in range(1, 8)))
    kept = tl.load_metrics(tmp_path, 5)
    assert [r["step"] for r in kept] == [1, 2, 3, 4, 5]
    assert len((tmp_path / tl.METRICS).read_text().splitlines()) == 5
    assert tl.load_metrics(tmp_path, None) == [] and (tmp_path / tl.METRICS).read_text() == ""


# --- the guard and best_step ----------------------------------------------------------------

def test_guard_fires_only_on_low_valid_and_rising_miss():
    assert tl.guard_fires(hist(200, 0.7, miss_step=0.001))
    assert not tl.guard_fires(hist(200, 0.7, miss_step=0.0))        # miss flat
    assert not tl.guard_fires(hist(200, 0.7, miss_step=-0.001))     # miss falling
    assert not tl.guard_fires(hist(200, 0.9, miss_step=0.001))      # valid rate fine
    assert not tl.guard_fires(hist(199, 0.5, miss_step=0.01))       # window not full
    assert not tl.guard_fires(hist(200, 0.7, miss_step=0.001, judged=0))
    # only the last 200 steps count
    assert not tl.guard_fires(hist(100, 0.1, miss_step=0.01) + hist(200, 0.95, miss_step=0.001))


def test_best_step_rules():
    assert tl.best_step([50, 100, 150], None) == 150
    assert tl.best_step([], None) is None
    assert tl.best_step([250, 300, 350, 400, 450], 450) == 250
    assert tl.best_step([50, 100, 150], 150) is None                 # nothing before the window


def test_rotation_keeps_the_guards_fallback(tmp_path):
    saved = []
    for step in range(50, 651, 50):
        saved.append(step)
        saved = [s for s in saved if s not in tl.to_delete(saved, step)]
        anchor = tl.guard_anchor(saved, step)
        if step >= 250:
            assert anchor == step - 200                              # still on disk for the guard
        assert len(saved) <= tl.SAVE_TOTAL_LIMIT + 1
    assert saved == [450, 500, 550, 600, 650]


def test_controller_guard_stops_and_records_best_step(tmp_path):
    ctl = tl.Controller(tmp_path, FakeJudge(), history=hist(199, 0.6, miss_step=0.001), window=200, log=lambda s: None)
    ckpts(tmp_path, 50, 100, 150, 200)
    ctl.record({"judged": 10, "valid": 6, "miss_mean": 0.9})
    assert ctl.on_step_end(201) and ctl.stopped == "valid-rate guard" and ctl.fired_at == 201
    out = ctl.finish(201)
    assert out["best_step"] is None                                  # no checkpoint at or before step 1
    ckpts(tmp_path, 1)
    assert ctl.finish(201)["best_step"] == 1
    assert json.loads((tmp_path / tl.BEST_STEP).read_text())["best_step"] == 1


def test_controller_judge_stop_and_metrics(tmp_path):
    lines = []
    judge = FakeJudge(stop_after=2)
    ctl = tl.Controller(tmp_path, judge, log=lines.append)
    for step in (1, 2):
        judge.begin_step()
        ctl.record(tl.step_metrics([rewards.Scored(reward=0.5, miss=0.6, harmless=True, valid=True)], 0, 1, 0, 0.01))
        stop = ctl.on_step_end(step)
    assert stop and ctl.stopped == "judge stop rule"
    ckpts(tmp_path, 2)
    assert ctl.finish(2) == {"best_step": 2, "last_step": 2, "stopped": "judge stop rule", "guard_fired_at": None,
                             "checkpoints": [2]}
    logged = [json.loads(line) for line in (tmp_path / tl.METRICS).read_text().splitlines()]
    assert [m["step"] for m in logged] == [1, 2]
    for m in logged:
        assert set(tl.METRIC_KEYS) <= set(m)
        assert all(isinstance(v, (int, float)) or v is None for v in m.values())


def test_callback_sets_stop_and_save(tmp_path):
    pytest.importorskip("transformers")
    ctl = tl.Controller(tmp_path, FakeJudge(stop_after=0), log=lambda s: None)
    cb = tl.make_callback(ctl)
    control = SimpleNamespace(should_training_stop=False, should_save=False)
    cb.on_step_end(None, SimpleNamespace(global_step=3), control)
    assert control.should_training_stop and control.should_save
    ckpts(tmp_path, 50, 100, 150, 200, 250, 300)
    cb.on_save(None, SimpleNamespace(global_step=300), control)
    assert tl.checkpoint_steps(tmp_path) == [100, 150, 200, 250, 300]


# --- the reward function --------------------------------------------------------------------

def test_reward_fn_maps_completions_to_seeds(tmp_path):
    judge = FakeJudge()
    ctx = make_ctx(judge)
    ctl = tl.Controller(tmp_path, judge, log=lambda s: None)
    by_id = {s["seed_id"]: s for s in (seed(0), seed(1))}
    fn = train.make_reward_fn(ctx, by_id, ctl)
    ids = ["seed000"] * 3 + ["seed001"] * 3
    completions = [[{"role": "assistant", "content": f"call {i}"}] for i in range(5)] + [[{"role": "assistant",
                                                                                            "content": "call bad"}]]
    out = fn(prompts=[[]] * 6, completions=completions, seed_id=ids, trainer_state=None)
    assert len(out) == 6 and out[0] == out[1] == out[2] == 0.0       # seed000: miss 0.2, not judged
    assert all(r > 0.9 for r in out[3:5]) and out[5] == 0.0          # seed001: judged valid; the last harmful
    assert judge.steps == 1 and judge.seeds == 2 and judge.calls == 2 and len(ctx.accepted) == 2
    m = ctl.pending
    assert m["samples"] == 6 and m["judged"] == 2 and m["valid_rate"] == 1.0 and m["harmful_filtered"] == 1
    assert MARK not in json.dumps(m)
    assert fn(prompts=["p"], completions=["call 9"], seed_id=["seed001"])[0] > 0.9   # plain-text completions


def test_prompts_from_train_part_shuffled():
    rows_ = train.prompt_rows([seed(i) for i in range(10)], "H")
    assert sorted(r["seed_id"] for r in rows_) == [f"seed{i:03d}" for i in range(10)]
    assert [r["seed_id"] for r in rows_] == [r["seed_id"] for r in train.prompt_rows([seed(i) for i in range(10)], "H")]
    assert rows_[0]["prompt"][0]["role"] == "system" and rows_[0]["prompt"][-1]["role"] == "user"


def test_settings_and_args():
    nb = ["--mode", "h", "--base", "B", "--seeds", "s.jsonl", "--bank", "b.npy", "--target", "models/security/alignment",
          "--judge-model", "J", "--ckpt-dir", "/tmp/x", "--max-steps", "20", "--credit-left", "5"]
    args = train.parse_args(nb + ["--price-input", "1", "--price-output", "2", "--lam-div", "0.3", "--lam-len", "0.4",
                                  "--scorers-device", "cpu", "--seed", "3"])
    assert args.mode == "H" and args.max_steps == 20 and args.save_steps == 50 and args.guard_window == 200
    s = train.grpo_settings(args, "fp32")
    assert s["num_generations"] == 8 and s["per_device_train_batch_size"] == 16 and s["gradient_accumulation_steps"] == 4
    assert (s["max_completion_length"], s["temperature"], s["beta"], s["learning_rate"]) == (96, 0.9, 0.04, 1e-5)
    assert s["warmup_steps"] == 20 and s["lr_scheduler_type"] == "cosine" and s["save_steps"] == 50
    assert s["use_cpu"] and not s["bf16"] and not s["fp16"] and not s["log_completions"]
    assert train.precision(True, True) == "bf16" and train.precision(True, False) == "fp16"
    assert train.precision(False, False) == "fp32"
    for bad in (["--mode", "A"] + nb[2:], nb[:-2]):
        with pytest.raises(SystemExit):
            train.parse_args(bad)


def test_grpo_config_accepts_settings(tmp_path):
    pytest.importorskip("trl")
    from trl import GRPOConfig

    args = train.parse_args(["--mode", "H", "--judge-model", "J", "--ckpt-dir", str(tmp_path), "--max-steps", "2",
                             "--credit-left", "1"])
    cfg = GRPOConfig(**train.grpo_settings(args, "fp32"))
    assert cfg.generation_batch_size == 64 and cfg.steps_per_generation == 4 and cfg.num_generations == 8


# --- sample.py ------------------------------------------------------------------------------

def sample_args(tmp_path, *extra):
    return sample.parse_args(["--mode", "H", "--judge-model", "J", "--ckpt-dir", str(tmp_path / "ck"), "--n", "3",
                              "--credit-left", "5", "--seeds", str(tmp_path / "seeds_h.jsonl"),
                              "--bank", str(tmp_path / "bank.npy"), "--out-dir", str(tmp_path / "gen"), *extra])


def test_sample_args_heldout_rules(tmp_path):
    with pytest.raises(SystemExit):
        sample_args(tmp_path, "--heldout", "--sample-seed", "11")    # no --step
    with pytest.raises(SystemExit):
        sample_args(tmp_path, "--heldout", "--step", "100")          # default sample seed
    assert sample_args(tmp_path, "--heldout", "--step", "100", "--sample-seed", "11").heldout
    with pytest.raises(SystemExit):
        sample.parse_args(["--mode", "L2", "--judge-model", "J", "--ckpt-dir", "x", "--credit-left", "1"])


def test_sample_step_choice(tmp_path):
    ck = tmp_path / "ck"
    with pytest.raises(SystemExit):
        sample.choose_step(sample_args(tmp_path))                    # no best_step.json
    ckpts(ck, 50, 100)
    tl.write_json(ck / tl.BEST_STEP, {"best_step": None})
    with pytest.raises(SystemExit):
        sample.choose_step(sample_args(tmp_path))
    tl.write_json(ck / tl.BEST_STEP, {"best_step": 100})
    assert sample.choose_step(sample_args(tmp_path)) == 100
    assert sample.choose_step(sample_args(tmp_path, "--step", "50")) == 50
    with pytest.raises(SystemExit):
        sample.choose_step(sample_args(tmp_path, "--step", "75"))   # no such checkpoint
    tl.record_sampled(ck, 100, heldout=False)
    held = ("--heldout", "--sample-seed", "11", "--step")
    with pytest.raises(SystemExit):
        sample.choose_step(sample_args(tmp_path, *held, "100"))      # used for a training batch
    assert sample.choose_step(sample_args(tmp_path, *held, "50")) == 50
    tl.record_sampled(ck, 50, heldout=True)
    with pytest.raises(SystemExit):
        sample.choose_step(sample_args(tmp_path, "--step", "50"))   # used for a held-out batch
    assert tl.read_sampled(ck) == {"rl": [100], "heldout": [50]}


@pytest.fixture
def offline(tmp_path, monkeypatch):
    seeds.write(tmp_path / "seeds_h.jsonl", [seed(i, "train") for i in range(3)] +
                [seed(i, "sample") for i in range(3, 9)], "hash1", "H")
    np.save(tmp_path / "bank.npy", np.zeros((0, 384), dtype=np.float16))
    (tmp_path / "bank.json").write_text(json.dumps({"data_hash": "hash1"}))
    monkeypatch.setattr(sample.seeds, "current_hash", lambda: "hash1")
    monkeypatch.setattr(sample, "load_target", lambda p: FakeTarget())
    monkeypatch.setattr(sample.rewards, "RewardContext", functools.partial(
        rewards.RewardContext, harm_verify_at=20, harm=lambda t: 0, embed=fake_embed))
    ckpts(tmp_path / "ck", 50, 100)
    tl.write_json(tmp_path / "ck" / tl.BEST_STEP, {"best_step": 100})
    return tmp_path


def gen(prompts):
    assert all(MARK in p[-1]["content"] for p in prompts)
    return [[f"call {k}" for k in range(8)] for _ in prompts]


def test_sample_writes_rl_rows_from_sample_part(offline, capsys):
    seen = []
    report = sample.run(sample_args(offline), judge_factory=lambda m, mode: FakeJudge(), generate=gen, log=seen.append)
    assert report["rows"] == 3 and report["adapter_step"] == 100 and not report["heldout"]
    written = rows.read_rows(offline / "gen" / "rl_gen_h_rl.jsonl")
    assert {r["meta"]["seed_id"] for r in written} <= {f"seed{i:03d}" for i in range(3, 9)}
    assert all(r["origin"] == "rl-gen-h:rl" and r["meta"]["adapter_step"] == 100 for r in written)
    assert written[0]["meta"]["batch"].startswith("h-rl-")
    assert tl.read_sampled(offline / "ck") == {"rl": [100], "heldout": []}
    assert MARK not in json.dumps(report) and MARK not in capsys.readouterr().out


def test_sample_heldout_batch(offline):
    sample.run(sample_args(offline), judge_factory=lambda m, mode: FakeJudge(), generate=gen)
    held = ("--heldout", "--step", "50", "--sample-seed", "11")
    with pytest.raises(SystemExit):                                   # same seeds as the rl batch
        sample.run(sample_args(offline, *held), judge_factory=lambda m, mode: FakeJudge(), generate=gen)
    seeds.write(offline / "seeds_h.jsonl", [seed(i, "sample") for i in range(20, 24)], "hash1", "H")
    report = sample.run(sample_args(offline, *held), judge_factory=lambda m, mode: FakeJudge(), generate=gen)
    assert report["heldout"] and report["adapter_step"] == 50 and report["rows"] == 3
    written = rows.read_rows(offline / "gen" / "rl_gen_h_heldout.jsonl")
    assert len(written) == 3 and all(r["meta"]["batch"].startswith("h-heldout-") for r in written)
    assert rows.parse_batch(written[0]["meta"]["batch"])["heldout"]
