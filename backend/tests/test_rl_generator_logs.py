"""No sample, seed or row text in what the generator's scripts print, log or report (rl-generator SNFR-20).

Offline: fakes with marker text that must never appear in output.
"""
import hashlib
import json
import logging
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from rl_generator import prompted, review, rewards, rows  # noqa: E402

MARK_INTENT, MARK_CALL = "zqxintentmarker", "zqxcallmarker"


def fake_embed(texts):
    out = []
    for t in texts:
        v = np.random.default_rng(int(hashlib.sha256(t.encode()).hexdigest(), 16) % 2**32).normal(size=384)
        out.append(v / np.linalg.norm(v))
    return np.asarray(out, dtype=np.float32)


class Target:
    version = "ac-test"

    def score_pair(self, intent, call_text):
        return (0.1, 0)   # mode L: miss 0.9


class Tok:
    def __call__(self, a, b):
        return {"input_ids": [0] * 10}


class Judge:
    model = "judge-model"

    def __init__(self):
        self.n = 0

    def __call__(self, intent, call, action=""):
        self.n += 1
        return self.n % 2 == 0

    def spent(self):
        return 0.0

    def error_stop(self):
        return False

    def end_seed(self):
        pass

    def report(self):
        return {"model": self.model, "calls": self.n}


def test_prompted_run_reports_counts_only(tmp_path, capsys, caplog):
    caplog.set_level(logging.DEBUG)
    ctx = rewards.RewardContext("L", Target(), np.zeros((0, 384)), Tok(), Judge(), harm_verify_at=20,
                                harm=lambda t: 50 if t.endswith("7") else 0, embed=fake_embed)
    seeds = [{"seed_id": f"s{i}", "intent": f"{MARK_INTENT} {i}", "call": MARK_CALL, "action": "plan",
              "origin": "asq:query", "groups": ["g"], "sources": ["asq"]} for i in range(6)]

    def generate(prompts):
        return [[f"{MARK_CALL} {j}" for j in range(8)] for _ in prompts]

    prov = prompted.Provenance("prompted", "g", "ac-test", "h", judge="judge-model", batch="l-prompted-20261024-1",
                               out=tmp_path / "rl_gen_l_prompted.jsonl")
    report = prompted.select_and_write(ctx, seeds, generate, prov, n=3, credit_left=10.0)
    print(json.dumps(report))
    printed = capsys.readouterr()
    for text in (printed.out, printed.err, caplog.text, json.dumps(report), json.dumps(ctx.record())):
        assert MARK_INTENT not in text and MARK_CALL not in text
    assert report["rows"] == 3 and report["harmful_filtered"] > 0
    assert MARK_CALL in prov.out.read_text()   # the batch file is the only place text goes


def test_review_prints_counts_only(tmp_path, capsys):
    gen, rev = tmp_path / "gen", tmp_path / "rev"
    seed = {"intent": MARK_INTENT, "groups": ["g"], "sources": ["asq"], "action": "plan", "origin": "asq:query"}
    batch = "h-prompted-20261024-1"
    rows.append(rows.batch_path("H", "prompted", out_dir=gen), [
        rows.make_row({**seed, "seed_id": f"s{i}"}, f"{MARK_CALL} {i}", "H", "prompted", batch=batch, generator="g",
                      adapter_step=None, target="t", judge="j", prompt="p1", miss=0.5, novelty=0.5, data_hash="h")
        for i in range(10)])
    review.main(["--make", batch, "--generated-dir", str(gen), "--review-dir", str(rev)])
    sheet = rev / f"{batch}.csv"
    sheet.write_text(sheet.read_text().replace(",\n", ",y\n").replace(",\r\n", ",y\r\n"))
    review.main(["--score", batch, "--generated-dir", str(gen), "--review-dir", str(rev)])
    out = capsys.readouterr().out
    assert MARK_INTENT not in out and MARK_CALL not in out
    assert MARK_CALL not in (gen / "rl_review.json").read_text()
