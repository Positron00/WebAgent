"""Only modes H and L, everywhere; benign prompts (rl-generator SEC-202).

Offline.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from rl_generator import judge, prompted, prompts, rewards, rows, seeds  # noqa: E402

BAD = ("A", "attack", "", "HL", "x")
SEED = {"seed_id": "s", "intent": "placeholder", "action": "plan", "call": "placeholder", "groups": ["g"],
        "sources": ["asq"]}


@pytest.mark.parametrize("mode", BAD)
def test_other_modes_refused(mode, tmp_path):
    calls = [
        lambda: seeds.select([], mode, 10),
        lambda: seeds.seed_path(mode),
        lambda: seeds.write(tmp_path / "s.jsonl", [], "h", mode),
        lambda: seeds.main(["--mode", mode, "--data-dir", str(tmp_path)]),
        lambda: prompts.generator_messages(mode, SEED),
        lambda: prompts.judge_prompt(mode, "i", "plan", "c"),
        lambda: judge.Judge("m", mode, tmp_path / "c.sqlite", client=object(), default_model="d",
                            price_in=1, price_out=1),
        lambda: rewards.RewardContext(mode, None, np.zeros((0, 384)), None, None, harm_verify_at=20,
                                      embed=lambda t: t),
        lambda: rows.batch_id(mode, "prompted", "20261024", 1),
        lambda: rows.batch_path(mode, "rl"),
        lambda: rows.make_row(SEED, "c", mode, "prompted", batch="b", generator="g", adapter_step=None,
                              target="t", judge="j", prompt="p1", miss=0, novelty=0, data_hash="h"),
        lambda: prompted.parse_args(["--mode", mode, "--judge-model", "m", "--estimate", "5"]),
    ]
    for call in calls:
        with pytest.raises(SystemExit):
            call()


def test_unknown_kind_refused():
    with pytest.raises(SystemExit):
        rows.batch_path("H", "attack")
    with pytest.raises(SystemExit):
        rows.batch_path("H", "prompted", heldout=True)


@pytest.mark.parametrize("mode", ["H", "L"])
def test_prompts_ask_for_nothing_risky(mode):
    text = " ".join(m["content"] for m in prompts.generator_messages(mode, SEED)).lower()
    text += " " + prompts.judge_prompt(mode, "placeholder", "plan", "placeholder").lower()
    for word in ("attack", "inject", "jailbreak", "exploit", "malicious", "harmful", "dangerous", "bypass",
                 "deceptive", "risky call", "unsafe"):
        assert word not in text
    assert "harmless" in text
