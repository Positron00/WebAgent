"""Generated sources in the alignment data rules (rl-generator SEC-200, SEC-213 rule, SEC-202).

Offline; neutral placeholder rows only.
"""
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import alignment_data as data  # noqa: E402
import alignment_training as training  # noqa: E402
import rl_generator  # noqa: E402


def gen_row(mode="H", seed_sources=("webagent-gen",), split="train", kind="prompted"):
    src = rl_generator.SOURCE[mode]
    return {"intent": "placeholder intent", "call": "placeholder call", "action": "web_search",
            "labels": dict(rl_generator.LABELS[mode]), "origin": f"{src}:{kind}", "role": "T",
            "split": split, "groups": ["g1"], "sources": [src, *seed_sources], "cluster": "gen-x"}


@pytest.mark.parametrize("mode", ["H", "L"])
def test_generated_row_is_valid_local_and_train_only(mode):
    row = gen_row(mode)
    assert data.schema_errors(row) == []
    assert data.rule_errors(row, committed=False) == []
    assert data.is_local(row)
    assert "gated or no-train row in a committed file" in data.rule_errors(row, committed=True)
    assert "generated row outside train" in data.rule_errors(gen_row(mode, split="test"), committed=False)


def test_generated_origins_are_neither_real_calls_nor_lookalikes():
    for mode in ("H", "L"):
        for kind in ("prompted", "rl"):
            row = gen_row(mode, kind=kind)
            assert not training.is_real_aligned(row)
            assert not training.is_lookalike(row)


def test_seed_licence_carries_over():
    row = gen_row("H", seed_sources=("asq",))
    assert set(row["sources"]) & data.SHARE_ALIKE
    mixed = dict(row, sources=row["sources"] + ["agentdam"])
    assert "CC BY-SA and CC BY-NC text in one row" in data.rule_errors(mixed, committed=False)


def test_readme_has_the_generated_sources():
    readme = (data.SOURCES and (SCRIPTS.parent / "data" / "security" / "alignment" / "README.md")).read_text()
    for name in data.GENERATED:
        assert data.SOURCES[name][3] in readme


def test_only_two_modes():
    assert rl_generator.MODES == ("H", "L")
    assert rl_generator.check_mode("h") == "H"
    for bad in ("A", "attack", "", "HL"):
        with pytest.raises(SystemExit):
            rl_generator.check_mode(bad)
