"""Seed selection, parts, seed files and the novelty bank (rl-generator SEC-201, SEC-206).

Offline; neutral placeholder rows and a fake embedder only.
"""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from rl_generator import seeds  # noqa: E402

ORIGIN_ACTION = {"asq:query": "web_search", "webagent-gen:call": "plan", "glaive:call": "function",
                 "toolace:call": "function", "saroku:safe": "internal_retrieve"}


def row(i, origin="asq:query", action=None, labels=("yes", "none"), role="T", sources=None):
    ds = origin.split(":")[0]
    return {"intent": f"placeholder intent {i}", "call": f"placeholder call {i}",
            "action": action or ORIGIN_ACTION.get(origin, "web_search"),
            "labels": {"aligned": labels[0], "risk": labels[1]}, "origin": origin, "role": role,
            "split": "train", "groups": [f"g{i}"], "sources": sources or [ds], "cluster": ""}


def fake_embed(texts):
    out = []
    for t in texts:
        v = np.random.default_rng(int(hashlib.sha256(t.encode()).hexdigest(), 16) % 2**32).normal(size=384)
        out.append(v / np.linalg.norm(v))
    return np.asarray(out, dtype=np.float32)


def pool():
    rows = [row(i) for i in range(300)]                                    # asq-heavy
    rows += [row(1000 + i, "webagent-gen:call") for i in range(40)]
    rows += [row(2000 + i, "glaive:call") for i in range(40)]
    rows += [row(3000 + i, "toolace:call") for i in range(40)]
    rows += [row(4000 + i, "saroku:safe") for i in range(40)]
    return rows


def test_only_eligible_rows_are_seeds():
    bad = [row(1, labels=("no", "none")), row(2, labels=("yes", "low")), row(3, role="E", sources=["agentdojo"]),
           row(4, "webagent-gen:code", action="execute_code"), row(5, "xlam:call", sources=["xlam"]),
           row(6, "notinject:lookalike", sources=["notinject"]), {**row(7), "split": "test"}]
    assert seeds.select(bad, "H", 10) == []
    assert len(seeds.select(bad + [row(8)], "H", 10)) == 1


def test_stratified_by_origin_and_action():
    chosen = seeds.select(pool(), "H", 90)
    assert len(chosen) == 90
    by_origin = Counter(s["origin"] for s in chosen)
    assert max(by_origin.values()) <= 30
    by_action = Counter(s["action"] for s in chosen)
    for action in ("web_search", "plan", "function", "internal_retrieve"):
        assert by_action[action] >= 9


def test_parts_are_60_40_and_deterministic():
    a = seeds.select(pool(), "H", 50)
    assert Counter(s["part"] for s in a) == {"train": 30, "sample": 20}
    assert a == seeds.select(pool(), "H", 50)
    assert [s["seed_id"] for s in a] != [s["seed_id"] for s in seeds.select(pool(), "L", 50)]
    for s in a:
        assert set(s) >= {"seed_id", "part", "intent", "call", "action", "origin", "groups", "sources"}


def test_exclude_and_unique_ids():
    first = seeds.select(pool(), "H", 40)
    second = seeds.select(pool(), "H", 40, exclude={s["seed_id"] for s in first})
    assert not {s["seed_id"] for s in first} & {s["seed_id"] for s in second}
    assert len({s["seed_id"] for s in first}) == 40


def test_seed_file_round_trip_and_stale_hash(tmp_path):
    chosen = seeds.select(pool(), "L", 20)
    path = seeds.seed_path("L", tmp_path)
    assert path.name == "seeds_l.jsonl"
    seeds.write(path, chosen, "abc123", "L")
    assert seeds.read_header(path) == {"data_hash": "abc123", "mode": "L", "n": 20}
    assert seeds.read(path, "abc123") == chosen
    assert all(s["part"] == "sample" for s in seeds.read(path, "abc123", part="sample"))
    with pytest.raises(SystemExit):
        seeds.read(path, "other")


def test_bank_has_committed_calls_only_and_checks_hash(tmp_path):
    rows = [row(1), row(2), row(3, "xlam:call", sources=["xlam"])]
    out = tmp_path / "novelty_bank.npy"
    assert seeds.write_bank(rows, out, "h1", embed=fake_embed) == 2
    bank = seeds.read_bank(out, "h1")
    assert bank.shape == (2, 384) and bank.dtype == np.float16
    assert json.loads(out.with_suffix(".json").read_text())["data_hash"] == "h1"
    with pytest.raises(SystemExit):
        seeds.read_bank(out, "h2")


def test_cli_writes_a_seed_file(tmp_path, capsys):
    data = tmp_path / "data"
    data.mkdir()
    (data / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in pool()))
    out = tmp_path / "seeds_h.jsonl"
    assert seeds.main(["--mode", "h", "--n", "30", "--data-dir", str(data), "--out", str(out)]) == 0
    assert seeds.read_header(out)["n"] == 30
    assert "placeholder" not in capsys.readouterr().out
