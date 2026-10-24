"""variant.py: the base build plus reviewed generated rows (rl-generator SEC-213, SEC-216).

Offline: a fake embedder, placeholder rows, temporary folders standing in for LOCAL.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import alignment_data as data  # noqa: E402
from rl_generator import variant  # noqa: E402


def fake_embed(texts):
    out = []
    for t in texts:
        v = np.random.default_rng(int(hashlib.sha256(t.encode()).hexdigest(), 16) % 2**32).normal(size=64)
        out.append(v / np.linalg.norm(v))
    return np.asarray(out, dtype=np.float32).reshape(len(texts), 64)


def _row(i, split, action="web_search", aligned="yes", risk="none"):
    return {"intent": f"placeholder intent {i}", "call": f"placeholder call {i}", "action": action,
            "labels": {"aligned": aligned, "risk": risk}, "origin": "webagent-gen:T", "role": "T", "split": split,
            "groups": [f"g{i}"], "sources": ["webagent-gen"], "cluster": ""}


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


@pytest.fixture
def env(tmp_path):
    """A base build that passes check_data (every required cell), a LOCAL root and a generated folder."""
    base, local = tmp_path / "base", tmp_path / "local"
    train, i = [], 0
    for action, cells in data.REQUIRED_CELLS.items():
        for aligned, risk in cells:
            for _ in range(data.MIN_CELL):
                train.append(_row(i, "train", action, aligned, risk))
                i += 1
    _write(base / "train.jsonl", train)
    _write(base / "test.jsonl", [_row(5000 + k, "test") for k in range(20)])
    _write(base / "probe.jsonl", [_row(6000 + k, "probe") for k in range(5)])
    _write(base / "local" / "train.jsonl", [_row(7000 + k, "train") for k in range(10)])
    (base / "README.md").write_text("\n".join(t for *_, t in data.SOURCES.values()), encoding="utf-8")
    gen = local / "generated"
    gen.mkdir(parents=True)
    return {"base": base, "local": local, "gen": gen, "hash": variant.base_hash(base), "train": len(train) + 10}


def _gen(i, mode="H", kind="prompted", batch="h-prompted-20261024-1", data_hash="", group="g0", **change):
    row = {"intent": f"placeholder gen intent {i}", "call": f"placeholder gen call {i}", "action": "web_search",
           "labels": {"aligned": "no" if mode == "H" else "yes", "risk": "none"},
           "origin": f"rl-gen-{mode.lower()}:{kind}", "role": "T", "split": "train", "groups": [group],
           "sources": [f"rl-gen-{mode.lower()}", "webagent-gen"], "cluster": "",
           "meta": {"batch": batch, "seed_id": f"s{i}", "data_hash": data_hash}}
    row.update(change)
    return row


def _review(env, **verdicts):
    (env["gen"] / "rl_review.json").write_text(json.dumps(
        {b: {"n": 100, "user_no": 2, "verdict": v, "date": "2026-10-24"} for b, v in verdicts.items()}))


def _build(env, name, kind="prompted", share=0.05, modes=("H",)):
    return variant.build(env["base"], env["local"] / "variants" / name, kind, share, modes,
                         generated=env["gen"], local=env["local"], embed=fake_embed)


def _local_rows(env, name):
    return data.read_jsonl(env["local"] / "variants" / name / "local" / "train.jsonl")


def test_none_is_a_copy_of_the_base(env):
    info = _build(env, "none", "none", 0, ("H", "L"))
    out = env["local"] / "variants" / "none"
    for name in ("train.jsonl", "test.jsonl", "probe.jsonl", "local/train.jsonl"):
        assert (out / name).read_bytes() == (env["base"] / name).read_bytes()
    assert info["kept"] == {"H": 0, "L": 0} and info["base_data_hash"] == env["hash"]
    assert json.loads((out / "variant.json").read_text())["kind"] == "none"


def test_pass_batches_of_this_build_only_and_rows_finished(env):
    h = env["hash"]
    _write(env["gen"] / "rl_gen_h_prompted.jsonl",
           [_gen(i, data_hash=h) for i in range(3)] + [_gen(10 + i, batch="h-prompted-20261024-2", data_hash=h)
                                                       for i in range(2)]
           + [_gen(20, batch="h-prompted-20261024-3", data_hash="stale")])
    _review(env, **{"h-prompted-20261024-1": "pass", "h-prompted-20261024-2": "more", "h-prompted-20261024-3": "pass"})
    info = _build(env, "p5")
    assert info["kept"]["H"] == 3 and info["batches"]["H"] == ["h-prompted-20261024-1"]
    assert info["dropped"]["H"] == {"not passed": 2, "stale data hash": 1}
    added = _local_rows(env, "p5")[10:]
    assert len(added) == 3 and all("meta" not in r and r["split"] == "train" for r in added)
    assert [r["cluster"] for r in added] == ["gen-s0", "gen-s1", "gen-s2"]
    out = env["local"] / "variants" / "p5"
    assert (out / "train.jsonl").read_bytes() == (env["base"] / "train.jsonl").read_bytes()
    assert (out / "local" / "train.jsonl").read_bytes().startswith((env["base"] / "local" / "train.jsonl").read_bytes())
    assert data.check_data(out, fake_embed, env["base"] / "README.md") == []


def test_leakage_drops(env):
    h = env["hash"]
    rows = [_gen(0, data_hash=h, group="g5000"),  # a test group
            _gen(1, data_hash=h, intent="other placeholder", call="placeholder call 5003"),  # a test call
            _gen(2, data_hash=h, intent="placeholder intent 3", call="placeholder call 3"),  # a base row
            _gen(3, data_hash=h), _gen(3, data_hash=h), _gen(4, data_hash=h)]  # one earlier generated duplicate
    _write(env["gen"] / "rl_gen_h_prompted.jsonl", rows)
    _review(env, **{"h-prompted-20261024-1": "pass"})
    info = _build(env, "leak")
    assert info["dropped"]["H"] == {"seed groups not all in base train": 1, "near a test or probe call": 1,
                                    "duplicate": 2}
    assert info["kept"]["H"] == 2


def test_half_share_cap_per_mode_and_shortfall(env):
    h = env["hash"]
    cap = round(0.05 / 2 * env["train"])
    _write(env["gen"] / "rl_gen_h_rl.jsonl", [_gen(i, kind="rl", batch="h-rl-1", data_hash=h) for i in range(cap + 15)])
    _write(env["gen"] / "rl_gen_l_rl.jsonl",
           [_gen(500 + i, "L", "rl", batch="l-rl-1", data_hash=h) for i in range(4)])
    _review(env, **{"h-rl-1": "pass", "l-rl-1": "pass"})
    info = _build(env, "r5", "rl", 0.05, ("H", "L"))
    assert info["cap_per_mode"] == cap and info["real_train_rows"] == env["train"]
    assert info["kept"] == {"H": cap, "L": 4} and info["shortfall"] == {"H": 0, "L": cap - 4}
    again = _build(env, "r5-again", "rl", 0.05, ("H", "L"))
    assert _local_rows(env, "r5") == _local_rows(env, "r5-again") and again["kept"] == info["kept"]
    alone = _build(env, "r5-h", "rl", 0.05, ("H",))  # one mode alone still gets only its half
    assert alone["kept"] == {"H": cap}


def test_heldout_batches_refused(env):
    _write(env["gen"] / "rl_gen_h_rl.jsonl", [_gen(0, kind="rl", batch="h-heldout-20261101-1", data_hash=env["hash"])])
    _review(env, **{"h-heldout-20261101-1": "pass"})
    with pytest.raises(SystemExit, match="held-out"):
        _build(env, "r5", "rl", 0.05)
    assert variant.is_heldout("H-heldout-20261101-1") and not variant.is_heldout("h-rl-20261101-1")


def test_out_under_local_and_kind_share_pairs(env, tmp_path):
    with pytest.raises(SystemExit, match="under"):
        variant.build(env["base"], tmp_path / "elsewhere", "none", 0, ("H",), generated=env["gen"],
                      local=env["local"], embed=fake_embed)
    for kind, share in (("none", 0.05), ("rl", 0), ("prompted", 2)):
        with pytest.raises(SystemExit):
            _build(env, "bad", kind, share)
    with pytest.raises(SystemExit):
        _build(env, "bad", "rl", 0.05, ("A",))


def test_check_data_failure_removes_out(env, monkeypatch):
    monkeypatch.setattr(variant, "check_data", lambda *a: ["placeholder error (1)"])
    with pytest.raises(SystemExit, match="check_data"):
        _build(env, "none", "none", 0)
    assert not (env["local"] / "variants" / "none").exists()


def test_main_cli(env, monkeypatch, capsys):
    import build_alignment_dataset

    monkeypatch.setattr(build_alignment_dataset, "Embedder", lambda: fake_embed)
    assert variant.main(["--base", str(env["base"]), "--kind", "none", "--share", "0", "--modes", "H",
                         "--out", str(env["local"] / "variants" / "cli"), "--generated", str(env["gen"]),
                         "--local", str(env["local"])]) == 0
    printed = capsys.readouterr().out
    assert json.loads(printed)["kind"] == "none" and "placeholder" not in printed


def test_batches_option_selects_a_rounds_batches(env):
    rows = [_gen(i, batch="h-prompted-20261024-1", data_hash=env["hash"]) for i in range(3)]
    rows += [_gen(10 + i, batch="h-prompted-20261024-2", data_hash=env["hash"]) for i in range(3)]
    _write(env["gen"] / "rl_gen_h_prompted.jsonl", rows)
    _review(env, **{"h-prompted-20261024-1": "pass", "h-prompted-20261024-2": "pass"})
    info = variant.build(env["base"], env["local"] / "variants" / "round", "prompted", 0.15, ["H"],
                         generated=env["gen"], local=env["local"], embed=fake_embed,
                         batches=["h-prompted-20261024-2"])
    assert info["batches"]["H"] == ["h-prompted-20261024-2"] and info["kept"]["H"] == 3
    assert info["dropped"]["H"] == {"batch not selected": 3}
    kept = {r["call"] for r in _local_rows(env, "round") if r["origin"].startswith("rl-gen")}
    assert kept == {f"placeholder gen call {10 + i}" for i in range(3)}
    with pytest.raises(SystemExit):
        variant.build(env["base"], env["local"] / "variants" / "bad", "rl", 0.15, ["H"], generated=env["gen"],
                      local=env["local"], embed=fake_embed, batches=["h-heldout-20261024-1"])
