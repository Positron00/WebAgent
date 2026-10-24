"""Calibration fold without generated rows, and --seed (rl-generator SEC-214, SEC-217). Offline, placeholder rows."""
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import train_alignment_model as tm  # noqa: E402


def _row(i, origin="webagent-gen:T", aligned="yes"):
    return {"intent": f"intent {i}", "call": f"call {i}", "action": "web_search",
            "labels": {"aligned": aligned, "risk": "none"}, "origin": origin, "role": "T", "split": "train"}


def _old_split(rows, share=0.1):
    """split_calibration as it was before SEC-214 (release 3.13.0)."""
    rng = random.Random(13)
    order = list(range(len(rows)))
    rng.shuffle(order)
    cut = int(len(rows) * share)
    return [rows[i] for i in order[cut:]], [rows[i] for i in order[:cut]]


def test_same_fold_and_order_as_before_without_generated_rows():
    for n in (7, 100, 1234, 34538):
        rows = [_row(i) for i in range(n)]
        assert tm.split_calibration(rows) == _old_split(rows)
    assert tm.SEED == 13


def test_no_generated_row_in_the_fold_and_size_from_real_rows():
    real = [_row(i) for i in range(500)]
    gen = [_row(1000 + i, origin=f"rl-gen-{m}:rl", aligned="no") for i, m in enumerate("hl" * 150)]
    rows = real[:250] + gen + real[250:]
    train, calib = tm.split_calibration(rows)
    assert len(calib) == 50  # 10% of the 500 real rows, whatever the generated share
    assert not any(r["origin"].startswith("rl-gen-") for r in calib)
    assert sum(r["origin"].startswith("rl-gen-") for r in train) == 300
    assert len(train) + len(calib) == len(rows)
    # The fold is the same rows as with no generated rows at all.
    assert calib == tm.split_calibration(real)[1]


def test_seed_changes_the_fold():
    rows = [_row(i) for i in range(400)]
    a, b = tm.split_calibration(rows, seed=13)[1], tm.split_calibration(rows, seed=17)[1]
    assert len(a) == len(b) == 40 and a != b
    assert tm.split_calibration(rows, seed=17) == tm.split_calibration(rows, seed=17)


def test_metadata_records_seed_and_variant(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "train.jsonl").write_text("{}\n", encoding="utf-8")
    (data / "variant.json").write_text(json.dumps({"kind": "rl", "share": 0.05}), encoding="utf-8")
    rows = [_row(i) for i in range(60)] + [_row(100 + i, origin="injecagent:T", aligned="no") for i in range(40)]
    rng = np.random.default_rng(1)
    p_raw = np.concatenate([rng.uniform(0.6, 1, 60), rng.uniform(0, 0.3, 40)])
    r_raw = rng.integers(0, 10, 100).astype(float) / 100
    out = tmp_path / "model"
    out.mkdir()
    tm.finish(out, "siamese", "placeholder", rows, p_raw, r_raw, {"trained_rows": 5}, data, seed=17)
    meta = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    assert meta["train_seed"] == 17 and meta["variant"] == {"kind": "rl", "share": 0.05}
    assert meta["data_hash"] == tm.data_hash(data)
    (data / "variant.json").unlink()
    tm.finish(out, "siamese", "placeholder", rows, p_raw, r_raw, {"trained_rows": 5}, data)
    meta = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    assert meta["train_seed"] == 13 and "variant" not in meta
