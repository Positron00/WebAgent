#!/usr/bin/env python3
"""The alignment classifier's gate (checkpoint2-intent SEC-124, SEC-125, SNFR-11; task B6).

Scores `test` (gated) and `probe` (reported) rows through the runtime loader
(`app.security.alignment_model.load(..., require_gate=False)`), checks each target, benchmarks one
pair at a time on one thread, and compares candidates: the cross-encoder ships only if its
off-intent recall beats the siamese baseline's (each at its own thresholds, chosen under the same
false-positive rule); otherwise the baseline ships. Writes `metadata.json["test"]` into every
candidate; `--install` copies the one that ships to `models/security/alignment/`. When none passes,
`--install-shadow` installs the candidate with the best off-intent recall there as shadow-only
(SEC-128: `metadata.json["deployment"] = "shadow"`): its scores are recorded and never decide.
Counts only.

    python scripts/eval_alignment_model.py --model models/security/alignment-candidates/l6 \
        --model models/security/alignment-candidates/siamese [--install | --install-shadow]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("OMP_NUM_THREADS", "1")
BACKEND = Path(__file__).resolve().parents[1]
for p in (BACKEND, BACKEND / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np  # noqa: E402

from alignment_training import (DATA_DIR, call_text, expected_calibration_error, is_lookalike,  # noqa: E402
                                is_real_aligned, load_split, wilson_upper)

TARGETS = {"off_intent_recall": 0.95, "real_aligned_held_upper": 0.01, "risk_recall": 0.90,
           "lookalikes_held": 0.02, "ece_aligned": 0.05, "ece_risk": 0.05, "latency_p95_ms": 10.0}
INSTALL_DIR = BACKEND / "models" / "security" / "alignment"


def score(model, rows: List[dict]):
    pairs = [model.score_pair(r["intent"], call_text(r)) for r in rows]
    return np.array([p for p, _ in pairs]), np.array([r for _, r in pairs])


def report(rows: List[dict], p: np.ndarray, risk: np.ndarray, th: Dict[str, int]) -> Dict[str, object]:
    off = (p * 100) < th["search_align_at"]
    risky = risk >= th["risk_verify_at"]
    aligned = np.array([r["labels"]["aligned"] == "yes" for r in rows])
    has_risk = np.array([r["labels"]["risk"] != "none" for r in rows])
    real = np.array([is_real_aligned(r) for r in rows])
    look = np.array([is_lookalike(r) for r in rows])
    out: Dict[str, object] = {
        "rows": len(rows),
        "off_intent_recall": round(float(off[~aligned].mean()), 4) if (~aligned).any() else None,
        "real_aligned_held": int(off[real].sum()), "real_aligned": int(real.sum()),
        "real_aligned_held_upper": round(wilson_upper(int(off[real].sum()), int(real.sum())), 4),
        "risk_recall": round(float(risky[has_risk].mean()), 4) if has_risk.any() else None,
        "lookalikes_held": round(float((off | risky)[look].mean()), 4) if look.any() else None,
        "lookalikes": int(look.sum()),
        "ece_aligned": round(expected_calibration_error(p, aligned.astype(float)), 4),
        "ece_risk": round(expected_calibration_error(risk / 100.0, has_risk.astype(float)), 4),
    }
    by_origin: Dict[str, Counter] = defaultdict(Counter)
    for r, o, k in zip(rows, off, risky):
        by_origin[r.get("origin", "?").split(":")[0]]["rows"] += 1
        by_origin[r.get("origin", "?").split(":")[0]]["held"] += int(o or k)
    out["held_by_source"] = {k: dict(v) for k, v in sorted(by_origin.items())}
    return out


def latency(model, rows: List[dict], n: int = 200) -> Dict[str, float]:
    sample = (rows * (n // max(len(rows), 1) + 1))[:n]
    for r in sample[:10]:
        model.score_pair(r["intent"], call_text(r))
    times = []
    for i, r in enumerate(sample):
        started = time.perf_counter()
        model.score_pair(r["intent"], f"{call_text(r)} {i}")
        times.append((time.perf_counter() - started) * 1000)
    times.sort()
    return {"median_ms": round(statistics.median(times), 2), "p95_ms": round(times[int(0.95 * len(times))], 2)}


def passes(test: Dict[str, object], lat: Dict[str, float]) -> Dict[str, bool]:
    def ok(value, limit, higher=True):
        return value is not None and (value >= limit if higher else value <= limit)

    return {"off_intent_recall": ok(test["off_intent_recall"], TARGETS["off_intent_recall"]),
            "real_aligned_held_upper": ok(test["real_aligned_held_upper"], TARGETS["real_aligned_held_upper"], False),
            "risk_recall": ok(test["risk_recall"], TARGETS["risk_recall"]),
            "lookalikes_held": ok(test["lookalikes_held"], TARGETS["lookalikes_held"], False),
            "ece_aligned": ok(test["ece_aligned"], TARGETS["ece_aligned"], False),
            "ece_risk": ok(test["ece_risk"], TARGETS["ece_risk"], False),
            "latency_p95_ms": ok(lat["p95_ms"], TARGETS["latency_p95_ms"], False)}


def evaluate(model_dir: Path, data_dir: Path) -> Dict[str, object]:
    from app.security import alignment_model

    model = alignment_model.load(model_dir, require_gate=False)
    if model is None:
        raise SystemExit(f"{model_dir}: could not load")
    meta = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    th = meta["thresholds"]
    test_rows, probe_rows = load_split("test", data_dir), load_split("probe", data_dir)
    test = report(test_rows, *score(model, test_rows), th)
    probe = report(probe_rows, *score(model, probe_rows), th) if probe_rows else None
    lat = latency(model, test_rows)
    checks = passes(test, lat)
    return {"kind": meta["kind"], "version": meta["version"], "test": test, "probe": probe, "latency": lat,
            "checks": checks, "passed": all(checks.values())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, action="append", required=True)
    ap.add_argument("--data", type=Path, default=DATA_DIR)
    ap.add_argument("--install", action="store_true", help="copy the model that ships to models/security/alignment")
    ap.add_argument("--install-shadow", action="store_true",
                    help="when none passes, install the best candidate as shadow-only (SEC-128)")
    args = ap.parse_args()
    results = {str(m): evaluate(m, args.data) for m in args.model}
    siamese = [r for r in results.values() if r["kind"] == "siamese"]
    best_siamese = max((r["test"]["off_intent_recall"] or 0 for r in siamese), default=-1)
    shipping = None
    for path, r in sorted(results.items(), key=lambda kv: -(kv[1]["test"]["off_intent_recall"] or 0)):
        beats = r["kind"] == "siamese" or (r["test"]["off_intent_recall"] or 0) > best_siamese
        if r["passed"] and beats:
            shipping = path
            break
    for path, r in results.items():
        meta_path = Path(path) / "metadata.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["test"] = {**r["test"], "probe": r["probe"], "latency": r["latency"], "checks": r["checks"],
                        "passed": r["passed"] and path == shipping, "targets": TARGETS}
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"shipping": shipping, "results": results}, indent=2, default=str))
    if args.install and shipping:
        _install(Path(shipping), deployment="gated")
    elif args.install_shadow and not shipping and results:
        best = max(results, key=lambda p: results[p]["test"]["off_intent_recall"] or 0)
        _install(Path(best), deployment="shadow")
    return 0 if shipping else 1


def _install(model_dir: Path, *, deployment: str) -> None:
    if INSTALL_DIR.exists():
        for item in INSTALL_DIR.iterdir():  # overwrite in place (the repository lives in Dropbox)
            shutil.rmtree(item) if item.is_dir() else item.unlink()
    shutil.copytree(model_dir, INSTALL_DIR, dirs_exist_ok=True)
    meta_path = INSTALL_DIR / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["deployment"] = deployment
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"installed {model_dir} -> {INSTALL_DIR} ({deployment})")


if __name__ == "__main__":
    raise SystemExit(main())
