#!/usr/bin/env python3
"""Blind review sheets and verdicts for generated batches (rl-generator SEC-212; design §9).

    python scripts/rl_generator/review.py --make h-prompted-20261024-1     # LOCAL/review/<batch>.csv
    python scripts/rl_generator/review.py --score h-prompted-20261024-1    # LOCAL/generated/rl_review.json
    python scripts/rl_generator/review.py --make judge-choice | --score judge-choice

The sheet has 100 rows stratified by action and seed origin (seed 7) and an empty `label_ok`
column (y or n); the judge's answer isn't shown. Every row in a batch was judged valid, so
disagreement is the user's `n`: <= 6% of 100 pass, > 14% fail, between `more` (100 more rows,
then pass iff <= 10% of 200). Prints counts only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Callable, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rl_generator  # noqa: E402
from rl_generator import GENERATED_DIR, REVIEW_DIR, rows  # noqa: E402
from rl_generator.prompts import QUESTION  # noqa: E402

rl_generator.bootstrap()

SHEET_ROWS = 100
SEED = 7
CHOICE = "judge-choice"
VERDICTS = GENERATED_DIR / "rl_review.json"


def verdict(n: int, user_no: int) -> str:
    """SEC-212's table: first 100 rows pass/more/fail; after `more`, 200 rows pass/fail at 10%."""
    if n <= 0:
        raise ValueError("no labelled rows")
    rate = user_no / n
    if n >= 2 * SHEET_ROWS:
        return "pass" if rate <= 0.10 else "fail"
    if rate <= 0.06:
        return "pass"
    return "fail" if rate > 0.14 else "more"


def row_id(*parts: str) -> str:
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:12]


def stratified(items: List[dict], key: Callable[[dict], tuple], k: int, seed: int = SEED) -> List[dict]:
    """`k` items with each stratum's share kept (largest remainder), random within strata."""
    rng = random.Random(seed)
    groups: Dict[tuple, List[dict]] = defaultdict(list)
    for item in items:
        groups[key(item)].append(item)
    k = min(k, len(items))
    order = sorted(groups)
    for g in order:
        rng.shuffle(groups[g])
    exact = {g: k * len(groups[g]) / len(items) for g in order}
    quota = {g: int(exact[g]) for g in order}
    for g in sorted(order, key=lambda g: exact[g] - quota[g], reverse=True)[:k - sum(quota.values())]:
        quota[g] += 1
    picked = [item for g in order for item in groups[g][:quota[g]]]
    rng.shuffle(picked)
    return picked


def read_sheet(path: Path) -> List[dict]:
    if not path.is_file():
        return []
    lines = [line for line in path.read_text(encoding="utf-8").splitlines(keepends=True) if not line.startswith("#")]
    return list(csv.DictReader(lines))


def write_sheet(path: Path, header: List[str], questions: List[str], items: List[dict], append: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not append or not path.is_file()
    with path.open("w" if new else "a", encoding="utf-8", newline="") as handle:
        if new:
            for q in questions:
                handle.write(f"# {q}\n")
        writer = csv.DictWriter(handle, fieldnames=header)
        if new:
            writer.writeheader()
        for item in items:
            writer.writerow({h: item.get(h, "") for h in header})


def find_batch(batch: str, gen_dir: Path) -> List[dict]:
    info = rows.parse_batch(batch)
    if info is None:
        raise SystemExit(f"not a batch id: {batch!r}")
    return rows.read_rows(rows.batch_path(info["mode"], info["kind"], info["heldout"], gen_dir), batch)


def load_verdicts(path: Path) -> dict:
    return json.loads(path.read_text()) if path.is_file() else {}


def make(batch: str, gen_dir: Path = GENERATED_DIR, review_dir: Path = REVIEW_DIR) -> dict:
    if batch == CHOICE:
        return make_choice(review_dir)
    sheet = review_dir / f"{batch}.csv"
    existing = read_sheet(sheet)
    if existing and load_verdicts(gen_dir / VERDICTS.name).get(batch, {}).get("verdict") != "more":
        raise SystemExit("sheet exists; score it first (a second --make needs a `more` verdict)")
    batch_rows = find_batch(batch, gen_dir)
    if not batch_rows:
        raise SystemExit("no rows in that batch")
    items = [{"row_id": row_id(batch, r["meta"]["seed_id"], r["call"]), "intent": r["intent"], "action": r["action"],
              "call": r["call"], "label_ok": "", "_origin": r["meta"].get("seed_origin") or r["sources"][-1]}
             for r in batch_rows]
    seen = {r["row_id"] for r in existing}
    items = [i for i in items if i["row_id"] not in seen]
    picked = stratified(items, lambda i: (i["action"], i["_origin"]), SHEET_ROWS, SEED + len(existing))
    mode = rows.parse_batch(batch)["mode"]
    write_sheet(sheet, ["row_id", "intent", "action", "call", "label_ok"], [QUESTION[mode]], picked, bool(existing))
    return {"batch": batch, "sheet_rows": len(existing) + len(picked), "added": len(picked)}


def _labels(sheet: List[dict]) -> Dict[str, bool]:
    bad = [r for r in sheet if (r.get("label_ok") or "").strip().lower() not in ("y", "n")]
    if bad:
        raise SystemExit(f"{len(bad)} rows have no y/n label_ok")
    return {r["row_id"]: r["label_ok"].strip().lower() == "y" for r in sheet}


def score(batch: str, gen_dir: Path = GENERATED_DIR, review_dir: Path = REVIEW_DIR) -> dict:
    if batch == CHOICE:
        return score_choice(review_dir)
    labels = _labels(read_sheet(review_dir / f"{batch}.csv"))
    if not labels:
        raise SystemExit("no sheet for that batch")
    user_no = sum(not ok for ok in labels.values())
    result = {"n": len(labels), "user_no": user_no, "verdict": verdict(len(labels), user_no),
              "date": date.today().isoformat()}
    path = gen_dir / VERDICTS.name
    all_verdicts = load_verdicts(path)
    all_verdicts[batch] = result
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(all_verdicts, indent=2) + "\n")
    return {"batch": batch, **result}


def _vote(item: dict) -> str:
    votes = [v for v in item["verdicts"].values() if v is not None]  # None: that candidate's call failed
    if not votes:
        return "failed"
    return "all_yes" if all(votes) else "all_no" if not any(votes) else "split"


def make_choice(review_dir: Path = REVIEW_DIR) -> dict:
    items = rows.read_rows(review_dir / f"{CHOICE}.jsonl")
    if not items:
        raise SystemExit("no judge-choice samples; run prompted.py --estimate N --judge-candidates ...")
    picked = stratified(items, lambda i: (i["mode"], _vote(i)), SHEET_ROWS)
    questions = [f"mode {m}: {QUESTION[m]}" for m in sorted(QUESTION)]
    write_sheet(review_dir / f"{CHOICE}.csv", ["row_id", "mode", "intent", "action", "call", "label_ok"],
                questions, [{**i, "label_ok": ""} for i in picked], append=False)
    return {"sheet_rows": len(picked)}


def score_choice(review_dir: Path = REVIEW_DIR) -> dict:
    labels = _labels(read_sheet(review_dir / f"{CHOICE}.csv"))
    out: Dict[str, Dict[str, int]] = {}
    for item in rows.read_rows(review_dir / f"{CHOICE}.jsonl"):
        if item["row_id"] not in labels:
            continue
        user = labels[item["row_id"]]
        for model, said in item["verdicts"].items():
            c = out.setdefault(model, {"accepted": 0, "accepted_agree": 0, "rejected": 0, "rejected_agree": 0,
                                       "failed": 0})
            if said is None:  # a failed call is neither an acceptance nor a rejection
                c["failed"] += 1
                continue
            side = "accepted" if said else "rejected"
            c[side] += 1
            c[f"{side}_agree"] += int(user == bool(said))
    return {"rows": len(labels), "candidates": out}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--make", metavar="BATCH")
    group.add_argument("--score", metavar="BATCH")
    ap.add_argument("--generated-dir", type=Path, default=GENERATED_DIR)
    ap.add_argument("--review-dir", type=Path, default=REVIEW_DIR)
    args = ap.parse_args(argv)
    result = (make(args.make, args.generated_dir, args.review_dir) if args.make
              else score(args.score, args.generated_dir, args.review_dir))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
