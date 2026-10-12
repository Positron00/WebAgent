#!/usr/bin/env python3
"""Release gate for the triage classifier, and a validator for the curated data.

Run from backend/:
    python scripts/eval_triage.py [--data data/triage/test.jsonl] [--model models/triage]
                                  [--check-data] [--write-metadata]
Exit code 0 only if research recall >= 95% and routine routed direct >= 80% (REQ-023),
or, with --check-data, if both data files are valid (REQ-020, REQ-021).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np

BACKEND = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
for p in (str(BACKEND), str(SCRIPTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

from triage_data import load_categories, load_rows  # noqa: E402
from train_triage import model_texts  # noqa: E402

RESEARCH_RECALL_TARGET = 0.95
ROUTINE_DIRECT_TARGET = 0.80
NEAR_DUPLICATE_SIMILARITY = 0.9
MIN_HARD_CASE_TRAIN = 20
MIN_HARD_CASE_TEST = 10


def _encoder(name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name, device="cpu")


def check_data(train_path: Path, test_path: Path, categories_path: Path, encoder=None,
               encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> List[str]:
    """Return a list of problems (empty means the data is valid)."""
    problems: List[str] = []
    cats = load_categories(categories_path)
    files = {"train": load_rows(train_path), "test": load_rows(test_path)}
    for name, rows in files.items():
        for row in rows:
            cat = cats.get(row["category"])
            if cat is None:
                problems.append(f"{name}: unknown category {row['category']!r}")
            elif cat["class"] != "mixed" and cat["class"] != row["label"]:
                problems.append(f"{name}: label {row['label']} does not match category {row['category']}")
        dupes = [t for t, n in Counter(r["text"] for r in rows).items() if n > 1]
        problems += [f"{name}: duplicate text {t[:60]!r}" for t in dupes]
        counts = Counter(r["category"] for r in rows)
        minimum = MIN_HARD_CASE_TRAIN if name == "train" else MIN_HARD_CASE_TEST
        for cid, cat in cats.items():
            if cat.get("hard_case") and counts[cid] < minimum:
                problems.append(f"{name}: hard case {cid} has {counts[cid]} rows, needs {minimum}")

    train_texts = {r["text"] for r in files["train"]}
    problems += [f"test row also in train: {r['text'][:60]!r}" for r in files["test"] if r["text"] in train_texts]

    enc = encoder or _encoder(encoder_name)
    train_emb = np.asarray(enc.encode(model_texts(files["train"]), normalize_embeddings=True))
    test_emb = np.asarray(enc.encode(model_texts(files["test"]), normalize_embeddings=True))
    sims = test_emb @ train_emb.T
    for i, j in zip(*np.where(sims >= NEAR_DUPLICATE_SIMILARITY)):
        problems.append(
            f"near-duplicate ({sims[i, j]:.2f}): test {files['test'][i]['text'][:50]!r} ~ "
            f"train {files['train'][j]['text'][:50]!r}"
        )
    return problems


def evaluate(model_dir: Path, test_path: Path, encoder=None) -> Dict[str, object]:
    meta = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    clf = joblib.load(model_dir / "classifier.joblib")["classifier"]
    threshold = float(meta["threshold"])
    rows = load_rows(test_path)
    # A fine-tuned encoder is saved next to the classifier (metadata "encoder_path").
    encoder_ref = str(model_dir / meta["encoder_path"]) if meta.get("encoder_path") else meta["encoder_name"]
    enc = encoder or _encoder(encoder_ref)
    X = np.asarray(enc.encode(model_texts(rows), normalize_embeddings=True))
    proba = clf.predict_proba(X)[:, list(clf.classes_).index("agentic")]
    pred_agentic = proba >= threshold
    is_agentic = np.array([r["label"] == "agentic" for r in rows])

    tp = int((pred_agentic & is_agentic).sum())
    research_recall = tp / int(is_agentic.sum()) if is_agentic.any() else 1.0
    routine_direct = float((~pred_agentic[~is_agentic]).mean()) if (~is_agentic).any() else 1.0
    precision = tp / int(pred_agentic.sum()) if pred_agentic.any() else 1.0

    per_category: Dict[str, Counter] = defaultdict(Counter)
    for row, pa in zip(rows, pred_agentic):
        per_category[row["category"]][f"{row['label']}->{'agentic' if pa else 'direct'}"] += 1

    errors = [
        (abs(p - threshold), row, p)
        for row, p, pa, ia in zip(rows, proba, pred_agentic, is_agentic)
        if pa != ia
    ]
    errors.sort(key=lambda e: -e[0])
    return {
        "threshold": threshold,
        "research_recall": round(research_recall, 4),
        "routine_direct": round(routine_direct, 4),
        "research_precision": round(precision, 4),
        "rows": len(rows),
        "per_category": {k: dict(v) for k, v in sorted(per_category.items())},
        "worst_errors": [(row["category"], row["label"], round(float(p), 3), row["text"][:100])
                         for _, row, p in errors[:20]],
    }


def passes(report: Dict[str, object]) -> bool:
    return (report["research_recall"] >= RESEARCH_RECALL_TARGET
            and report["routine_direct"] >= ROUTINE_DIRECT_TARGET)


def print_report(report: Dict[str, object]) -> None:
    print(f"threshold={report['threshold']} rows={report['rows']}")
    print(f"research recall   {report['research_recall']:.3f} (target {RESEARCH_RECALL_TARGET})")
    print(f"routine direct    {report['routine_direct']:.3f} (target {ROUTINE_DIRECT_TARGET})")
    print(f"research precision {report['research_precision']:.3f}")
    print("\nPer category (true label -> predicted route):")
    for cat, counts in report["per_category"].items():
        print(f"  {cat:22s} " + ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())))
    print("\nWorst errors (category, label, p_agentic, text):")
    for err in report["worst_errors"]:
        print(f"  {err}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=BACKEND / "data/triage/test.jsonl")
    parser.add_argument("--train", type=Path, default=BACKEND / "data/triage/train.jsonl")
    parser.add_argument("--categories", type=Path, default=BACKEND / "data/triage/categories.json")
    parser.add_argument("--model", type=Path, default=BACKEND / "models/triage")
    parser.add_argument("--check-data", action="store_true")
    parser.add_argument("--write-metadata", action="store_true")
    args = parser.parse_args(argv)

    if args.check_data:
        problems = check_data(args.train, args.data, args.categories)
        for problem in problems:
            print(problem)
        print("check-data OK" if not problems else f"check-data FAILED ({len(problems)} problems)")
        return 0 if not problems else 1

    report = evaluate(args.model, args.data)
    print_report(report)
    ok = passes(report)
    print("\nGATE PASSED" if ok else "\nGATE FAILED")
    if args.write_metadata:
        meta_path = args.model / "metadata.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["test"] = {k: report[k] for k in ("research_recall", "routine_direct", "research_precision", "rows")}
        meta["test"]["passed"] = ok
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
