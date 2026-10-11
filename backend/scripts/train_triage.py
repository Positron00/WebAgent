#!/usr/bin/env python3
"""Train the triage classifier: frozen MiniLM embeddings + logistic regression.

Run from backend/:
    python scripts/train_triage.py [--data data/triage/train.jsonl] [--out models/triage]
                                   [--version triage-YYYY-MM-x] [--recall-target 0.95] [--seed 13]
Reads only train.jsonl (never test.jsonl or collected files).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

BACKEND = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
for p in (str(BACKEND), str(SCRIPTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

from triage_data import load_categories, load_rows  # noqa: E402
from app.services.triage import build_model_input  # noqa: E402

DEFAULT_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
MAX_THRESHOLD = 0.5


def data_hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def rates(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> Dict[str, float]:
    """Research recall (agentic rows sent agentic) and direct rate (direct rows sent direct)."""
    pred = proba >= threshold
    research = y_true == 1
    routine = y_true == 0
    return {
        "research_recall": float(pred[research].mean()) if research.any() else 1.0,
        "direct_rate": float((~pred[routine]).mean()) if routine.any() else 1.0,
    }


def pick_threshold(y_val: np.ndarray, proba_val: np.ndarray, recall_target: float) -> float:
    """Highest threshold (at most 0.5) whose validation research recall meets the target.

    Any lower threshold also meets the target but sends more routine questions to research.
    """
    candidates = sorted({float(x) for x in proba_val if x <= MAX_THRESHOLD} | {MAX_THRESHOLD}, reverse=True)
    for thr in candidates:
        if rates(y_val, proba_val, thr)["research_recall"] >= recall_target:
            return thr
    return min(candidates)


def model_texts(rows: List[dict]) -> List[str]:
    # Follow-up rows are already joined ("previous → current"), so no previous question is passed.
    return [build_model_input(r["text"], None) for r in rows]


def category_centroids(emb: np.ndarray, rows: List[dict]):
    ids = sorted({r["category"] for r in rows})
    cats = np.array([r["category"] for r in rows])
    centroids = []
    for cat in ids:
        mean = emb[cats == cat].mean(axis=0)
        centroids.append(mean / (np.linalg.norm(mean) or 1.0))
    return np.vstack(centroids).astype(np.float32), ids


def train(
    data_path: Path,
    out_dir: Path,
    version: str,
    encoder_name: str = DEFAULT_ENCODER,
    recall_target: float = 0.95,
    seed: int = 13,
    encoder=None,
    categories_path: Optional[Path] = None,
) -> dict:
    rows = load_rows(data_path)
    y = np.array([1 if r["label"] == "agentic" else 0 for r in rows])
    cat_path = categories_path or data_path.parent / "categories.json"
    labels = {cid: c["label"] for cid, c in load_categories(cat_path).items()} if cat_path.is_file() else {}

    if encoder is None:
        from sentence_transformers import SentenceTransformer

        encoder = SentenceTransformer(encoder_name, device="cpu")
    X = np.asarray(encoder.encode(model_texts(rows), normalize_embeddings=True))

    idx = np.arange(len(rows))
    fit_idx, val_idx = train_test_split(idx, test_size=0.2, random_state=seed, stratify=y)
    clf = LogisticRegression(class_weight="balanced", C=1.0, max_iter=1000)
    clf.fit(X[fit_idx], y[fit_idx])
    proba_val = clf.predict_proba(X[val_idx])[:, list(clf.classes_).index(1)]
    threshold = pick_threshold(y[val_idx], proba_val, recall_target)
    validation = rates(y[val_idx], proba_val, threshold)

    # Refit on all rows; the classifier keeps string classes so the server can find "agentic".
    final = LogisticRegression(class_weight="balanced", C=1.0, max_iter=1000)
    final.fit(X, np.array(["agentic" if v == 1 else "direct" for v in y]))
    centroids, category_ids = category_centroids(X, rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"classifier": final, "centroids": centroids, "category_ids": category_ids},
        out_dir / "classifier.joblib",
    )
    meta = {
        "version": version,
        "encoder_name": encoder_name,
        "threshold": round(threshold, 4),
        "data_hash": data_hash(data_path),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "categories": {cid: labels.get(cid, cid) for cid in category_ids},
        "validation": {k: round(v, 4) for k, v in validation.items()},
        "test": None,
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(
        f"Wrote {out_dir}: threshold={threshold:.3f} "
        f"val_research_recall={validation['research_recall']:.3f} val_direct_rate={validation['direct_rate']:.3f}"
    )
    return meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=BACKEND / "data/triage/train.jsonl")
    parser.add_argument("--out", type=Path, default=BACKEND / "models/triage")
    parser.add_argument("--version", default=f"triage-{datetime.now():%Y-%m}-a")
    parser.add_argument("--encoder", default=DEFAULT_ENCODER)
    parser.add_argument("--recall-target", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--setfit", action="store_true", help="Fine-tune the encoder (phase 4; not built yet)")
    args = parser.parse_args()
    if args.setfit:
        print("--setfit is planned for phase 4 (REQ-024) and is not implemented yet.", file=sys.stderr)
        return 2
    train(args.data, args.out, args.version, args.encoder, args.recall_target, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
