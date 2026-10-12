#!/usr/bin/env python3
"""Train the triage classifier: MiniLM embeddings + logistic regression.

Run from backend/:
    python scripts/train_triage.py [--data data/triage/train.jsonl] [--out models/triage]
                                   [--version triage-YYYY-MM-x] [--recall-target 0.95] [--seed 13]
                                   [--finetune [--pairs-per-row 4] [--epochs 1]]
Reads only train.jsonl (never test.jsonl or collected files).

By default the encoder stays frozen. --finetune (alias --setfit) first fine-tunes MiniLM the
SetFit way on the fit split: pairs of questions with the same label are pulled together, pairs
with different labels pushed apart (cosine-similarity loss). The encoder is saved to
<out>/encoder/ and metadata.json records it as "encoder_path". The validation split is never
used for fine-tuning, so the threshold is still picked on questions the encoder hasn't seen.
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


def contrastive_pairs(texts: List[str], y: np.ndarray, pairs_per_row: int, seed: int):
    """SetFit-style pairs: for every row, `pairs_per_row` same-label partners (score 1.0)
    and as many different-label partners (score 0.0)."""
    rng = np.random.default_rng(seed)
    by_label = {label: np.flatnonzero(y == label) for label in np.unique(y)}
    first, second, scores = [], [], []
    for i, label in enumerate(y):
        same = by_label[label][by_label[label] != i]
        other = np.concatenate([idx for lab, idx in by_label.items() if lab != label])
        for j in rng.choice(same, size=min(pairs_per_row, len(same)), replace=False):
            first.append(texts[i]); second.append(texts[j]); scores.append(1.0)
        for j in rng.choice(other, size=min(pairs_per_row, len(other)), replace=False):
            first.append(texts[i]); second.append(texts[j]); scores.append(0.0)
    return first, second, scores


def finetune_encoder(encoder, texts: List[str], y: np.ndarray, pairs_per_row: int, epochs: int, seed: int):
    """Fine-tune `encoder` in place on contrastive pairs (sentence-transformers, no extra packages)."""
    import tempfile

    from datasets import Dataset
    from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
    from sentence_transformers.losses import CosineSimilarityLoss

    first, second, scores = contrastive_pairs(texts, y, pairs_per_row, seed)
    dataset = Dataset.from_dict({"sentence1": first, "sentence2": second, "score": scores}).shuffle(seed=seed)
    print(f"Fine-tuning on {len(dataset)} pairs for {epochs} epoch(s)")
    with tempfile.TemporaryDirectory() as tmp:
        args = SentenceTransformerTrainingArguments(
            output_dir=tmp,
            num_train_epochs=epochs,
            per_device_train_batch_size=32,
            learning_rate=2e-5,
            warmup_ratio=0.1,
            seed=seed,
            save_strategy="no",
            logging_steps=100,
            report_to="none",
        )
        SentenceTransformerTrainer(
            model=encoder, args=args, train_dataset=dataset, loss=CosineSimilarityLoss(encoder)
        ).train()
    return encoder


def train(
    data_path: Path,
    out_dir: Path,
    version: str,
    encoder_name: str = DEFAULT_ENCODER,
    recall_target: float = 0.95,
    seed: int = 13,
    encoder=None,
    categories_path: Optional[Path] = None,
    finetune: bool = False,
    pairs_per_row: int = 4,
    epochs: int = 1,
) -> dict:
    rows = load_rows(data_path)
    y = np.array([1 if r["label"] == "agentic" else 0 for r in rows])
    cat_path = categories_path or data_path.parent / "categories.json"
    labels = {cid: c["label"] for cid, c in load_categories(cat_path).items()} if cat_path.is_file() else {}
    texts = model_texts(rows)

    if encoder is None:
        from sentence_transformers import SentenceTransformer

        # Fine-tuning runs faster on the GPU when there is one; the server always uses the CPU.
        device = "cpu"
        if finetune:
            import torch

            device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        encoder = SentenceTransformer(encoder_name, device=device)

    idx = np.arange(len(rows))
    fit_idx, val_idx = train_test_split(idx, test_size=0.2, random_state=seed, stratify=y)
    if finetune:
        finetune_encoder(encoder, [texts[i] for i in fit_idx], y[fit_idx], pairs_per_row, epochs, seed)
    X = np.asarray(encoder.encode(texts, normalize_embeddings=True))
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
    encoder_dir = out_dir / "encoder"
    if finetune:
        import shutil

        shutil.rmtree(encoder_dir, ignore_errors=True)
        encoder.save(str(encoder_dir))
    joblib.dump(
        {"classifier": final, "centroids": centroids, "category_ids": category_ids},
        out_dir / "classifier.joblib",
    )
    meta = {
        "version": version,
        "encoder_name": encoder_name,
        # Fine-tuned encoder saved next to the classifier; the base name stays in encoder_name.
        "encoder_path": "encoder" if finetune else None,
        "fine_tuning": (
            {"method": "setfit-style contrastive pairs", "pairs_per_row": pairs_per_row, "epochs": epochs,
             "fit_rows": int(len(fit_idx))}
            if finetune else None
        ),
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
    parser.add_argument("--finetune", "--setfit", action="store_true",
                        help="Fine-tune the encoder on contrastive pairs before fitting the classifier")
    parser.add_argument("--pairs-per-row", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()
    train(args.data, args.out, args.version, args.encoder, args.recall_target, args.seed,
          finetune=args.finetune, pairs_per_row=args.pairs_per_row, epochs=args.epochs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
