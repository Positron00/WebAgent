#!/usr/bin/env python3
"""Train the checkpoint 2 alignment classifier (checkpoint2-intent SEC-123, SEC-124; task B5).

Two candidates from the same rows:
- cross_encoder: MiniLM (L6 or L12) reading (intent, "<action>: <call>") as one pair, mean-pooled,
  with two heads: `aligned` (1 logit) and `risk` (none/low/high);
- siamese: logistic heads on [u, v, |u-v|, u*v] of ContextDefender's encoder embeddings.
Both are calibrated (isotonic, on a fold held out from train) and get thresholds and an uncertain
band from that fold (`alignment_training.choose_thresholds`). The gate (`eval_alignment_model.py`)
then decides which one ships. Generated rows (rl-generator) never enter the calibration fold; `--seed`
(default 13) sets the fold, shuffling, torch and numpy; metadata.json records it and any `variant.json`.

    python scripts/train_alignment_model.py --kind cross_encoder --encoder sentence-transformers/all-MiniLM-L6-v2 \
        --out models/security/alignment-candidates/l6
    python scripts/train_alignment_model.py --kind siamese --out models/security/alignment-candidates/siamese
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from datetime import date
from pathlib import Path
from typing import List

import joblib
import numpy as np

BACKEND = Path(__file__).resolve().parents[1]
for p in (BACKEND, BACKEND / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from alignment_data import GENERATED, dataset_of  # noqa: E402
from alignment_training import (DATA_DIR, call_text, choose_thresholds, labels, load_split,  # noqa: E402
                                raw_risk)
from sklearn.isotonic import IsotonicRegression  # noqa: E402

SEED = 13


def data_hash(data_dir: Path) -> str:
    h = hashlib.sha256()
    for name in ("train.jsonl", "local/train.jsonl"):
        path = data_dir / name
        if path.exists():
            h.update(path.read_bytes())
    return h.hexdigest()[:16]


def split_calibration(rows: List[dict], share: float = 0.1, seed: int = SEED):
    """(train, calibration): the fold is `share` of the real rows only; generated rows (rl-generator
    SEC-214) always train. Without generated rows this is the same fold and order as before."""
    real = [i for i, r in enumerate(rows) if dataset_of(r["origin"]) not in GENERATED]
    generated = [i for i, r in enumerate(rows) if dataset_of(r["origin"]) in GENERATED]
    random.Random(seed).shuffle(real)
    cut = int(len(real) * share)
    calib = [rows[i] for i in real[:cut]]
    train = [rows[i] for i in real[cut:] + generated]
    return train, calib


def calibrate(p_aligned_raw, risk_raw, rows):
    aligned, risk = labels(rows)
    cal_a = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p_aligned_raw, aligned)
    cal_r = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(risk_raw, (risk > 0).astype(float))
    return {"aligned": cal_a, "risk": cal_r}


def finish(out: Path, kind: str, base: str, calib_rows, p_raw, r_raw, extra: dict, data_dir: Path = DATA_DIR,
           seed: int = SEED) -> None:
    calibrators = calibrate(p_raw, r_raw, calib_rows)
    joblib.dump(calibrators, out / "calibrators.joblib")
    p_cal = calibrators["aligned"].predict(p_raw)
    r_cal = np.round(100 * calibrators["risk"].predict(r_raw)).astype(int)
    chosen = choose_thresholds(calib_rows, p_cal, r_cal)
    meta = {"kind": kind, "version": f"ac-{date.today().isoformat()}-{extra.pop('n', 1)}", "base_encoder": base,
            "max_length": 128, "data_hash": data_hash(data_dir), "trained_rows": extra.pop("trained_rows"),
            **chosen, **extra, "train_seed": seed}
    variant = data_dir / "variant.json"
    if variant.is_file():
        meta["variant"] = json.loads(variant.read_text(encoding="utf-8"))
    (out / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: meta[k] for k in ("kind", "version", "thresholds", "uncertain_band")}, indent=2))


def train_cross_encoder(args, train_rows, calib_rows) -> None:
    import torch
    from torch import nn
    from transformers import AutoModel, AutoTokenizer

    seed = getattr(args, "seed", SEED)
    torch.manual_seed(seed)
    device = "mps" if args.device == "auto" and torch.backends.mps.is_available() else (
        "cpu" if args.device == "auto" else args.device)
    tok = AutoTokenizer.from_pretrained(args.encoder)
    enc = AutoModel.from_pretrained(args.encoder).to(device)
    hidden = enc.config.hidden_size
    heads = nn.ModuleDict({"aligned": nn.Linear(hidden, 1), "risk": nn.Linear(hidden, 3)}).to(device)
    params = list(enc.parameters()) + list(heads.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr)
    aligned_y, risk_y = labels(train_rows)
    risk_w = torch.tensor(len(risk_y) / (3 * np.maximum(np.bincount(risk_y, minlength=3), 1)), dtype=torch.float32)
    pos_w = torch.tensor([(aligned_y == 0).sum() / max((aligned_y == 1).sum(), 1)], dtype=torch.float32)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_w.to(device))
    ce = nn.CrossEntropyLoss(weight=risk_w.to(device))

    def forward(batch_rows):
        t = tok([r["intent"] for r in batch_rows], [call_text(r) for r in batch_rows], truncation=True,
                max_length=128, padding=True, return_tensors="pt").to(device)
        out = enc(**t).last_hidden_state
        mask = t["attention_mask"].unsqueeze(-1).float()
        pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
        return heads["aligned"](pooled).squeeze(-1), heads["risk"](pooled)

    order = list(range(len(train_rows)))
    for epoch in range(args.epochs):
        random.Random(seed + epoch).shuffle(order)
        enc.train()
        total = 0.0
        for start in range(0, len(order), args.batch):
            idx = order[start:start + args.batch]
            batch = [train_rows[i] for i in idx]
            a_logit, r_logit = forward(batch)
            loss = bce(a_logit, torch.tensor(aligned_y[idx], dtype=torch.float32, device=device)) + \
                ce(r_logit, torch.tensor(risk_y[idx], dtype=torch.long, device=device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss) * len(idx)
        print(f"epoch {epoch + 1}: loss {total / len(order):.4f}")
    enc.eval()
    p_raw, r_raw = [], []
    with torch.no_grad():
        for start in range(0, len(calib_rows), 64):
            a_logit, r_logit = forward(calib_rows[start:start + 64])
            p_raw += torch.sigmoid(a_logit).cpu().tolist()
            r_raw += raw_risk(torch.softmax(r_logit, dim=-1).cpu().numpy()).tolist()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    enc.cpu().save_pretrained(out / "encoder")
    tok.save_pretrained(out / "encoder")
    torch.save({"aligned": heads["aligned"].cpu().state_dict(), "risk": heads["risk"].cpu().state_dict(),
                "hidden": hidden, "pooling": "mean", "max_length": 128}, out / "heads.pt")
    finish(out, "cross_encoder", args.encoder, calib_rows, np.array(p_raw), np.array(r_raw),
           {"trained_rows": len(train_rows), "epochs": args.epochs, "lr": args.lr}, args.data, seed)


def siamese_features(rows: List[dict]) -> np.ndarray:
    from app.security.alignment_model import siamese_embed  # same encoder and features as the runtime

    u = siamese_embed([r["intent"] for r in rows])
    v = siamese_embed([call_text(r) for r in rows])
    return np.hstack([u, v, np.abs(u - v), u * v])


def train_siamese(args, train_rows, calib_rows) -> None:
    from sklearn.linear_model import LogisticRegression

    seed = getattr(args, "seed", SEED)
    aligned_y, risk_y = labels(train_rows)
    x = siamese_features(train_rows)
    heads = {"aligned": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed).fit(x, aligned_y),
             "risk": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed).fit(x, risk_y)}
    xc = siamese_features(calib_rows)
    p_raw = heads["aligned"].predict_proba(xc)[:, 1]
    r_raw = raw_risk(heads["risk"].predict_proba(xc))
    args.out.mkdir(parents=True, exist_ok=True)
    joblib.dump(heads, args.out / "heads.joblib")
    finish(args.out, "siamese", "context_defender", calib_rows, p_raw, r_raw, {"trained_rows": len(train_rows)},
           args.data, seed)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", choices=("cross_encoder", "siamese"), required=True)
    ap.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--data", type=Path, default=DATA_DIR)
    ap.add_argument("--seed", type=int, default=SEED, help="calibration split, shuffling, torch and numpy (SEC-217)")
    args = ap.parse_args()
    np.random.seed(args.seed)
    rows = load_split("train", args.data)
    if not rows:
        ap.error(f"no training rows under {args.data}")
    train_rows, calib_rows = split_calibration(rows, seed=args.seed)
    (train_cross_encoder if args.kind == "cross_encoder" else train_siamese)(args, train_rows, calib_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
