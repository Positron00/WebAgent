#!/usr/bin/env python3
"""Train ContextDefender heads (MiniLM + TF-IDF routes, calibrated per axis).

Run from backend/:
    python scripts/train_context_defender.py [--data data/security/train.jsonl]
                                             [--out models/security] [--version cd-YYYY-MM-x]
                                             [--tfidf-only] [--embedding-only] [--C 10]
                                             [--chunk-size 96] [--chunk-stride 64]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

BACKEND = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
for p in (str(BACKEND), str(SCRIPTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

from security_data import AXES, LABEL_LEVELS, LOOKALIKE_ORIGINS, is_ordinary_question, load_rows  # noqa: E402
from app.security.context_defender import DEFAULT_ENCODER, chunk_text, featurize, route_chunk  # noqa: E402


def data_hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def build_vocabulary_from_texts(texts: Sequence[str], max_size: int = 50_000) -> set:
    counts = Counter()
    for text in texts:
        counts.update(re.findall(r"[A-Za-z0-9_]+", text.lower()))
    return set(w for w, _ in counts.most_common(max_size))


def row_chunks(text: str, tokenizer) -> List[str]:
    return chunk_text(text, tokenizer) or [text]


def prepare_chunk_examples(
    rows: List[dict],
    vocabulary: set,
    tokenizer,
    tfidf_only: bool = False,
) -> Dict[str, Dict[str, List[Tuple[str, str]]]]:
    """axis -> route -> list of (chunk_text, label_level)."""
    buckets: Dict[str, Dict[str, List[Tuple[str, str]]]] = {
        axis: {"minilm": [], "tfidf": []} for axis in AXES
    }
    for row in rows:
        for chunk in row_chunks(row["text"], tokenizer):
            route = "tfidf" if tfidf_only else route_chunk(chunk, vocabulary)
            for axis in AXES:
                buckets[axis][route].append((chunk, row["labels"][axis]))
    return buckets


def _features(texts: List[str], axis: str, encoder, route: str, vectorizer: TfidfVectorizer) -> np.ndarray:
    # The runtime's featurize, so training and serving features can't drift (SEC-022).
    return featurize(texts, route, encoder, vectorizer)


def _alnum(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _span_anchors(row: dict):
    """Short alphanumeric snippets from the start, middle and end of an embedded injection, used to
    find the chunks that contain it (tokenizer decoding changes spacing and case)."""
    span = row.get("span")
    if not span:
        return None
    inj = _alnum(row["text"][span[0]:span[1]])
    if len(inj) < 24:
        return [inj] if inj else None
    mid = len(inj) // 2
    return [inj[:24], inj[mid - 12: mid + 12], inj[-24:]]


def fit_head_features(x, y_labels: List[str], seed: int, c: float = 1.0, sample_weight=None):
    """Calibrated binary head (none vs. low/high) on precomputed features."""
    y = np.array([0 if l == "none" else 1 for l in y_labels])
    if len(set(y)) < 2:
        from sklearn.dummy import DummyClassifier

        return DummyClassifier(strategy="constant", constant=int(y[0])).fit(x, y)
    base = LogisticRegression(max_iter=3000, random_state=seed, class_weight="balanced", C=c)
    sw = None if sample_weight is None else np.asarray(sample_weight)
    folds = int(min(3, np.bincount(y).min()))
    if folds < 2:
        return base.fit(x, y, sample_weight=sw)
    from sklearn.calibration import CalibratedClassifierCV

    return CalibratedClassifierCV(base, method="sigmoid", cv=folds).fit(x, y, sample_weight=sw)


def fit_head(
    texts: List[str],
    y_labels: List[str],
    axis: str,
    encoder,
    route: str,
    vectorizer: TfidfVectorizer,
    seed: int,
):
    if not texts:
        return None
    y = np.array([0 if l == "none" else 1 for l in y_labels])
    x = _features(texts, axis, encoder, route, vectorizer)
    if len(set(y)) < 2:
        from sklearn.dummy import DummyClassifier

        return DummyClassifier(strategy="constant", constant=int(np.bincount(y).argmax())).fit(x, y)
    base = LogisticRegression(max_iter=2000, random_state=seed, class_weight="balanced", C=1.0)
    # Calibrated probabilities (SEC-024): Platt scaling, which needs less data than isotonic.
    folds = int(min(3, np.bincount(y).min()))
    if folds < 2:
        return base.fit(x, y)
    from sklearn.calibration import CalibratedClassifierCV

    return CalibratedClassifierCV(base, method="sigmoid", cv=folds).fit(x, y)


# Benign sets written to look risky on one axis; the 2026-10-11 model flagged 36-46% of them (harm)
# and 39% of role-play prompts (jailbreak). Their chunks weigh more on that axis when fitting.
HARD_NEGATIVES: Dict[str, Tuple[str, ...]] = LOOKALIKE_ORIGINS
HARD_NEGATIVE_WEIGHT = 3.0


def chunk_weight(row: dict, axis: str, weight: float = HARD_NEGATIVE_WEIGHT) -> float:
    return weight if axis in HARD_NEGATIVES.get(row.get("origin", ""), ()) else 1.0


def build_chunks(rows: List[dict], tokenizer, vocabulary: set, use_minilm: bool,
                 hard_negative_weight: float = HARD_NEGATIVE_WEIGHT):
    """One example per chunk: (chunks, routes, labels per axis, weights per axis, row index).

    A row's labels apply to each of its chunks, except that an inserted injection (`span`) only
    labels the chunks containing it."""
    chunks: List[str] = []
    routes: List[str] = []
    owners: List[int] = []
    chunk_labels: Dict[str, List[str]] = {axis: [] for axis in AXES}
    weights: Dict[str, List[float]] = {axis: [] for axis in AXES}
    for r, row in enumerate(rows):
        anchors = _span_anchors(row)
        for chunk in row_chunks(row["text"], tokenizer):
            chunks.append(chunk)
            owners.append(r)
            for axis in AXES:
                weights[axis].append(chunk_weight(row, axis, hard_negative_weight))
            routes.append("tfidf" if not use_minilm else route_chunk(chunk, vocabulary))
            for axis in AXES:
                level = row["labels"][axis]
                if anchors is not None and axis == "injection" and level != "none":
                    # The rest of the page is benign text (SEC-023).
                    flat = _alnum(chunk)
                    level = level if any(a in flat for a in anchors) else "none"
                chunk_labels[axis].append(level)
    return chunks, routes, chunk_labels, weights, owners


def calibration_key(row: dict) -> str:
    """Rows that reuse one inserted attack share a key, so they land on the same side of the split."""
    span = row.get("span")
    text = row["text"][span[0]:span[1]] if span else row["text"]
    return re.sub(r"\s+", " ", text.strip().lower())


def split_for_calibration(rows: List[dict], seed: int, fraction: float = 0.15):
    fit, cal = [], []
    for row in rows:
        h = int(hashlib.sha256(f"cal:{seed}:{calibration_key(row)}".encode()).hexdigest(), 16) % 1000
        (cal if h < fraction * 1000 else fit).append(row)
    return fit, cal


def finetune_encoder(encoder, chunks: List[str], chunk_labels: Dict[str, List[str]], seed: int,
                     epochs: int = 2, batch_size: int = 32, lr: float = 2e-5, log=print,
                     weights: Optional[Dict[str, List[float]]] = None):
    """Fine-tune the SentenceTransformer in place with a 4-axis multi-label head (BCE)."""
    import torch
    from torch import nn

    torch.manual_seed(seed)
    device = encoder.device
    y = torch.tensor([[0.0 if chunk_labels[a][i] == "none" else 1.0 for a in AXES] for i in range(len(chunks))])
    pos = y.sum(0).clamp(min=1)
    pos_weight = ((len(chunks) - pos) / pos).clamp(max=10.0).to(device)
    dim = encoder.get_sentence_embedding_dimension()
    head = nn.Linear(dim, len(AXES)).to(device)
    params = [{"params": encoder.parameters(), "lr": lr}, {"params": head.parameters(), "lr": 1e-3}]
    opt = torch.optim.AdamW(params, weight_decay=0.01)
    steps = epochs * ((len(chunks) + batch_size - 1) // batch_size)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: min(1.0, (k + 1) / (0.06 * steps)) * max(0.0, 1 - k / steps))
    w = (torch.tensor([[weights[a][i] for a in AXES] for i in range(len(chunks))]) if weights
         else torch.ones_like(y))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    rng = np.random.default_rng(seed)
    encoder.train()
    for epoch in range(epochs):
        order = rng.permutation(len(chunks))
        total = 0.0
        for b in range(0, len(order), batch_size):
            idx = order[b:b + batch_size]
            feats = encoder.tokenize([chunks[i] for i in idx])
            feats = {k: v.to(device) for k, v in feats.items()}
            emb = encoder(feats)["sentence_embedding"]
            # The encoder normalizes embeddings (norm 1); a fixed scale lets the head reach confident logits.
            wb = w[idx].to(device)
            loss = (loss_fn(head(emb) * 10.0, y[idx].to(device)) * wb).sum() / wb.sum()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            opt.step()
            sched.step()
            total += float(loss) * len(idx)
        log(f"fine-tune epoch {epoch + 1}/{epochs}: loss {total / len(chunks):.4f}")
    encoder.eval()
    return encoder


def fit_prefit_calibrated(x_fit, y_fit: List[str], x_cal, y_cal: List[str], seed: int, c: float,
                          sample_weight=None):
    """Logistic head fitted on one part, Platt-calibrated on rows it (and the encoder) never saw.

    Calibration is unweighted, so the probabilities stay true to the data; only fitting is weighted."""
    from sklearn.calibration import CalibratedClassifierCV

    yf = np.array([0 if l == "none" else 1 for l in y_fit])
    yc = np.array([0 if l == "none" else 1 for l in y_cal])
    if len(set(yf)) < 2 or len(set(yc)) < 2:
        return fit_head_features(x_fit, y_fit, seed, c=c, sample_weight=sample_weight)
    base = LogisticRegression(max_iter=3000, random_state=seed, class_weight="balanced", C=c)
    base.fit(x_fit, yf, sample_weight=None if sample_weight is None else np.asarray(sample_weight))
    return CalibratedClassifierCV(base, method="sigmoid", cv="prefit").fit(x_cal, yc)


UNTRUSTED = {"web_result", "retrieved_doc", "document", "tool_output", "unlabelled"}
# Acting thresholds: (key, axis, benign rows that count as false positives, positives, range,
# use the confidence bound). Every acting score is chosen against the upper confidence bound of the
# false-positive rate: on a few hundred held-out questions a point estimate of 0.6% came out at 3.9%
# on test. (Injection used a point estimate with a margin until 2026-10-13, while its held-out set was small.)
THRESHOLD_RULES = (
    # Since 2026-10-13 injection uses the bound too: the held-out part now has ~800 benign untrusted
    # rows, and the point estimate twice gave thresholds whose test false positives sat at the limit.
    ("injection_drop_at", "injection", lambda r: r["source"] in UNTRUSTED and _benign(r),
     lambda r: r["source"] in UNTRUSTED, (30, 95), True),
    ("harm_decline_at", "harm", is_ordinary_question, lambda r: r["source"] == "user_turn", (50, 98), True),
    ("jailbreak_consent_at", "jailbreak", is_ordinary_question, lambda r: r["source"] == "user_turn", (50, 98), True),
)
CONFIDENCE_Z = 1.645  # one-sided 95%


def wilson_upper(hits: int, n: int, z: float = CONFIDENCE_Z) -> float:
    """Upper end of the Wilson score interval for a rate of hits/n (1.0 when n is 0)."""
    if n == 0:
        return 1.0
    p = hits / n
    centre = p + z * z / (2 * n)
    spread = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return min(1.0, (centre + spread) / (1 + z * z / n))
# Verify thresholds: the highest score that still sends >= this share of positives to decline or
# the judge (the gate's recall target plus a margin, since the held-out part is small).
VERIFY_RULES = (("harm_verify_at", "harm", "harm_decline_at"), ("jailbreak_verify_at", "jailbreak", "jailbreak_consent_at"))
VERIFY_RECALL = 0.92
VERIFY_MIN = 20
THRESHOLD_MARGIN = 0.75  # aim under the gate's false-positive limit, since the held-out part is small


def _benign(row: dict) -> bool:
    return all(v == "none" for v in row["labels"].values())


def row_scores(heads, index, features, owners: List[int], n_rows: int) -> Dict[str, np.ndarray]:
    """Per row and axis, the max calibrated probability over its chunks (the runtime's pooling)."""
    out = {a: np.zeros(n_rows) for a in AXES}
    for route, ids in index.items():
        for axis in AXES:
            head = heads[axis].get(route)
            if not ids or head is None:
                continue
            probs = head.predict_proba(features[route])[:, -1]
            for i, p in zip(ids, probs):
                out[axis][owners[i]] = max(out[axis][owners[i]], p)
    return out


def select_thresholds(rows: List[dict], scores: Dict[str, np.ndarray]) -> Tuple[Dict[str, int], Dict[str, Any]]:
    """Lowest score per axis whose false-positive rate on held-out benign rows meets the gate's limit
    (with a margin), so recall is as high as that limit allows (SEC-028a)."""
    from eval_context_defender import HARM_FP, INJECTION_FP, JAILBREAK_FP

    limits = {"injection": INJECTION_FP, "harm": HARM_FP, "jailbreak": JAILBREAK_FP}
    chosen: Dict[str, int] = {}
    report: Dict[str, Any] = {}
    for key, axis, fp_mask, pos_mask, (lo, hi), use_bound in THRESHOLD_RULES:
        ints = np.round(100 * scores[axis]).astype(int)
        benign = [i for i, r in enumerate(rows) if fp_mask(r)]
        positive = [i for i, r in enumerate(rows) if pos_mask(r) and r["labels"][axis] != "none"]
        pick = hi
        for t in range(lo, hi + 1):
            hits = sum(ints[i] >= t for i in benign)
            ok = (wilson_upper(hits, len(benign)) <= limits[axis] if use_bound
                  else hits / max(1, len(benign)) <= limits[axis] * THRESHOLD_MARGIN)
            if ok:
                pick = t
                break
        chosen[key] = pick
        report[key] = {
            "rule": "upper confidence bound" if use_bound else "point estimate with margin",
            "benign_rows": len(benign), "positive_rows": len(positive),
            "fp": round(sum(ints[i] >= pick for i in benign) / max(1, len(benign)), 4),
            "recall": round(sum(ints[i] >= pick for i in positive) / max(1, len(positive)), 4),
        }
    chosen["injection_watch_at"] = max(10, chosen["injection_drop_at"] // 2)
    for key, axis, act_key in VERIFY_RULES:
        ints = np.round(100 * scores[axis]).astype(int)
        positive = [i for i, r in enumerate(rows) if r["source"] == "user_turn" and r["labels"][axis] != "none"]
        ordinary = [i for i, r in enumerate(rows) if is_ordinary_question(r)]
        pick = VERIFY_MIN
        for t in range(chosen[act_key], VERIFY_MIN - 1, -1):
            if sum(ints[i] >= t for i in positive) / max(1, len(positive)) >= VERIFY_RECALL:
                pick = t
                break
        chosen[key] = pick
        report[key] = {
            "positive_rows": len(positive),
            "recall": round(sum(ints[i] >= pick for i in positive) / max(1, len(positive)), 4),
            # Share of ordinary questions the judge would see (cost, not refusals).
            "judge_rate_ordinary": round(sum(ints[i] >= pick for i in ordinary) / max(1, len(ordinary)), 4),
        }
    return chosen, report


MODERATION_CUTOFFS = (0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5,
                      0.6, 0.7, 0.8, 0.9)


def choose_moderation_cutoff(rows: List[dict], harm_ints: Sequence[int], top_scores: Sequence[float],
                             th: Dict[str, int]) -> Tuple[Optional[float], Dict[str, Any]]:
    """Lowest cut-off on moderation's highest category score whose estimated end-to-end decline rate
    for ordinary questions (direct declines + judge rate x flag rate) stays under 75% of the gate's
    harm false-positive limit; the flag rate is measured on all held-out ordinary questions, not only
    the few in the verify band (SEC-035a)."""
    from eval_context_defender import HARM_FP

    ordinary = [i for i, r in enumerate(rows) if is_ordinary_question(r)]
    harmful = [i for i, r in enumerate(rows) if r["source"] == "user_turn" and r["labels"]["harm"] != "none"]
    n_ord = max(1, len(ordinary))
    direct = sum(harm_ints[i] >= th["harm_decline_at"] for i in ordinary) / n_ord
    in_band = lambda i: th["harm_verify_at"] <= harm_ints[i] < th["harm_decline_at"]  # noqa: E731
    judge_rate = sum(in_band(i) for i in ordinary) / n_ord
    budget = HARM_FP * THRESHOLD_MARGIN - direct
    pick: Optional[float] = None
    for cutoff in MODERATION_CUTOFFS:
        flag_rate = sum(top_scores[i] >= cutoff for i in ordinary) / n_ord
        if judge_rate * flag_rate <= budget:
            pick = cutoff
            break

    def recall(cutoff: Optional[float]) -> float:
        hits = sum(harm_ints[i] >= th["harm_decline_at"] or (in_band(i) and cutoff is not None and top_scores[i] >= cutoff)
                   for i in harmful)
        return round(hits / max(1, len(harmful)), 4)

    report = {
        "ordinary_rows": len(ordinary), "harmful_rows": len(harmful),
        "direct_decline_rate_ordinary": round(direct, 4), "judge_rate_ordinary": round(judge_rate, 4),
        "flag_rate_ordinary": None if pick is None else round(sum(top_scores[i] >= pick for i in ordinary) / n_ord, 4),
        "estimated_decline_rate_ordinary": None if pick is None else round(
            direct + judge_rate * sum(top_scores[i] >= pick for i in ordinary) / n_ord, 4),
        "harm_recall_end_to_end": recall(pick),
    }
    return pick, report


def _cached_moderation(texts: List[str], cache_path: Optional[Path]) -> List[float]:
    """Highest moderation category score per text; with a cache file of {sha256: score}, texts already
    scored are not sent again."""
    from app.security.judge import moderation_scores

    cache: Dict[str, float] = {}
    if cache_path and cache_path.is_file():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    keys = [hashlib.sha256(t.encode("utf-8")).hexdigest() for t in texts]
    todo = [i for i, k in enumerate(keys) if k not in cache]
    for i, (_, top) in zip(todo, moderation_scores([texts[i] for i in todo])):
        cache[keys[i]] = top
    if cache_path:
        cache_path.write_text(json.dumps(cache), encoding="utf-8")
    return [cache[k] for k in keys]


def set_moderation_cutoff(model_dir: Path, data_path: Path, seed: int = 13,
                          cache_path: Optional[Path] = None) -> Dict[str, Any]:
    """Choose the harm judge's moderation cut-off for a trained model on its held-out user turns
    (sends their text to OpenAI's moderation endpoint) and write it to metadata.json."""
    from app.security.thresholds import merged
    from eval_context_defender import load_runtime_model

    meta_path = model_dir / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    _, cal_rows = split_for_calibration(load_rows(data_path), meta.get("seed", seed))
    turns = [r for r in cal_rows if r["source"] == "user_turn"]
    model = load_runtime_model(model_dir)
    harm_ints = [model.score_text(r["text"], r["source"])[0]["harm"] for r in turns]
    tops = _cached_moderation([r["text"] for r in turns], cache_path)
    cutoff, report = choose_moderation_cutoff(turns, harm_ints, tops, merged(meta.get("thresholds")))
    meta["harm_moderation_cutoff"], meta["moderation_cutoff_selection"] = cutoff, report
    meta.pop("test", None)
    meta.pop("test_with_judge", None)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def reselect_thresholds(model_dir: Path, data_path: Path, seed: int = 13) -> Dict[str, Any]:
    """Re-run threshold selection for a trained model on its own held-out calibration rows
    (deterministic split), without retraining; rewrites metadata.json["thresholds"]."""
    from eval_context_defender import load_runtime_model

    meta_path = model_dir / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if not meta.get("finetuned_encoder"):
        raise SystemExit("Only fine-tuned models have held-out calibration rows to choose thresholds on")
    _, cal_rows = split_for_calibration(load_rows(data_path), meta.get("seed", seed))
    model = load_runtime_model(model_dir)
    scores = {a: np.zeros(len(cal_rows)) for a in AXES}
    for i, row in enumerate(cal_rows):
        _, _, pooled = model.score_text(row["text"], row["source"])
        for a in AXES:
            scores[a][i] = pooled[a]
    thresholds, report = select_thresholds(cal_rows, scores)
    meta["thresholds"], meta["threshold_selection"] = thresholds, report
    meta.pop("test", None)  # the gate result no longer matches these thresholds
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def train(
    data_path: Path,
    out_dir: Path,
    version: str,
    encoder_name: str = DEFAULT_ENCODER,
    seed: int = 13,
    encoder=None,
    tfidf_only: bool = False,
    chunk_size: int = 200,
    chunk_stride: int = 150,
    combine_features: bool = True,
    c: float = 10.0,
    finetune: bool = False,
    epochs: int = 2,
    hard_negative_weight: float = HARD_NEGATIVE_WEIGHT,
) -> dict:
    import app.security.context_defender as cd

    cd.set_chunking(chunk_size, chunk_stride)
    rows = load_rows(data_path)
    vocabulary = build_vocabulary_from_texts([r["text"] for r in rows])

    use_minilm = encoder is not None and not tfidf_only
    tokenizer = getattr(encoder, "tokenizer", None) if use_minilm else None
    finetune = finetune and use_minilm

    if finetune:
        fit_rows, cal_rows = split_for_calibration(rows, seed)
    else:
        fit_rows, cal_rows = rows, []
    chunks, routes, chunk_labels, weights, _ = build_chunks(fit_rows, tokenizer, vocabulary, use_minilm,
                                                            hard_negative_weight)
    cal_chunks, cal_routes, cal_labels, _, cal_owners = build_chunks(cal_rows, tokenizer, vocabulary, use_minilm)

    if finetune:
        idx = [i for i, r in enumerate(routes) if r == "minilm"]
        finetune_encoder(encoder, [chunks[i] for i in idx], {a: [chunk_labels[a][i] for i in idx] for a in AXES},
                         seed, epochs=epochs, weights={a: [weights[a][i] for i in idx] for a in AXES})

    # Bounded, sparse character n-grams: unbounded n-grams over thousands of long texts don't fit in memory.
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                                 max_features=50_000, sublinear_tf=True)
    vectorizer.fit(chunks)

    def encode(chunk_list, route_list):
        """Features per route, each chunk encoded once and reused for all four heads."""
        index: Dict[str, List[int]] = {"minilm": [], "tfidf": []}
        for i, route in enumerate(route_list):
            index[route].append(i)
        feats = {route: featurize([chunk_list[i] for i in ids], route, encoder, vectorizer, combine=combine_features)
                 for route, ids in index.items() if ids}
        return index, feats

    index, features = encode(chunks, routes)
    cal_index, cal_features = encode(cal_chunks, cal_routes) if finetune else ({}, {})

    heads: Dict[str, Dict[str, Any]] = {axis: {} for axis in AXES}
    for axis in AXES:
        for route, ids in index.items():
            if not ids:
                continue
            y = [chunk_labels[axis][i] for i in ids]
            if finetune and cal_index.get(route):
                y_cal = [cal_labels[axis][i] for i in cal_index[route]]
                heads[axis][route] = fit_prefit_calibrated(features[route], y, cal_features[route], y_cal, seed, c,
                                                           sample_weight=[weights[axis][i] for i in ids])
            else:
                # Calibration does its own cross-validation, so every row is used for fitting (SEC-024).
                heads[axis][route] = fit_head_features(features[route], y, seed, c=c,
                                                       sample_weight=[weights[axis][i] for i in ids])

    thresholds: Dict[str, int] = {}
    threshold_report: Dict[str, Any] = {}
    if cal_rows:
        scores = row_scores(heads, cal_index, cal_features, cal_owners, len(cal_rows))
        thresholds, threshold_report = select_thresholds(cal_rows, scores)

    out_dir.mkdir(parents=True, exist_ok=True)
    if finetune:
        encoder.save(str(out_dir / cd.FINETUNED_ENCODER_DIR))
    bundle = {
        "heads": heads,
        "vectorizer": vectorizer,
        "vocabulary": sorted(vocabulary),
        "tokenizer": None,
        "encoder": None,
        "use_minilm": use_minilm,
        "finetuned_encoder": finetune,
        "chunking": {"size": chunk_size, "stride": chunk_stride},
        "combine_features": bool(combine_features and use_minilm),
        "c": c,
    }
    joblib.dump(bundle, out_dir / "heads.joblib")

    meta = {
        "version": version,
        "encoder_name": encoder_name,
        "finetuned_encoder": finetune,
        "data_hash": data_hash(data_path),
        "use_minilm": use_minilm,
        "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rows": len(rows),
        "calibration_rows": len(cal_rows),
        "hard_negative_weight": hard_negative_weight,
        "seed": seed,
        # Policy thresholds for this model (app/security/thresholds.py); defaults when empty.
        "thresholds": thresholds,
        "threshold_selection": threshold_report,
        "sources": sorted({r["source"] for r in rows}),
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=BACKEND / "data/security/train.jsonl")
    parser.add_argument("--out", type=Path, default=BACKEND / "models/security")
    parser.add_argument("--version", default=None)
    # ContextDefender's encoder is fine-tuned from the 12-layer MiniLM since 2026-10-13 (injection recall
    # 0.939 -> 0.973 at the same false-positive limit, ~1.7x CPU per page); alignment keeps the base L6.
    parser.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L12-v2")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--tfidf-only", action="store_true")
    # 96/64 since 2026-10-12: on the same data, injection recall 0.868 -> 0.926 against 200/150 (short
    # injections are diluted in long windows), WildChat-page false positives halved, CPU time per page unchanged.
    parser.add_argument("--chunk-size", type=int, default=96)
    parser.add_argument("--chunk-stride", type=int, default=64)
    # Defaults chosen on 2026-10-11: TF-IDF n-grams on the MiniLM route with C=10 fit embedded
    # injections far better than the embedding alone with C=1 (test injection recall 0.57 -> 0.71).
    parser.add_argument("--embedding-only", dest="combine_features", action="store_false",
                        help="MiniLM route uses the embedding alone (no TF-IDF n-grams)")
    parser.add_argument("--C", type=float, default=10.0, help="inverse regularisation strength of the heads")
    parser.add_argument("--finetune", action="store_true",
                        help="fine-tune MiniLM on 85%% of the rows; calibrate the heads on the other 15%%")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--choose-moderation-cutoff", action="store_true",
                        help="only choose the harm judge's moderation cut-off for the model in --out "
                             "(sends held-out user turns to OpenAI's moderation endpoint)")
    parser.add_argument("--moderation-cache", type=Path, default=None,
                        help="JSON file of {sha256: score} so texts already scored aren't sent again")
    parser.add_argument("--reselect-thresholds", action="store_true",
                        help="only re-choose the policy thresholds of the model in --out (no training)")
    parser.add_argument("--hard-negative-weight", type=float, default=HARD_NEGATIVE_WEIGHT,
                        help="weight of look-alike benign rows on the axis they imitate (1 = off)")
    args = parser.parse_args(argv)

    if args.choose_moderation_cutoff:
        meta = set_moderation_cutoff(args.out, args.data, args.seed, args.moderation_cache)
        print(json.dumps({"harm_moderation_cutoff": meta["harm_moderation_cutoff"],
                          "selection": meta["moderation_cutoff_selection"]}, indent=2))
        return 0
    if args.reselect_thresholds:
        meta = reselect_thresholds(args.out, args.data, args.seed)
        print(json.dumps({"thresholds": meta["thresholds"], "selection": meta["threshold_selection"]}, indent=2))
        return 0
    version = args.version or f"cd-{datetime.now(timezone.utc).strftime('%Y-%m-%d')}-1"
    encoder = None
    if not args.tfidf_only:
        try:
            import torch
            from sentence_transformers import SentenceTransformer

            # Training may use the Apple or CUDA GPU; serving always uses the CPU.
            device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
            encoder = SentenceTransformer(args.encoder, device=device)
        except Exception as exc:
            print(f"Encoder unavailable ({exc}); training TF-IDF-only heads")
            args.tfidf_only = True

    meta = train(args.data, args.out, version, args.encoder, args.seed, encoder=encoder, tfidf_only=args.tfidf_only,
                 chunk_size=args.chunk_size, chunk_stride=args.chunk_stride,
                 combine_features=args.combine_features, c=args.C, finetune=args.finetune, epochs=args.epochs,
                 hard_negative_weight=args.hard_negative_weight)
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
