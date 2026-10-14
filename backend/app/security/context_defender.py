"""ContextDefender: score context items on four risk axes (SEC-020–SEC-026)."""
from __future__ import annotations

import ast
import hashlib
import json
import logging
import re
import threading
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import yaml

from app.core.config import BACKEND_DIR, settings
from app.core.metrics import GUARD_CONTEXT_SCORE
from app.core.model_lock import lock_for
from app.security.patterns import scan

from app.security.types import ContextItem, ItemScores, Source

logger = logging.getLogger(__name__)

AXES = ("injection", "harm", "sensitive_data", "jailbreak")
LABEL_LEVELS = ("none", "low", "high")
CHUNK_SIZE = 200
CHUNK_STRIDE = 150
CACHE_MAX = 10_000
_FAIL_BACKOFF_SEC = 60.0
DEFAULT_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
_BASE64_HEX_RE = re.compile(r"[A-Za-z0-9+/=]{24,}")

_lock = threading.Lock()
_cached_model: Optional["ContextDefenderModel"] = None
_failed_until: float = 0.0
_score_cache: "OrderedDict[Tuple[str, str], ItemScores]" = OrderedDict()


@dataclass(frozen=True)
class DefenderMetadata:
    version: str
    encoder_name: str
    data_hash: str
    thresholds: Dict[str, int] = field(default_factory=dict)  # chosen at training (thresholds.py)
    moderation_cutoff: Optional[float] = None  # harm judge cut-off on moderation scores (SEC-035a)


class IsotonicBinaryHead:
    """Kept so older pickles still load; new artifacts store CalibratedClassifierCV heads."""

    def __init__(self, classifier, calibrator):
        self._clf = classifier
        self._iso = calibrator

    def predict_proba(self, x):
        raw = self._clf.predict_proba(x)[:, 1]
        pos = np.clip(self._iso.predict(raw), 0.0, 1.0)
        return np.column_stack([1.0 - pos, pos])


def featurize(texts: Sequence[str], route: str, encoder, vectorizer, combine: bool = False):
    """Features for one route (SEC-022); shared by training and runtime so they can't drift.

    With ``combine`` the MiniLM route also carries the TF-IDF n-grams, so a phrase-level cue isn't
    lost when the injection is only part of a chunk's meaning.
    """
    if route == "minilm":
        if encoder is None:
            raise RuntimeError("MiniLM route needs the encoder")
        with lock_for(encoder):
            emb = np.asarray(encoder.encode(list(texts), normalize_embeddings=True))
        if combine and vectorizer is not None:
            from scipy.sparse import csr_matrix, hstack

            return hstack([csr_matrix(emb), vectorizer.transform(list(texts))]).tocsr()
        return emb
    if vectorizer is None:
        raise RuntimeError("TF-IDF vectorizer not loaded")
    return vectorizer.transform(list(texts))  # sparse; the heads accept sparse input


class ContextDefenderModel:
    def __init__(self, bundle: Dict[str, Any], metadata: DefenderMetadata):
        self._bundle = bundle
        self._encoder = bundle.get("encoder")
        self.metadata = metadata
        self._vocabulary = set(bundle.get("vocabulary") or ())
        self._vectorizer = bundle.get("vectorizer")
        self._heads = bundle["heads"]
        self.tokenizer = bundle.get("tokenizer")
        # The window the heads were trained with; every loader goes through here.
        chunking = bundle.get("chunking") or {"size": 200, "stride": 150}
        set_chunking(chunking["size"], chunking["stride"])

    def _route(self, chunk: str) -> str:
        if not self._bundle.get("use_minilm") or self._encoder is None:
            return "tfidf"
        return route_chunk(chunk, self._vocabulary)

    def _chunk_risks(self, chunks: Sequence[str]) -> List[Dict[str, float]]:
        """P(risk present) per axis for each chunk; binary heads (none vs. low/high).

        All chunks of a text go through the encoder in one batch: encoding them one by one cost
        about 90 ms per window on CPU, so a long web page took seconds per hop.
        """
        routes = [self._route(c) for c in chunks]
        combine = bool(self._bundle.get("combine_features"))
        cache: Dict[str, Any] = {}
        out: List[Dict[str, float]] = [{} for _ in chunks]
        for axis in AXES:
            axis_heads = self._heads.get(axis) or {}
            if not axis_heads:
                for risk in out:
                    risk[axis] = 0.0
                continue
            groups: Dict[str, List[int]] = defaultdict(list)
            for i, route in enumerate(routes):
                groups[route if route in axis_heads else next(iter(axis_heads))].append(i)
            for use_route, idx in groups.items():
                if use_route not in cache:
                    cache[use_route] = featurize(list(chunks), use_route, self._encoder, self._vectorizer,
                                                 combine=combine)
                probs = axis_heads[use_route].predict_proba(cache[use_route][idx])[:, -1]
                for i, p in zip(idx, probs):
                    out[i][axis] = float(p)
        return out

    def score_text(self, text: str, source: Source) -> Tuple[Dict[str, int], Tuple[Dict[str, int], ...], Dict[str, float]]:
        """Return pooled axis ints, per-chunk axis ints, and the pooled (max) risk per axis."""
        chunks = chunk_text(text, self.tokenizer) or [text]
        chunk_scores: List[Dict[str, int]] = []
        pooled = {a: 0.0 for a in AXES}
        for risk in self._chunk_risks(chunks):
            chunk_scores.append({a: round(100 * risk[a]) for a in AXES})
            for a in AXES:
                pooled[a] = max(pooled[a], risk[a])
        return {a: round(100 * pooled[a]) for a in AXES}, tuple(chunk_scores), pooled


def route_chunk(text: str, vocabulary: set) -> str:
    """TF-IDF route for code, base64-heavy, or mostly OOV text (SEC-022)."""
    stripped = text.strip()
    if stripped:
        try:
            tree = ast.parse(stripped)
            if len(tree.body) >= 2:
                return "tfidf"
        except SyntaxError:
            pass
    alnum = sum(1 for c in stripped if c.isalnum() or c in "+/=")
    if stripped and alnum / len(stripped) >= 0.30:
        match = _BASE64_HEX_RE.search(stripped)
        if match and sum(c.isdigit() for c in match.group()) >= 2:
            return "tfidf"
    tokens = [t.lower() for t in re.findall(r"[A-Za-z0-9_]+", stripped)]
    if not tokens:
        return "tfidf"
    in_vocab = sum(1 for t in tokens if t in vocabulary)
    if in_vocab / len(tokens) < 0.50:
        return "tfidf"
    return "minilm"


def _units(text: str, tokenizer) -> Tuple[List[Any], Any]:
    """Token ids (with the MiniLM tokenizer) or words, plus a function that joins units back."""
    if tokenizer is None:
        return text.split(), lambda units: " ".join(units)
    with lock_for(tokenizer):
        ids = tokenizer.encode(text, add_special_tokens=False)

    def join(units):
        with lock_for(tokenizer):
            return tokenizer.decode(units, skip_special_tokens=True)

    return list(ids), join


def set_chunking(size: int, stride: int) -> None:
    """Use the window size the loaded model was trained with (stored in its bundle)."""
    global CHUNK_SIZE, CHUNK_STRIDE
    CHUNK_SIZE, CHUNK_STRIDE = int(size), int(stride)


def windows(n_units: int) -> List[Tuple[int, int]]:
    """[start, end) unit spans of CHUNK_SIZE units every CHUNK_STRIDE (SEC-023)."""
    if n_units <= CHUNK_SIZE:
        return [(0, n_units)]
    spans: List[Tuple[int, int]] = []
    start = 0
    while True:
        spans.append((start, min(start + CHUNK_SIZE, n_units)))
        if start + CHUNK_SIZE >= n_units:
            return spans
        start += CHUNK_STRIDE


def chunk_text(text: str, tokenizer) -> List[str]:
    """Chunks for scoring: MiniLM tokens when the tokenizer is loaded, else words."""
    units, join = _units(text, tokenizer)
    if len(units) <= CHUNK_SIZE:
        return [text]
    return [join(units[a:b]) for a, b in windows(len(units))]


def text_without_chunks(text: str, flagged: Sequence[int], tokenizer) -> str:
    """Remove the units covered by the flagged chunks, keeping every other unit once.

    Chunks overlap by 50 units, so joining kept chunks would repeat text; removing spans doesn't.
    """
    units, join = _units(text, tokenizer)
    spans = windows(len(units))
    drop = [False] * len(units)
    for idx in flagged:
        if 0 <= idx < len(spans):
            a, b = spans[idx]
            for i in range(a, b):
                drop[i] = True
    kept = [u for u, d in zip(units, drop) if not d]
    return join(kept).strip() if kept else ""


@lru_cache(maxsize=1)
def _source_weights() -> Dict[str, Dict[str, float]]:
    path = Path(BACKEND_DIR) / "config" / "security" / "source_weights.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    defaults = raw.get("defaults", {})
    out: Dict[str, Dict[str, float]] = {"defaults": defaults}
    for key, weights in raw.items():
        if key == "defaults":
            continue
        out[key] = {**defaults, **weights}
    return out


def threat_risk(source: Source, pooled: Dict[str, float]) -> Tuple[int, str]:
    """Source-weighted risk (SEC-025) from the pooled per-axis risk; top axis "none" when all are 0."""
    weights_table = _source_weights()
    w = weights_table.get(source, weights_table["defaults"])
    best_axis, best = "none", 0.0
    for axis in AXES:
        weighted = float(w.get(axis, 1.0)) * float(pooled.get(axis, 0.0))
        if weighted > best:
            best, best_axis = weighted, axis
    return round(100 * best), best_axis


def top_axis_from_ints(axes: Dict[str, int]) -> str:
    return max(AXES, key=lambda a: axes[a])


def model_dir() -> Path:
    cfg = getattr(settings, "SECURITY_GUARD", None)
    rel = cfg.model_dir if cfg else "models/security"
    p = Path(rel)
    return p if p.is_absolute() else Path(BACKEND_DIR) / p


def model_files_present(model_dir_path: Optional[Path] = None) -> bool:
    d = model_dir_path or model_dir()
    return (d / "heads.joblib").is_file() and (d / "metadata.json").is_file()


def gate_passed(model_dir_path: Optional[Path] = None) -> bool:
    try:
        meta = json.loads(((model_dir_path or model_dir()) / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return bool((meta.get("test") or {}).get("passed"))


FINETUNED_ENCODER_DIR = "encoder"


def finetuned_encoder_present(model_dir_path: Optional[Path] = None) -> bool:
    d = (model_dir_path or model_dir()) / FINETUNED_ENCODER_DIR
    return (d / "config.json").is_file() and any(d.glob("*.safetensors"))


def encoder_files_present(encoder_name: str = DEFAULT_ENCODER) -> bool:
    from app.services.triage_model import encoder_files_present as triage_encoder_present

    return triage_encoder_present(encoder_name)


def load_model(model_dir_path: Optional[Path] = None) -> Optional[ContextDefenderModel]:
    d = model_dir_path or model_dir()
    joblib_path = d / "heads.joblib"
    meta_path = d / "metadata.json"
    if not joblib_path.is_file() or not meta_path.is_file():
        return None

    meta_raw = json.loads(meta_path.read_text(encoding="utf-8"))
    metadata = DefenderMetadata(
        version=meta_raw["version"],
        encoder_name=meta_raw.get("encoder_name", DEFAULT_ENCODER),
        data_hash=meta_raw.get("data_hash", ""),
        thresholds=dict(meta_raw.get("thresholds") or {}),
        moderation_cutoff=meta_raw.get("harm_moderation_cutoff"),
    )
    bundle: Dict[str, Any] = joblib.load(joblib_path)
    bundle.setdefault("encoder", None)
    bundle.setdefault("tokenizer", None)

    if bundle.get("use_minilm"):
        from app.security.encoder import get_base_encoder, get_encoder_at

        if bundle.get("finetuned_encoder"):
            # Fine-tuned for this classifier; alignment keeps using the base encoder.
            encoder, tokenizer = get_encoder_at(d / FINETUNED_ENCODER_DIR)
        else:
            encoder, tokenizer = get_base_encoder()
        if encoder is None:
            # Without the encoder the MiniLM-route heads can't run; scores_available becomes false.
            logger.warning("ContextDefender's MiniLM encoder is not available")
            return None
        bundle["encoder"], bundle["tokenizer"] = encoder, tokenizer
    return ContextDefenderModel(bundle, metadata)


def get_model() -> Optional[ContextDefenderModel]:
    global _cached_model, _failed_until
    now = time.time()
    if _cached_model is not None:
        return _cached_model
    if now < _failed_until:
        return None
    with _lock:
        if _cached_model is not None:
            return _cached_model
        try:
            model = load_model()
        except Exception as exc:
            logger.error("Could not load ContextDefender: %s", exc)
            model = None
        if model is None:
            _failed_until = now + _FAIL_BACKOFF_SEC
            return None
        _cached_model = model
        return _cached_model


def current_thresholds() -> Dict[str, int]:
    """The loaded model's policy thresholds (defaults when none is loaded); never loads a model."""
    from app.security.thresholds import merged

    meta = getattr(_cached_model, "metadata", None)
    return merged(getattr(meta, "thresholds", None))


def current_moderation_cutoff() -> Optional[float]:
    """The loaded model's cut-off for the harm judge's moderation scores; None means use `flagged`."""
    return getattr(getattr(_cached_model, "metadata", None), "moderation_cutoff", None)


def set_model_for_tests(model: Optional[ContextDefenderModel]) -> None:
    global _cached_model, _failed_until
    with _lock:
        _cached_model = model
        _failed_until = 0.0
    _score_cache.clear()


def _cache_get(key: Tuple[str, str]) -> Optional[ItemScores]:
    if key in _score_cache:
        _score_cache.move_to_end(key)
        return _score_cache[key]
    return None


def _cache_put(key: Tuple[str, str], value: ItemScores) -> None:
    _score_cache[key] = value
    _score_cache.move_to_end(key)
    while len(_score_cache) > CACHE_MAX:
        _score_cache.popitem(last=False)


def _unavailable_scores(pattern_hits: frozenset) -> ItemScores:
    zeros = {a: 0 for a in AXES}
    return ItemScores(
        axes=zeros,
        top_axis="none",
        threat_risk=0,
        pattern_hits=pattern_hits,
        chunks=(),
        scores_available=False,
        model_version="",
    )


def score(items: Sequence[ContextItem]) -> List[ItemScores]:
    """Score context items: patterns first, then classifier when available."""
    model = get_model()
    version = model.metadata.version if model else ""
    results: List[ItemScores] = []

    for item in items:
        pattern_hits = frozenset(scan(item.text))
        if item.source == "image":
            results.append(_unavailable_scores(pattern_hits))
            continue

        text_key = hashlib.sha256(item.text.encode("utf-8")).hexdigest()
        cache_key = (text_key, version)
        cached = _cache_get(cache_key)
        if cached is not None:
            results.append(
                ItemScores(
                    axes=cached.axes,
                    top_axis=cached.top_axis,
                    threat_risk=cached.threat_risk,
                    pattern_hits=pattern_hits,
                    chunks=cached.chunks,
                    scores_available=cached.scores_available,
                    model_version=cached.model_version,
                )
            )
            continue

        if model is None:
            results.append(_unavailable_scores(pattern_hits))
            continue

        try:
            axes_int, chunks, pooled = model.score_text(item.text, item.source)
        except Exception as exc:
            # An incompatible or broken model must not break the hop: patterns still apply (SEC-061).
            logger.error("ContextDefender scoring failed: %s", exc)
            results.append(_unavailable_scores(pattern_hits))
            continue
        tr, top = threat_risk(item.source, pooled)
        item_scores = ItemScores(
            axes=axes_int,
            top_axis=top,
            threat_risk=tr,
            pattern_hits=pattern_hits,
            chunks=chunks,
            scores_available=True,
            model_version=version,
        )
        _cache_put(cache_key, item_scores)
        # Scores of freshly scored items (a cached repeat isn't counted twice)
        for axis, value in axes_int.items():
            GUARD_CONTEXT_SCORE.labels(axis=axis, source=item.source).observe(value)
        results.append(
            ItemScores(
                axes=item_scores.axes,
                top_axis=item_scores.top_axis,
                threat_risk=item_scores.threat_risk,
                pattern_hits=pattern_hits,
                chunks=item_scores.chunks,
                scores_available=True,
                model_version=version,
            )
        )
    return results


def current_tokenizer():
    """Tokenizer of the loaded model (None when words are used), for chunk removal in hop_guard."""
    model = get_model()
    return model.tokenizer if model is not None else None


def warm_up() -> None:
    start = time.perf_counter()
    model = get_model()
    if model is None:
        logger.warning("ContextDefender warm-up skipped: model not available")
        return
    try:
        score([ContextItem("user_turn", "warm-up sentence for context defender")])
    except Exception as exc:
        logger.warning("ContextDefender warm-up failed: %s", exc)
        return
    logger.info(
        "ContextDefender warm-up complete (version=%s, %.0f ms)",
        model.metadata.version,
        (time.perf_counter() - start) * 1000,
    )
