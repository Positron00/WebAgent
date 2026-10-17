"""The intent-alignment classifier, phase B (checkpoint2-intent SEC-123 – SEC-127).

Reads (intent, ``"<action>: <text>"``) and gives P(aligned) and a 0-100 risk. Two kinds ship
(``metadata.json["kind"]``):

- ``cross_encoder``: ``encoder/`` (Hugging Face ``save_pretrained`` of an ``AutoModel`` and its
  tokenizer), ``heads.pt`` (``{"aligned": Linear(hidden, 1) state_dict, "risk": Linear(hidden, 3)
  state_dict, "hidden", "pooling": "mean", "max_length": 128}``); the pair is tokenized as
  ``tokenizer(intent, call_text, truncation=True, max_length=128)`` and mean-pooled over the mask.
- ``siamese`` (SEC-124 baseline): ``heads.joblib`` (``{"aligned": LogisticRegression, "risk":
  LogisticRegression}`` with risk classes none/low/high) on ``[u, v, |u-v|, u*v]`` of ContextDefender's
  normalised encoder embeddings of the intent (u) and the call (v).

Both have ``calibrators.joblib`` (``{"aligned": IsotonicRegression, "risk": IsotonicRegression}``):
``aligned`` maps the raw P(aligned) to a calibrated one; ``risk`` maps the raw
``r = P(low) * 0.5 + P(high)`` to the calibrated probability that the call's risk is not none, and
the returned risk is ``round(100 * that)``. ``metadata.json`` holds ``kind``, ``version``
(``ac-YYYY-MM-DD-N``), ``base_encoder``, ``max_length``, ``thresholds`` (0-100, the intent keys of
``thresholds.DEFAULT_THRESHOLDS``), ``uncertain_band`` ([lo, hi] probabilities, SEC-127),
``data_hash`` and, after the gate, ``test.passed``. The runtime never uses a model whose gate has
not passed (SEC-125).
"""
from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from app.core.config import BACKEND_DIR, settings
from app.core.model_lock import lock_for

logger = logging.getLogger(__name__)

KINDS = ("cross_encoder", "siamese")
_FAIL_BACKOFF_SEC = 60.0
# Intent keys a model's metadata may set (the checkpoint-1 keys belong to ContextDefender).
INTENT_THRESHOLD_KEYS = ("rewrite_at", "subgoal_align_at", "search_align_at", "drift_align_at",
                         "risk_verify_at", "risk_decline_at")

_lock = threading.Lock()
_cached: Optional["AlignmentModel"] = None
_failed_until = 0.0


def model_dir() -> Path:
    cfg = getattr(settings, "SECURITY_GUARD", None)
    p = Path(getattr(cfg, "alignment_model_dir", None) or "models/security/alignment")
    return p if p.is_absolute() else Path(BACKEND_DIR) / p


def read_metadata(d: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(((d or model_dir()) / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def gate_passed(meta: Optional[Dict[str, Any]]) -> bool:
    return bool(meta and (meta.get("test") or {}).get("passed") is True)


def _calibrate(calibrator: Any, raw: float) -> float:
    if calibrator is None:
        return float(np.clip(raw, 0.0, 1.0))
    return float(np.clip(calibrator.predict(np.asarray([raw], dtype=float))[0], 0.0, 1.0))


class AlignmentModel:
    """A loaded, calibrated classifier; ``score_pair`` is deterministic and runs on CPU."""

    def __init__(self, kind: str, metadata: Dict[str, Any], calibrators: Dict[str, Any], *,
                 encoder: Any = None, tokenizer: Any = None, heads: Any = None, max_length: int = 128):
        self.kind = kind
        self.metadata = metadata
        self.version = str(metadata.get("version") or "")
        self._calibrators = calibrators or {}
        self._encoder = encoder
        self.tokenizer = tokenizer
        self._heads = heads
        self._max_length = int(max_length)

    def _raw_cross(self, intent: str, call_text: str) -> Tuple[float, np.ndarray]:
        import torch

        with lock_for(self), torch.no_grad():
            batch = self.tokenizer(intent, call_text, truncation=True, max_length=self._max_length,
                                   return_tensors="pt")
            hidden = self._encoder(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            aligned = torch.sigmoid(self._heads["aligned"](pooled))[0, 0].item()
            risk = torch.softmax(self._heads["risk"](pooled), dim=-1)[0].numpy()
        return float(aligned), np.asarray(risk, dtype=float)

    def _raw_siamese(self, intent: str, call_text: str) -> Tuple[float, np.ndarray]:
        u, v = siamese_embed([intent, call_text], self._encoder)
        x = siamese_features(u[None, :], v[None, :])
        aligned = float(self._heads["aligned"].predict_proba(x)[0, -1])
        return aligned, np.asarray(self._heads["risk"].predict_proba(x)[0], dtype=float)

    def score_pair(self, intent: str, call_text: str) -> Tuple[float, int]:
        """(calibrated P(aligned), risk 0-100) for the intent and ``"<action>: <text>"``."""
        raw = self._raw_cross if self.kind == "cross_encoder" else self._raw_siamese
        p_aligned, risk_probs = raw(intent or "", call_text or "")
        r = float(risk_probs[1]) * 0.5 + float(risk_probs[2])
        return (_calibrate(self._calibrators.get("aligned"), p_aligned),
                int(round(100 * _calibrate(self._calibrators.get("risk"), r))))

    def uncertain_band(self) -> Tuple[float, float]:
        """The P(aligned) range the judge decides (SEC-127); the whole range when not set."""
        band = self.metadata.get("uncertain_band") or (0.0, 1.0)
        try:
            lo, hi = float(band[0]), float(band[1])
        except (TypeError, ValueError, IndexError):
            return 0.0, 1.0
        return (min(lo, hi), max(lo, hi))

    def thresholds(self) -> Dict[str, int]:
        raw = self.metadata.get("thresholds") or {}
        return {k: int(raw[k]) for k in INTENT_THRESHOLD_KEYS if isinstance(raw.get(k), (int, float))}


def _load_cross_encoder(d: Path, meta: Dict[str, Any], calibrators: Dict[str, Any]) -> Optional[AlignmentModel]:
    import torch
    from transformers import AutoModel, AutoTokenizer

    enc_dir = d / "encoder"
    if not (d / "heads.pt").is_file() or not (enc_dir / "config.json").is_file():
        return None
    blob = torch.load(d / "heads.pt", map_location="cpu", weights_only=True)
    if blob.get("pooling", "mean") != "mean":
        raise ValueError(f"unsupported pooling {blob.get('pooling')!r}")
    hidden = int(blob["hidden"])
    heads = {"aligned": torch.nn.Linear(hidden, 1), "risk": torch.nn.Linear(hidden, 3)}
    for name, layer in heads.items():
        layer.load_state_dict(blob[name])
        layer.eval()
    encoder = AutoModel.from_pretrained(str(enc_dir), local_files_only=True).to("cpu").eval()
    tokenizer = AutoTokenizer.from_pretrained(str(enc_dir), local_files_only=True)
    max_length = int(blob.get("max_length") or meta.get("max_length") or 128)
    return AlignmentModel("cross_encoder", meta, calibrators, encoder=encoder, tokenizer=tokenizer,
                          heads=heads, max_length=max_length)


def _defender_encoder():
    """The encoder ContextDefender loads: its fine-tuned one when present, else the base MiniLM."""
    from app.security import context_defender
    from app.security.encoder import get_base_encoder, get_encoder_at

    cd_meta = read_metadata(context_defender.model_dir()) or {}
    if cd_meta.get("finetuned_encoder") and context_defender.finetuned_encoder_present():
        return get_encoder_at(context_defender.model_dir() / context_defender.FINETUNED_ENCODER_DIR)[0]
    return get_base_encoder()[0]


def siamese_embed(texts, encoder: Any = None) -> np.ndarray:
    """Normalised embeddings (one row per text) from ContextDefender's encoder, as the baseline
    uses at runtime; the trainer calls this so features can't drift (SEC-124). Takes the lock."""
    encoder = encoder if encoder is not None else _defender_encoder()
    with lock_for(encoder):
        return np.asarray(encoder.encode(list(texts), normalize_embeddings=True), dtype=float)


def siamese_features(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """``[u, v, |u-v|, u*v]`` per row of intent (u) and call (v) embeddings."""
    return np.concatenate([u, v, np.abs(u - v), u * v], axis=1)


def _load_siamese(d: Path, meta: Dict[str, Any], calibrators: Dict[str, Any]) -> Optional[AlignmentModel]:
    import joblib

    if not (d / "heads.joblib").is_file():
        return None
    encoder = _defender_encoder()
    if encoder is None:
        raise RuntimeError("the baseline needs ContextDefender's encoder, which is not available")
    return AlignmentModel("siamese", meta, calibrators, encoder=encoder, heads=joblib.load(d / "heads.joblib"))


def load(d: Optional[Path] = None, *, require_gate: bool = True) -> Optional[AlignmentModel]:
    """The model in ``d``; None when missing, incomplete, of an unknown kind or (with
    ``require_gate``, always so at runtime) not gated. The gate script passes ``require_gate=False``."""
    import joblib

    d = Path(d) if d is not None else model_dir()
    meta = read_metadata(d)
    if meta is None or not (d / "calibrators.joblib").is_file():
        return None
    if require_gate and not gate_passed(meta):
        logger.warning("Alignment classifier %s has not passed its gate; phase A signals are used",
                       meta.get("version", "?"))
        return None
    kind = meta.get("kind")
    if kind not in KINDS:
        logger.warning("Alignment classifier kind %r unknown; phase A signals are used", kind)
        return None
    calibrators = joblib.load(d / "calibrators.joblib")
    loader = _load_cross_encoder if kind == "cross_encoder" else _load_siamese
    return loader(d, meta, calibrators)


def get_model() -> Optional[AlignmentModel]:
    """The gated model of this worker, loaded once; a failed load is retried after a backoff."""
    global _cached, _failed_until
    now = time.time()
    if _cached is not None or now < _failed_until:
        return _cached
    with _lock:
        if _cached is not None:
            return _cached
        try:
            model = load()
        except Exception as exc:  # noqa: BLE001 - phase A signals are used instead (SEC-126)
            logger.error("Could not load the alignment classifier: %s", exc)
            model = None
        if model is None:
            _failed_until = now + _FAIL_BACKOFF_SEC
            return None
        _cached = model
        return _cached


def current_thresholds() -> Dict[str, int]:
    """The loaded model's intent thresholds, clamped by ``thresholds.merged``; {} without a model.
    Never loads a model."""
    from app.security.thresholds import merged

    if _cached is None:
        return {}
    full = merged(_cached.thresholds())
    return {k: full[k] for k in INTENT_THRESHOLD_KEYS}


def band_outcome(p_aligned: Optional[float]) -> str:
    """SEC-127: "judge" inside the loaded model's uncertain band, "allow" above, "decline" below;
    "judge" without a model or a score."""
    if _cached is None or p_aligned is None:
        return "judge"
    lo, hi = _cached.uncertain_band()
    return "allow" if p_aligned > hi else "decline" if p_aligned < lo else "judge"


def set_model_for_tests(model: Optional[AlignmentModel]) -> None:
    global _cached, _failed_until
    with _lock:
        _cached = model
        _failed_until = 0.0


def warm_up() -> None:
    try:
        model = get_model()
        if model is not None:
            model.score_pair("warm-up request", "web_search: warm-up query")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Alignment classifier warm-up failed: %s", exc)
