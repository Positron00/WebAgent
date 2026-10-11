"""Load sentence-transformers encoder + sklearn classifier for triage."""
from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np

from app.core.config import BACKEND_DIR, settings

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_cached_model: Optional["MiniLMTriageModel"] = None
_failed_until: float = 0.0
_FAIL_BACKOFF_SEC = 60.0


@dataclass(frozen=True)
class TriageMetadata:
    version: str
    encoder_name: str
    threshold: float
    data_hash: str
    categories: Dict[str, str]


class MiniLMTriageModel:
    def __init__(self, encoder, classifier, centroids: np.ndarray, category_ids: list, metadata: TriageMetadata):
        self._encoder = encoder
        self._classifier = classifier
        self._centroids = centroids
        self._category_ids = category_ids
        self.metadata = metadata

    def predict_proba(self, text: str) -> float:
        emb = self._encoder.encode([text], normalize_embeddings=True)
        proba = self._classifier.predict_proba(emb)[0]
        classes = list(self._classifier.classes_)
        idx = classes.index("agentic") if "agentic" in classes else int(np.argmax(proba))
        return float(proba[idx])

    def nearest_category(self, text: str) -> str:
        if self._centroids.size == 0:
            return "general"
        emb = self._encoder.encode([text], normalize_embeddings=True)[0]
        sims = self._centroids @ emb
        cat_id = self._category_ids[int(np.argmax(sims))]
        return self.metadata.categories.get(cat_id, cat_id)


def model_dir() -> Path:
    cfg = getattr(settings, "TRIAGE", None)
    rel = cfg.model_dir if cfg else "models/triage"
    p = Path(rel)
    return p if p.is_absolute() else Path(BACKEND_DIR) / p


def model_files_present(model_dir_path: Optional[Path] = None) -> bool:
    d = model_dir_path or model_dir()
    return (d / "classifier.joblib").is_file() and (d / "metadata.json").is_file()


def gate_passed(model_dir_path: Optional[Path] = None) -> bool:
    """True if metadata.json records a passed release gate (REQ-023)."""
    import json

    try:
        meta = json.loads(((model_dir_path or model_dir()) / "metadata.json").read_text())
    except (OSError, ValueError):
        return False
    return bool((meta.get("test") or {}).get("passed"))


def encoder_cache_dirs() -> list:
    """Folders where sentence-transformers looks for a downloaded encoder, most specific first."""
    dirs = []
    for env_name in ("SENTENCE_TRANSFORMERS_HOME", "HF_HUB_CACHE"):
        if os.getenv(env_name):
            dirs.append(Path(os.environ[env_name]))
    hf_home = os.getenv("HF_HOME")
    dirs.append(Path(hf_home) / "hub" if hf_home else Path.home() / ".cache" / "huggingface" / "hub")
    return dirs


def encoder_files_present(encoder_name: Optional[str] = None) -> bool:
    """True if the encoder is already downloaded. Checks folders only, so it never imports torch."""
    cfg = getattr(settings, "TRIAGE", None)
    name = encoder_name or (cfg.encoder_name if cfg else "sentence-transformers/all-MiniLM-L6-v2")
    folder = "models--" + name.replace("/", "--")
    return any((d / folder).is_dir() for d in encoder_cache_dirs())


def load_model(model_dir_path: Optional[Path] = None) -> Optional[MiniLMTriageModel]:
    d = model_dir_path or model_dir()
    joblib_path = d / "classifier.joblib"
    meta_path = d / "metadata.json"
    if not joblib_path.is_file() or not meta_path.is_file():
        return None

    import json

    meta_raw = json.loads(meta_path.read_text())
    metadata = TriageMetadata(
        version=meta_raw["version"],
        encoder_name=meta_raw["encoder_name"],
        threshold=float(meta_raw["threshold"]),
        data_hash=meta_raw.get("data_hash", ""),
        categories=meta_raw.get("categories", {}),
    )
    cfg = getattr(settings, "TRIAGE", None)
    expected_encoder = cfg.encoder_name if cfg else metadata.encoder_name
    if metadata.encoder_name != expected_encoder:
        logger.error("Encoder name mismatch: %s vs %s", metadata.encoder_name, expected_encoder)
        return None

    bundle: Dict[str, Any] = joblib.load(joblib_path)
    classifier = bundle["classifier"]
    centroids = bundle.get("centroids", np.zeros((0, 384)))
    category_ids = bundle.get("category_ids", [])

    try:
        import torch
        from sentence_transformers import SentenceTransformer

        torch.set_num_threads(1)
        local_only = settings.WEBAGENT_ENV in ("uat", "prod")
        encoder = SentenceTransformer(metadata.encoder_name, device="cpu", local_files_only=local_only)
    except Exception as exc:
        logger.warning("Could not load triage encoder: %s", exc)
        return None

    return MiniLMTriageModel(encoder, classifier, centroids, category_ids, metadata)


def get_model() -> Optional[MiniLMTriageModel]:
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
            # A corrupt joblib or metadata file must not add load time to every request.
            logger.error("Could not load the triage classifier: %s", exc)
            model = None
        if model is None:
            _failed_until = now + _FAIL_BACKOFF_SEC
            return None
        _cached_model = model
        return _cached_model


def set_model_for_tests(model: Optional[MiniLMTriageModel]) -> None:
    global _cached_model, _failed_until
    with _lock:
        _cached_model = model
        _failed_until = 0.0


def warm_up() -> None:
    """Load the model and embed one sentence. Never raises, so a broken model can't stop startup."""
    start = time.perf_counter()
    model = get_model()
    if model is None:
        logger.warning("Triage warm-up skipped: model not available; triage falls back to agentic")
        return
    try:
        model.predict_proba("warm-up sentence for triage")
    except Exception as exc:
        logger.warning("Triage warm-up failed: %s", exc)
        return
    logger.info(
        "Triage model warm-up complete (version=%s, %.0f ms)",
        model.metadata.version,
        (time.perf_counter() - start) * 1000,
    )
