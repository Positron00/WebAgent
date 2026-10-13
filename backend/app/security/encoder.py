"""The base MiniLM encoder shared by ContextDefender and alignment (one copy per worker)."""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Optional, Tuple

from app.core.config import settings

logger = logging.getLogger(__name__)

BASE_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"

_lock = threading.Lock()
_loaded: Optional[Tuple[Any, Any]] = None  # (SentenceTransformer, tokenizer)
_failed = False
_test_override: Optional[Tuple[Any, Any]] = None
_at_path: dict = {}  # str(path) -> (encoder, tokenizer) or (None, None) after a failure


def set_for_tests(encoder: Any, tokenizer: Any = None) -> None:
    global _test_override
    _test_override = (encoder, tokenizer) if encoder is not None else None


def get_encoder_at(path: Path) -> Tuple[Optional[Any], Optional[Any]]:
    """A SentenceTransformer saved on disk (ContextDefender's fine-tuned encoder); loaded once."""
    if _test_override is not None:
        return _test_override
    key = str(path)
    if key in _at_path:
        return _at_path[key]
    with _lock:
        if key not in _at_path:
            try:
                import torch
                from sentence_transformers import SentenceTransformer

                torch.set_num_threads(1)
                encoder = SentenceTransformer(key, device="cpu", local_files_only=True)
                _at_path[key] = (encoder, encoder.tokenizer)
            except Exception as exc:
                logger.warning("Encoder at %s unavailable for the security guard: %s", path, exc)
                _at_path[key] = (None, None)
        return _at_path[key]


def get_base_encoder() -> Tuple[Optional[Any], Optional[Any]]:
    """Return (encoder, tokenizer), or (None, None) when unavailable.

    In uat/prod the encoder must already be on disk (baked into the image); it is never downloaded.
    A failed load is not retried, so a missing model doesn't slow every request.
    """
    global _loaded, _failed
    if _test_override is not None:
        return _test_override
    if _loaded is not None:
        return _loaded
    if _failed:
        return None, None
    with _lock:
        if _loaded is not None:
            return _loaded
        try:
            import torch
            from sentence_transformers import SentenceTransformer

            torch.set_num_threads(1)
            local_only = settings.WEBAGENT_ENV in ("uat", "prod")
            encoder = SentenceTransformer(BASE_ENCODER, device="cpu", local_files_only=local_only)
            _loaded = (encoder, encoder.tokenizer)
            return _loaded
        except Exception as exc:
            logger.warning("Base MiniLM encoder unavailable for the security guard: %s", exc)
            _failed = True
            return None, None
