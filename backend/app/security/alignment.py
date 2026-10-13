"""Cosine alignment between tool arguments and the task text (SEC-044)."""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Protocol

import numpy as np

from app.core.config import settings
from app.models.research_plan import research_query
from app.models.task import WorkflowState

logger = logging.getLogger(__name__)

_TASK_EMBED_KEY = "_guard_task_embedding"


class _Encoder(Protocol):
    def encode(self, texts, normalize_embeddings=True): ...


_encoder: Optional[_Encoder] = None
_test_encoder: Optional[_Encoder] = None


def set_encoder_for_tests(encoder: Optional[_Encoder]) -> None:
    global _test_encoder
    _test_encoder = encoder


def _load_encoder() -> Optional[_Encoder]:
    global _encoder
    if _test_encoder is not None:
        return _test_encoder
    # The base MiniLM shared with ContextDefender (one copy per worker; never downloaded in uat/prod).
    from app.security.encoder import get_base_encoder

    encoder, _tokenizer = get_base_encoder()
    return encoder


def task_text(state: WorkflowState) -> str:
    return research_query(state) or state.query or ""


def _task_embedding(state: WorkflowState, encoder: _Encoder) -> np.ndarray:
    cache = state.context.setdefault(_TASK_EMBED_KEY, {})
    text = task_text(state)
    if text in cache:
        return np.asarray(cache[text])
    vec = np.asarray(encoder.encode([text], normalize_embeddings=True)[0])
    cache[text] = vec.tolist()
    return vec


def cosine_alignment(args_text: str, state: WorkflowState) -> int:
    """Return 0–100 alignment score; 100 when args are empty or the encoder is down."""
    args_text = (args_text or "").strip()
    if not args_text:
        return 100
    task = task_text(state).strip()
    if not task:
        return 100
    encoder = _load_encoder()
    if encoder is None:
        return 100
    try:
        arg_vec = np.asarray(encoder.encode([args_text], normalize_embeddings=True)[0])
        task_vec = _task_embedding(state, encoder)
        cos = float(np.dot(arg_vec, task_vec))
        return int(round(100 * max(0.0, cos)))
    except Exception as exc:
        logger.warning("Alignment scoring failed: %s", exc)
        return 100


def alignment_for_action(action: str, args: Dict[str, Any], state: WorkflowState) -> int:
    if action in ("tool::execute_code", "tool::document_extract"):
        return 100
    query = str(args.get("query") or args.get("code") or "")
    return cosine_alignment(query, state)
