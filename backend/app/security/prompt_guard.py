"""Meta's Llama Prompt Guard 2, the local jailbreak judge for borderline user turns (SEC-035a).

Llama Prompt Guard 2 is licensed under the Llama 4 Community License, Copyright © Meta Platforms,
Inc. All Rights Reserved. The model files, LICENSE, USE_POLICY.md and NOTICE live in
models/security/prompt_guard/ (downloaded with scripts/fetch_security_sources.py --prompt-guard; the
Hugging Face repository is gated). Built with Llama.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Optional, Tuple

from app.core.config import BACKEND_DIR

logger = logging.getLogger(__name__)

MODEL_DIR = Path(BACKEND_DIR) / "models" / "security" / "prompt_guard"
MAX_TOKENS = 512  # the model's context; longer messages are scored in windows
STRIDE = 384

_lock = threading.Lock()
_loaded: Optional[Tuple[Any, Any]] = None
_failed = False
_test_scorer = None


def set_scorer_for_tests(scorer) -> None:
    """A callable text -> probability, replacing the model (None restores it)."""
    global _test_scorer
    _test_scorer = scorer


def available(model_dir: Optional[Path] = None) -> bool:
    d = model_dir or MODEL_DIR
    return (d / "config.json").is_file() and any(d.glob("*.safetensors"))


def _load() -> Optional[Tuple[Any, Any]]:
    global _loaded, _failed
    if _loaded is not None or _failed:
        return _loaded
    with _lock:
        if _loaded is not None or _failed:
            return _loaded
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            torch.set_num_threads(1)
            tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
            model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR), local_files_only=True)
            model.eval()
            _loaded = (tokenizer, model)
        except Exception as exc:
            logger.warning("Prompt Guard unavailable at %s: %s", MODEL_DIR, exc)
            _failed = True
        return _loaded


def jailbreak_probability(text: str) -> Optional[float]:
    """P(malicious) for a message: the highest over 512-token windows; None if the model isn't there."""
    if _test_scorer is not None:
        return float(_test_scorer(text))
    loaded = _load()
    if loaded is None:
        return None
    import torch

    tokenizer, model = loaded
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    body = MAX_TOKENS - 2
    starts = list(range(0, max(1, len(ids) - body + STRIDE), STRIDE)) if len(ids) > body else [0]
    best = 0.0
    with torch.no_grad():
        for start in starts:
            window = tokenizer.decode(ids[start:start + body])
            enc = tokenizer(window, return_tensors="pt", truncation=True, max_length=MAX_TOKENS)
            probs = torch.softmax(model(**enc).logits, dim=-1)[0]
            best = max(best, float(probs[-1]))  # last label is "malicious"
    return best
