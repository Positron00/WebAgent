"""One lock per tokenizer: Hugging Face fast tokenizers are not thread-safe.

The guard scores in worker threads (`asyncio.to_thread`), so two tasks scoring at once used the same
tokenizer from two threads and one failed with "Already borrowed"; ContextDefender then fell back to
patterns only and the item went unscored. A SentenceTransformer's `encode()` uses its tokenizer, so
the lock is keyed on the tokenizer: encoding and direct tokenizer calls on one model share it.
"""
from __future__ import annotations

import threading
from typing import Any, Dict

_locks: Dict[int, threading.RLock] = {}
_guard = threading.Lock()


def lock_for(model: Any) -> threading.RLock:
    """The lock for this encoder or tokenizer (an encoder maps to its tokenizer's lock)."""
    key = id(getattr(model, "tokenizer", None) or model)
    lock = _locks.get(key)
    if lock is None:
        with _guard:
            lock = _locks.setdefault(key, threading.RLock())
    return lock
