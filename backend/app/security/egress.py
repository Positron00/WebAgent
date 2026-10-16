"""Private-text egress signal (checkpoint2-intent spec SEC-106, SNFR-13).

After an allowed internal retrieval or a document read, the 8-word windows (shingles) of that text
are stored as salted hashes in the task's security ledger, minus windows that also occur in the
user's intent. A web search then gets ``egress_overlap``: the length in words of its longest run
covered by stored windows. Only hashes are kept, in memory with the task; the ledger keys copied
into results and the API (``TaskManager._security_block``) don't include them.
"""
from __future__ import annotations

import hashlib
import re
import secrets
from typing import Any, Dict, List, Optional

from app.models.task import WorkflowState
from app.security import ledger

WINDOW = 8
MAX_HASHES = 20_000
DIGEST_HEX = 16
_WORD = re.compile(r"[a-z0-9]+")


def _windows(text: str) -> List[str]:
    words = _WORD.findall((text or "").lower())
    return [" ".join(words[i:i + WINDOW]) for i in range(len(words) - WINDOW + 1)]


def _hash(salt: str, window: str) -> str:
    return hashlib.sha256((salt + window).encode("utf-8")).hexdigest()[:DIGEST_HEX]


def _store(state: WorkflowState) -> Dict[str, Any]:
    led = ledger.get_ledger(state)
    store = led.get("egress")
    if not isinstance(store, dict) or "salt" not in store:
        store = {"salt": secrets.token_hex(16), "hashes": []}
        led["egress"] = store
    return store


def _intent_text(state: WorkflowState) -> str:
    from app.security.intent import intent_text

    return intent_text(state)


def remember(state: Optional[WorkflowState], text: str) -> None:
    """Store hashed shingles of private text seen in this task. Never raises."""
    if state is None or not text:
        return
    try:
        store = _store(state)
        salt = store["salt"]
        exclude = {_hash(salt, w) for w in _windows(_intent_text(state))}
        known = set(store["hashes"])
        for window in _windows(text):
            digest = _hash(salt, window)
            if digest not in exclude and digest not in known:
                store["hashes"].append(digest)
                known.add(digest)
        if len(store["hashes"]) > MAX_HASHES:
            del store["hashes"][: len(store["hashes"]) - MAX_HASHES]
    except Exception:  # noqa: BLE001 - a signal, never a reason to fail a retrieval
        return


def overlap(state: Optional[WorkflowState], query: str) -> int:
    """Length in words of the longest run of ``query`` covered by stored private-text windows."""
    if state is None:
        return 0
    store = ledger.get_ledger(state).get("egress")
    if not isinstance(store, dict) or not store.get("hashes"):
        return 0
    known = set(store["hashes"])
    salt = store["salt"]
    longest = run = 0
    for window in _windows(query):
        run = run + 1 if _hash(salt, window) in known else 0
        longest = max(longest, run)
    return longest + WINDOW - 1 if longest else 0
