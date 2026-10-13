"""Fast pattern scans for checkpoint 1 (context) and checkpoint 2 (tool args)."""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import AbstractSet, Dict, Iterable, Optional, Set

BACKEND_DIR = Path(__file__).resolve().parents[2]
DEFAULT_PATTERNS_PATH = BACKEND_DIR / "config" / "security" / "patterns.json"

CONTEXT_HITS = frozenset(
    {
        "injection_phrase",
        "role_marker",
        "hidden_text",
        "api_key",
        "private_key",
        "password",
    }
)
TOOL_HITS = frozenset(
    {
        "api_key",
        "private_key",
        "password",
        "email",
        "phone",
        "long_token",
        "url",
        "site_operator",
        "query_too_long",
        "document_provenance",
    }
)
QUERY_MAX_LEN = 300


@lru_cache(maxsize=1)
def _compiled_table() -> Dict[str, tuple[re.Pattern[str], ...]]:
    path = DEFAULT_PATTERNS_PATH
    raw = json.loads(path.read_text(encoding="utf-8"))
    table: Dict[str, tuple[re.Pattern[str], ...]] = {}
    for name, patterns in raw.items():
        if name == "query_too_long":
            table[name] = tuple()
            continue
        table[name] = tuple(re.compile(p) for p in patterns)
    return table


def reload_patterns() -> None:
    _compiled_table.cache_clear()


def _match_names(text: str, names: Iterable[str]) -> Set[str]:
    hits: Set[str] = set()
    table = _compiled_table()
    for name in names:
        for pattern in table.get(name, ()):
            if pattern.search(text):
                hits.add(name)
                break
    return hits


def scan(text: str) -> Set[str]:
    """Scan context-item text (checkpoint 1)."""
    return _match_names(text, CONTEXT_HITS)


def _matched_strings(text: str, name: str) -> Set[str]:
    found: Set[str] = set()
    for pattern in _compiled_table().get(name, ()):
        found.update(m.group(0) for m in pattern.finditer(text))
    return found


def scan_args(
    query: str,
    *,
    user_turn: Optional[str] = None,
    document_id: Optional[str] = None,
    task_document_id: Optional[str] = None,
) -> Set[str]:
    """Scan tool arguments (SEC-041).

    A kind is reported when at least one matched string is **not** in the current user turn:
    an email the user typed is fine to search for; a different email (for example one an
    injected page asked for) is still reported.
    """
    query = query or ""
    hits: Set[str] = set()
    if len(query) > QUERY_MAX_LEN:
        hits.add("query_too_long")
    turn = user_turn or ""
    for name in TOOL_HITS - {"document_provenance", "query_too_long"}:
        matches = _matched_strings(query, name)
        if any(m not in turn for m in matches):
            hits.add(name)
    if document_id is not None and task_document_id is not None and document_id != task_document_id:
        hits.add("document_provenance")
    return hits


def vocabulary() -> AbstractSet[str]:
    return CONTEXT_HITS | TOOL_HITS
