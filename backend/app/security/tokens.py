"""Consent and approval tokens (SEC-050–SEC-053)."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
import uuid
from pathlib import Path
from typing import Dict, Literal, Optional

from app.core.config import settings
from app.security.records import _tokens_dir

TokenKind = Literal["consent", "approval"]


def _sign(payload: bytes) -> str:
    key = settings.SECRET_KEY.encode("utf-8")
    return base64.urlsafe_b64encode(hmac.new(key, payload, hashlib.sha256).digest()).decode("ascii").rstrip("=")


def binding_hash(binding: Dict[str, str]) -> str:
    canonical = json.dumps(binding, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def issue(
    kind: TokenKind,
    binding: Dict[str, str],
    policy_id: str,
    ttl_s: int = 900,
    action: Optional[str] = None,
) -> str:
    """Sign a token. `binding` is what it must match at redeem time; `action` (approval tokens)
    is the tool action being approved, carried signed in the payload."""
    jti = str(uuid.uuid4())
    payload = {
        "kid": "v1",
        "kind": kind,
        "binding_hash": binding_hash(binding),
        "policy_id": policy_id,
        "exp": int(time.time()) + ttl_s,
        "jti": jti,
    }
    if action:
        payload["action"] = action
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    body = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
    return f"{body}.{_sign(raw)}"


def _decode(token: str) -> Optional[Dict[str, object]]:
    try:
        body, sig = token.split(".", 1)
        raw = base64.urlsafe_b64decode(body + "==")
        expected = _sign(raw)
        if not hmac.compare_digest(sig, expected):
            return None
        return json.loads(raw.decode("utf-8"))
    except (ValueError, json.JSONDecodeError):
        return None


def redeem(token: str, kind: str, binding: Dict[str, str]) -> Optional[Dict[str, object]]:
    """Check signature, kind, expiry and binding, then claim the token once across workers.

    Returns the payload, or None (invalid, expired, reused, or the token store unavailable).
    """
    payload = _decode(token)
    if not payload:
        return None
    if payload.get("kind") != kind:
        return None
    if int(payload.get("exp", 0)) < int(time.time()):
        return None
    if payload.get("binding_hash") != binding_hash(binding):
        return None
    jti = str(payload.get("jti"))
    try:
        directory = _tokens_dir()
        directory.mkdir(parents=True, exist_ok=True)
        fd = os.open(directory / jti, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        os.close(fd)
    except FileExistsError:
        return None
    except OSError:
        # SEC-061: an unavailable token store means no consent and no approval.
        return None
    return payload


def query_hash(query: str) -> str:
    return hashlib.sha256((query or "").encode("utf-8")).hexdigest()


def consent_binding(query: str, history: object) -> Dict[str, str]:
    """What a consent token is bound to: this exact user turn and history."""
    return {"text_hash": binding_hash({"query": query, "history": str(history)})}
