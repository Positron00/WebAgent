"""Signed confirm and unsubscribe tokens of topic email (recurring-research spec REC-133, REC-134, RNFR-06).

A token is `base64url(payload) + "." + base64url(hmac_sha256(SECRET_KEY, base64url(payload)))`, where
the payload is the compact JSON `{p, t, a, e}`: purpose, topic ID, address and expiry (Unix seconds,
or null for no expiry). Confirm tokens expire after 48 h and are single-use (the email ledger records
used ones); unsubscribe tokens never expire and are safe to repeat.

`a` is the address's keyed hash (`address_ref`), not the address: tokens travel in URLs, and request
and access logs write URLs, so an address in base64 would reach the logs (RNFR-03). The binding is the
same, since the signature covers the hash; callers match the hash against the topic's subscribers.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from typing import Literal, Optional, Tuple

from app.core.config import settings
from app.services.topic_files import is_uuid

Purpose = Literal["confirm", "unsubscribe"]
PURPOSES = ("confirm", "unsubscribe")
CONFIRM_TTL_SECONDS = 48 * 3600
_MAX_TOKEN_CHARS = 2048


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def _signature(payload_b64: str) -> str:
    key = (settings.SECRET_KEY or "").encode("utf-8")
    return _b64(hmac.new(key, payload_b64.encode("ascii"), hashlib.sha256).digest())


def address_ref(address: str) -> str:
    """The address as tokens and the ledger hold it: HMAC-SHA256 with the secret key, in hex."""
    key = (settings.SECRET_KEY or "").encode("utf-8")
    return hmac.new(key, b"address:" + address.strip().lower().encode("utf-8"), hashlib.sha256).hexdigest()


def matches(ref: str, address: str) -> bool:
    """True if `ref` (from `verify`) is `address`'s reference. Constant time."""
    return hmac.compare_digest(ref.encode("ascii", "replace"), address_ref(address).encode("ascii"))


def sign(purpose: Purpose, topic_id: str, address: str, expires: Optional[int]) -> str:
    """A token for `purpose` on (topic, address), valid until `expires` (Unix seconds) or forever."""
    if purpose not in PURPOSES:
        raise ValueError("unknown token purpose")
    payload = {"p": purpose, "t": str(topic_id).lower(), "a": address_ref(address), "e": expires}
    payload_b64 = _b64(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8"))
    return f"{payload_b64}.{_signature(payload_b64)}"


def confirm_token(topic_id: str, address: str, now: Optional[float] = None) -> str:
    return sign("confirm", topic_id, address, int((now if now is not None else time.time()) + CONFIRM_TTL_SECONDS))


def unsubscribe_token(topic_id: str, address: str) -> str:
    return sign("unsubscribe", topic_id, address, None)


def verify(token: str, purpose: str, now: Optional[float] = None) -> Optional[Tuple[str, str]]:
    """(topic_id, address_ref) if `token` is a valid, unexpired token for `purpose`; else None.

    The signature is compared in constant time before the payload is parsed. Match the second value
    against addresses with `matches`.
    """
    if not isinstance(token, str) or not token or len(token) > _MAX_TOKEN_CHARS or token.count(".") != 1:
        return None
    payload_b64, sent_sig = token.split(".")
    try:
        expected = _signature(payload_b64)
    except (UnicodeEncodeError, ValueError):
        return None
    if not hmac.compare_digest(sent_sig.encode("ascii", "replace"), expected.encode("ascii")):
        return None
    try:
        payload = json.loads(_unb64(payload_b64).decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("p") != purpose or purpose not in PURPOSES:
        return None
    topic_id, address, expires = payload.get("t"), payload.get("a"), payload.get("e")
    if not is_uuid(topic_id) or not isinstance(address, str) or len(address) != 64:
        return None
    if expires is not None:
        if not isinstance(expires, int) or isinstance(expires, bool):
            return None
        if expires <= (now if now is not None else time.time()):
            return None
    elif purpose == "confirm":
        return None   # a confirm token always expires
    return topic_id, address


def verify_any(token: str, now: Optional[float] = None) -> Optional[Tuple[str, str, str]]:
    """(purpose, topic_id, address_ref) for a valid token of either purpose, else None."""
    for purpose in PURPOSES:
        found = verify(token, purpose, now=now)
        if found is not None:
            return purpose, found[0], found[1]
    return None


def token_hash(token: str) -> str:
    """SHA-256 of a token, as the ledger records used confirm tokens."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()
