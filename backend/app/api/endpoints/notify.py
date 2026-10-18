"""Confirm and unsubscribe (recurring-research spec REC-133, REC-134; contracts in the low-level design §12a).

- `GET /api/notify/lookup?token=…` shows what a token is for: `{topic_name, purpose}`. Read-only.
- `POST /api/notify/confirm` `{token}` marks the address confirmed. The token is single-use.
- `POST /api/notify/unsubscribe` `{token}`, or the RFC 8058 one-click form (`List-Unsubscribe=One-Click`
  with `?token=` in the query), removes the address. Idempotent.

A GET never changes anything: mail scanners follow links. The routes need the API key; they're public
only in the web app, whose proxies add it. Tokens carry a keyed hash of the address, never the address.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional, Tuple
from urllib.parse import parse_qs

from fastapi import APIRouter, HTTPException, Query, Request
from starlette.concurrency import run_in_threadpool

from app.services import notify_email, notify_tokens, topics
from app.services import topic_files as files
from app.services.topics import TopicDefinition

logger = logging.getLogger(__name__)
router = APIRouter()

_MAX_BODY = 4096


def _invalid() -> HTTPException:
    return HTTPException(status_code=404, detail="Invalid or expired link")


def _live_topic(topic_id: str) -> Optional[TopicDefinition]:
    defn = topics.load_definition(topic_id)
    if defn is None or defn.status == "deleted":
        return None
    return defn


def _lookup(token: str) -> Optional[Dict[str, Any]]:
    found = notify_tokens.verify_any(token)
    if found is None:
        return None
    purpose, topic_id, ref = found
    if purpose == "confirm" and notify_email.confirm_token_used(token):
        return None
    defn = _live_topic(topic_id)
    if defn is None:
        return None
    if purpose == "confirm" and not any(notify_tokens.matches(ref, s.email) for s in defn.notify.subscribers):
        return None
    return {"topic_name": defn.name, "purpose": purpose}


@router.get("/lookup")
async def lookup(token: str = Query(default="", max_length=2048)) -> Dict[str, Any]:
    found = await run_in_threadpool(_lookup, token)
    if found is None:
        raise _invalid()
    return found


async def _token_from_json(request: Request) -> str:
    raw = await request.body()
    if len(raw) > _MAX_BODY:
        raise HTTPException(status_code=413, detail="Body too large")
    try:
        data = json.loads(raw or b"{}")
    except ValueError:
        raise HTTPException(status_code=422, detail="Body must be JSON {token}") from None
    token = data.get("token") if isinstance(data, dict) else None
    if not isinstance(token, str) or not token:
        raise HTTPException(status_code=422, detail="token is required")
    return token


def _confirm(token: str) -> bool:
    found = notify_tokens.verify(token, "confirm")
    if found is None:
        return False
    topic_id, ref = found
    defn = _live_topic(topic_id)
    if defn is None or not any(notify_tokens.matches(ref, s.email) for s in defn.notify.subscribers):
        return False
    if not notify_email.use_confirm_token(token):
        return False

    hit = []

    def mark(d: TopicDefinition) -> None:
        if d.status == "deleted":
            return
        for subscriber in d.notify.subscribers:
            if notify_tokens.matches(ref, subscriber.email):
                if subscriber.confirmed_at is None:
                    subscriber.confirmed_at = files.utcnow()
                hit.append(True)

    topics.update_definition(topic_id, mark)
    if hit:
        logger.info("Topic %s subscriber confirmed", topic_id)
    return bool(hit)


@router.post("/confirm")
async def confirm(request: Request) -> Dict[str, str]:
    token = await _token_from_json(request)
    if not await run_in_threadpool(_confirm, token):
        raise _invalid()
    return {"status": "confirmed"}


async def _unsubscribe_token(request: Request) -> Tuple[str, bool]:
    """The token and whether the request is an RFC 8058 one-click POST."""
    content_type = (request.headers.get("content-type") or "").split(";")[0].strip().lower()
    if content_type == "application/x-www-form-urlencoded" or (
        not content_type and request.query_params.get("token")
    ):
        raw = await request.body()
        if len(raw) > _MAX_BODY:
            raise HTTPException(status_code=413, detail="Body too large")
        form = parse_qs(raw.decode("utf-8", "replace"), keep_blank_values=True)
        if form.get("List-Unsubscribe") != ["One-Click"]:
            raise HTTPException(status_code=422, detail="Expected List-Unsubscribe=One-Click")
        token = request.query_params.get("token") or ""
        if not token:
            raise HTTPException(status_code=422, detail="token is required")
        return token, True
    return await _token_from_json(request), False


def _unsubscribe(token: str) -> bool:
    found = notify_tokens.verify(token, "unsubscribe")
    if found is None:
        return False
    topic_id, ref = found
    if _live_topic(topic_id) is None:
        return True   # the topic (and so the address) is already gone

    removed = []

    def drop(d: TopicDefinition) -> None:
        kept = [s for s in d.notify.subscribers if not notify_tokens.matches(ref, s.email)]
        if len(kept) != len(d.notify.subscribers):
            removed.append(True)
        d.notify.subscribers = kept

    topics.update_definition(topic_id, drop)
    if removed:
        logger.info("Topic %s subscriber removed (unsubscribe)", topic_id)
    return True


@router.post("/unsubscribe")
async def unsubscribe(request: Request) -> Dict[str, str]:
    token, _one_click = await _unsubscribe_token(request)
    if not await run_in_threadpool(_unsubscribe, token):
        raise _invalid()
    return {"status": "unsubscribed"}
