"""The scheduler's own credential (REC-107, RNFR-06).

The scheduler sends `X-WebAgent-Scheduler: <WEBAGENT_SCHEDULER_TOKEN>`. The API key can't
identify it (the web proxy adds that key to every browser request), so scheduler-only fields
and routes need this header. The token is compared in constant time and never logged.
"""
import hmac

from fastapi import HTTPException, Request

from app.core.config import settings

SCHEDULER_HEADER = "X-WebAgent-Scheduler"


def _configured_token() -> str:
    return getattr(settings, "SCHEDULER_TOKEN", None) or ""


def scheduler_header_ok(request: Request) -> bool:
    """True when a scheduler token is set and the request's header matches it."""
    token = _configured_token()
    if not token:
        return False
    sent = request.headers.get(SCHEDULER_HEADER, "")
    return hmac.compare_digest(sent.encode("utf-8"), token.encode("utf-8"))


def require_scheduler(request: Request) -> None:
    """Dependency for scheduler-only routes: 404 while no token is set, 403 for a missing or wrong header."""
    if not _configured_token():
        raise HTTPException(status_code=404, detail="Not Found")
    if not scheduler_header_ok(request):
        raise HTTPException(status_code=403, detail="Scheduler credential required")
