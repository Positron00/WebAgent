"""
Security related utilities for the WebAgent backend.
"""
import os
import time
import secrets
from typing import Dict, Optional, List, Any

from fastapi import FastAPI, HTTPException, Security, status
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader

from app.core.api_errors import error_body
from app.core.config import settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

INSECURE_SECRET_DEFAULT = "insecure_key_for_dev_only_change_in_production"


def _configured_api_keys() -> List[str]:
    """Valid API keys: only WEBAGENT_API_KEY, in every environment."""
    env_key = os.getenv("WEBAGENT_API_KEY")
    return [env_key] if env_key else []


async def require_api_key(api_key: Optional[str] = Security(api_key_header)) -> None:
    """Require a valid API key when API_KEY_ENABLED is on."""
    if not settings.API_KEY_ENABLED:
        return
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required",
        )
    for valid in _configured_api_keys():
        if secrets.compare_digest(api_key, valid):
            return
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid API key",
    )

# Rate limiting implementation
rate_limit_data: Dict[str, Dict[str, Any]] = {}
_last_rate_limit_sweep = 0.0

# Requests from these addresses are proxies (the Next.js server by default); the caller's
# own address is taken from their X-Forwarded-For header instead.
TRUSTED_PROXIES = {
    host.strip()
    for host in os.getenv("WEBAGENT_TRUSTED_PROXIES", "127.0.0.1,::1").split(",")
    if host.strip()
}

# Probes and metrics scrapes are frequent and must never be throttled.
RATE_LIMIT_EXEMPT_PATHS = {"/", "/health", "/metrics"}


CLIENT_HEADER = "x-webagent-client"


def _holds_api_key(request) -> bool:
    """The caller sent a valid X-API-Key (only servers, such as the web app, hold one)."""
    given = request.headers.get("x-api-key")
    return bool(
        settings.API_KEY_ENABLED
        and given
        and any(secrets.compare_digest(given, key) for key in _configured_api_keys())
    )


def client_identifier(request) -> str:
    """Who to rate-limit: the web app's session, else the nearest untrusted address."""
    peer = request.client.host if request.client else "unknown"
    via_proxy = peer in TRUSTED_PROXIES
    if via_proxy or _holds_api_key(request):
        client = request.headers.get(CLIENT_HEADER, "").strip()
        if client.startswith("s:") and 3 <= len(client) <= 66:
            return client
    if not via_proxy:
        return peer
    forwarded = request.headers.get("x-forwarded-for", "")
    hops = [hop.strip() for hop in forwarded.split(",") if hop.strip()]
    for hop in reversed(hops):  # right-most first: the hop our proxy added
        if hop not in TRUSTED_PROXIES:
            return hop
    return peer


def _is_rate_limit_exempt(path: str) -> bool:
    return (
        path in RATE_LIMIT_EXEMPT_PATHS
        or path == f"{settings.API_V1_STR}/metrics"
        or path.startswith(f"{settings.API_V1_STR}/health")
    )


def _sweep_idle_identifiers(current_time: float) -> None:
    """Drop callers with no requests inside the window, so the table doesn't grow forever."""
    global _last_rate_limit_sweep
    if current_time - _last_rate_limit_sweep < settings.RATE_LIMIT_WINDOW_SIZE:
        return
    _last_rate_limit_sweep = current_time
    window_start = current_time - settings.RATE_LIMIT_WINDOW_SIZE
    for identifier in list(rate_limit_data):
        requests = rate_limit_data[identifier]["requests"]
        if not requests or requests[-1] <= window_start:
            del rate_limit_data[identifier]


def apply_rate_limit(identifier: str) -> bool:
    """
    Apply rate limiting logic.
    
    Args:
        identifier: The identifier to rate limit (e.g., IP address, user ID)
        
    Returns:
        True if the request is allowed, False if it's rate limited
    """
    if not settings.RATE_LIMIT_ENABLED:
        return True
    
    current_time = time.time()
    _sweep_idle_identifiers(current_time)

    # Initialize rate limiting data for this identifier if it doesn't exist
    if identifier not in rate_limit_data:
        rate_limit_data[identifier] = {
            "requests": [],
            "last_reset": current_time
        }
    
    # Clean up old requests
    window_start = current_time - settings.RATE_LIMIT_WINDOW_SIZE
    rate_limit_data[identifier]["requests"] = [
        req_time for req_time in rate_limit_data[identifier]["requests"] 
        if req_time > window_start
    ]
    
    # Check if rate limited
    if len(rate_limit_data[identifier]["requests"]) >= settings.RATE_LIMIT_MAX_REQUESTS:
        return False
    
    # Add this request
    rate_limit_data[identifier]["requests"].append(current_time)
    return True


def sanitize_error_message(error_message: str) -> str:
    """
    Sanitize error messages to remove potentially sensitive information.
    
    Args:
        error_message: The raw error message
        
    Returns:
        A sanitized version of the error message
    """
    # List of patterns to sanitize (keys, passwords, connection strings, etc.)
    patterns_to_sanitize = [
        # API keys
        (r'(api[_-]?key|apikey)[":\s=\']+[a-zA-Z0-9_\-\.]+', r'\1="*****"'),
        (r'(sk|pk)_(test|live)_[a-zA-Z0-9]+', r'[API_KEY_REDACTED]'),
        # Passwords
        (r'(password|passwd|pwd)[":\s=\']+\S+', r'\1="*****"'),
        # Connection strings
        (r'(mongodb|postgres|mysql|jdbc)://\S+', r'\1://[CONNECTION_STRING_REDACTED]'),
    ]
    
    sanitized = error_message
    for pattern, replacement in patterns_to_sanitize:
        import re
        sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)
    
    return sanitized


def check_request_size(content_length: int) -> bool:
    """
    Check if the request size is within allowed limits.
    
    Args:
        content_length: Content length in bytes
        
    Returns:
        True if the size is acceptable, False otherwise
    """
    max_size = settings.REQUEST_SIZE_LIMIT
    return content_length <= max_size


def setup_security(app: FastAPI) -> None:
    """Configure security settings for the FastAPI app."""
    
    @app.middleware("http")
    async def security_middleware(request, call_next):
        """
        Middleware for handling security concerns like rate limiting and request size.
        """
        # Apply rate limiting per caller (behind the web app's proxy, per end user)
        if not _is_rate_limit_exempt(request.url.path) and not apply_rate_limit(
            client_identifier(request)
        ):
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content=error_body("rate_limited", "Too many requests"),
                headers={"Retry-After": str(settings.RATE_LIMIT_WINDOW_SIZE)},
            )

        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > settings.REQUEST_SIZE_LIMIT:
            return JSONResponse(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                content=error_body("too_large", "Request too large"),
            )
        
        # Continue processing the request
        response = await call_next(request)
        return response 