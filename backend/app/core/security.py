"""
Security related utilities for the WebAgent backend.
"""
import os
import time
import secrets
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, List, Any, Union

import jwt
from fastapi import Depends, FastAPI, HTTPException, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader
from pydantic import BaseModel, Field

from app.core.config import settings

# Security token handling
security = HTTPBearer()
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Hard-coded API keys for development only - these would be stored in a secure 
# database in production
DEV_API_KEYS = {
    "admin": {"key": "dev_admin_key", "scope": "admin"},
    "user": {"key": "dev_user_key", "scope": "user"}
}

# Rate limiting implementation
rate_limit_data: Dict[str, Dict[str, Any]] = {}


def get_authorization_header(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Extract and validate the authorization token."""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials


def validate_auth(auth_value: str) -> Dict[str, Any]:
    """Validate either JWT token or API key."""
    # Try to validate as JWT
    if settings.JWT_ENABLED:
        try:
            payload = decode_jwt(auth_value)
            return {"user_id": payload.get("sub"), "scope": payload.get("scope", "user")}
        except Exception:
            # Not a valid JWT, try API key next
            pass
    
    # Try to validate as API key
    if settings.API_KEY_ENABLED:
        api_key = auth_value.split("Bearer ")[-1] if auth_value.startswith("Bearer ") else auth_value
        for user_id, data in DEV_API_KEYS.items():
            if secrets.compare_digest(api_key, data["key"]):
                return {"user_id": user_id, "scope": data["scope"]}
    
    # If we reach here, auth failed
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )


def create_jwt(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Create a new JWT token."""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt


def decode_jwt(token: str) -> Dict[str, Any]:
    """Decode and validate a JWT token."""
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        return payload
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


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
        # Get client identifier (IP address or user ID if authenticated)
        client_host = request.client.host if request.client else "unknown"
        
        # Apply rate limiting
        if not apply_rate_limit(client_host):
            return HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many requests"
            )
        
        # Check request size
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > settings.REQUEST_SIZE_LIMIT:
            return HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="Request too large"
            )
        
        # Continue processing the request
        response = await call_next(request)
        return response 