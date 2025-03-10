"""
Security module for the WebAgent backend.

This module provides various security features including:
- Input validation and sanitization
- Rate limiting
- JWT token generation and validation
- Password hashing and verification
"""
import re
import time
import secrets
from datetime import datetime, timedelta
from typing import Dict, Optional, Union, Any

import bcrypt
from fastapi import HTTPException, Request, Depends
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from pydantic import BaseModel, EmailStr, validator, Field, ValidationError

from app.core.config import settings

# JWT token handling
oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{settings.API_V1_STR}/auth/token")

# Rate limiting
class RateLimiter:
    """Simple in-memory rate limiter."""
    
    def __init__(self):
        self.requests: Dict[str, Dict[str, Union[int, float]]] = {}
        self.max_requests = 100  # per minute
        self.window_size = 60  # seconds
    
    def is_rate_limited(self, client_id: str) -> bool:
        """Check if a client is rate limited."""
        now = time.time()
        if client_id not in self.requests:
            self.requests[client_id] = {"count": 1, "window_start": now}
            return False
        
        client_data = self.requests[client_id]
        window_start = client_data["window_start"]
        
        # Reset window if it has expired
        if now - window_start > self.window_size:
            client_data["count"] = 1
            client_data["window_start"] = now
            return False
        
        # Check if rate limit is exceeded
        if client_data["count"] >= self.max_requests:
            return True
        
        # Increment request count
        client_data["count"] += 1
        return False

# Initialize rate limiter
rate_limiter = RateLimiter()

async def rate_limit_dependency(request: Request) -> None:
    """Dependency for rate limiting."""
    client_id = request.client.host
    if rate_limiter.is_rate_limited(client_id):
        raise HTTPException(
            status_code=429, 
            detail="Rate limit exceeded. Please try again later."
        )

# JWT Token Functions
def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a JWT access token.
    """
    to_encode = data.copy()
    expire = datetime.utcnow() + (
        expires_delta if expires_delta else
        timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)

def verify_token(token: str) -> Dict[str, Any]:
    """
    Verify a JWT token and return its payload.
    """
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

# User Authentication Models
class UserCreate(BaseModel):
    """Model for user creation."""
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8)
    
    @validator("password")
    def validate_password(cls, v):
        """Validate password strength."""
        if not re.search(r"[A-Z]", v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not re.search(r"[a-z]", v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not re.search(r"[0-9]", v):
            raise ValueError("Password must contain at least one digit")
        if not re.search(r"[^A-Za-z0-9]", v):
            raise ValueError("Password must contain at least one special character")
        return v
    
    @validator("username")
    def validate_username(cls, v):
        """Validate username format."""
        if not re.match(r"^[a-zA-Z0-9_]+$", v):
            raise ValueError("Username must only contain letters, numbers, and underscores")
        return v

# Password handling
def get_password_hash(password: str) -> str:
    """
    Generate a password hash using bcrypt.
    """
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode(), salt).decode()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a password against a hash.
    """
    return bcrypt.checkpw(plain_password.encode(), hashed_password.encode())

# Input validation
def sanitize_input(input_str: str) -> str:
    """
    Sanitize user input to prevent injection attacks.
    """
    # Remove potentially dangerous characters
    sanitized = re.sub(r"[;<>/]", "", input_str)
    return sanitized

def generate_secure_token(length: int = 32) -> str:
    """
    Generate a cryptographically secure token.
    """
    return secrets.token_urlsafe(length)

# Request validation
def validate_api_request(request: Request) -> None:
    """
    Validate API request headers and parameters.
    """
    # Check for required headers
    user_agent = request.headers.get("User-Agent")
    if not user_agent:
        raise HTTPException(status_code=400, detail="User-Agent header is required")
    
    # Check for suspicious parameters
    for param in request.query_params:
        if len(param) > 100:
            raise HTTPException(status_code=400, detail="Query parameter too long")
        if re.search(r"[;<>]", param):
            raise HTTPException(status_code=400, detail="Invalid characters in query parameter")
    
    return None 