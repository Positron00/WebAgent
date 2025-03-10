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
import html
import secrets
from datetime import datetime, timedelta
from typing import Dict, Optional, Union, Any, List

import bcrypt
from fastapi import HTTPException, Request, Depends, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt, ExpiredSignatureError
from pydantic import BaseModel, EmailStr, validator, Field, ValidationError

from app.core.config import settings
from app.core.logger import logger

# JWT token handling
oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{settings.API_V1_STR}/auth/token")

# Rate limiting
class RateLimiter:
    """Rate limiter with support for Redis-based persistence."""
    
    def __init__(self):
        self.in_memory_requests: Dict[str, Dict[str, Union[int, float]]] = {}
        self.redis_client = None
        self.max_requests = settings.RATE_LIMIT_MAX_REQUESTS
        self.window_size = settings.RATE_LIMIT_WINDOW_SIZE
        self.burst_limit = settings.RATE_LIMIT_BURST_LIMIT
    
    def setup_redis(self, redis_client):
        """Set up Redis for distributed rate limiting."""
        self.redis_client = redis_client
        logger.info("Redis-based rate limiting enabled")
    
    def is_rate_limited(self, client_id: str) -> bool:
        """Check if a client is rate limited."""
        # If Redis is configured, use it for distributed rate limiting
        if self.redis_client:
            return self._is_rate_limited_redis(client_id)
        else:
            # Fall back to in-memory rate limiting
            return self._is_rate_limited_memory(client_id)
    
    def _is_rate_limited_redis(self, client_id: str) -> bool:
        """Check rate limiting using Redis."""
        try:
            now = time.time()
            pipeline = self.redis_client.pipeline()
            
            # Key names for Redis
            request_count_key = f"rate_limit:{client_id}:count"
            window_start_key = f"rate_limit:{client_id}:window_start"
            
            # Get current values
            pipeline.get(request_count_key)
            pipeline.get(window_start_key)
            count_bytes, window_start_bytes = pipeline.execute()
            
            # Convert from bytes or set defaults
            count = int(count_bytes) if count_bytes else 0
            window_start = float(window_start_bytes) if window_start_bytes else now
            
            # Reset window if it has expired
            if now - window_start > self.window_size:
                pipeline.set(request_count_key, 1)
                pipeline.set(window_start_key, now)
                pipeline.expire(request_count_key, int(self.window_size * 2))
                pipeline.expire(window_start_key, int(self.window_size * 2))
                pipeline.execute()
                return False
            
            # Check if rate limit is exceeded
            if count >= self.max_requests:
                # Allow burst up to the burst limit
                if count < self.burst_limit:
                    logger.warning(f"Client {client_id} exceeded regular rate limit but within burst limit. Count: {count}")
                    pipeline.incr(request_count_key)
                    pipeline.execute()
                    return False
                else:
                    logger.warning(f"Client {client_id} exceeded rate limit. Count: {count}")
                    return True
            
            # Increment request count
            pipeline.incr(request_count_key)
            pipeline.execute()
            return False
        except Exception as e:
            logger.error(f"Error in Redis rate limiting: {str(e)}")
            # Fall back to in-memory rate limiting on Redis errors
            return self._is_rate_limited_memory(client_id)
    
    def _is_rate_limited_memory(self, client_id: str) -> bool:
        """Check rate limiting using in-memory storage."""
        now = time.time()
        if client_id not in self.in_memory_requests:
            self.in_memory_requests[client_id] = {"count": 1, "window_start": now}
            return False
        
        client_data = self.in_memory_requests[client_id]
        window_start = client_data["window_start"]
        
        # Reset window if it has expired
        if now - window_start > self.window_size:
            client_data["count"] = 1
            client_data["window_start"] = now
            return False
        
        # Check if rate limit is exceeded
        if client_data["count"] >= self.max_requests:
            # Allow burst up to the burst limit
            if client_data["count"] < self.burst_limit:
                logger.warning(f"Client {client_id} exceeded regular rate limit but within burst limit. Count: {client_data['count']}")
                client_data["count"] += 1
                return False
            else:
                logger.warning(f"Client {client_id} exceeded rate limit. Count: {client_data['count']}")
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
        logger.warning(f"Rate limit exceeded for client: {client_id}")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS, 
            detail="Rate limit exceeded. Please try again later.",
            headers={"Retry-After": str(rate_limiter.window_size)}
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
    to_encode.update({"nbf": datetime.utcnow()})  # Not valid before now
    to_encode.update({"iat": datetime.utcnow()})  # Issued at
    
    token = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    logger.debug(f"Created token for user: {data.get('sub', 'unknown')}")
    return token

def verify_token(token: str) -> Dict[str, Any]:
    """
    Verify a JWT token and return its payload.
    """
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        return payload
    except ExpiredSignatureError:
        logger.warning(f"Expired token detected")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except JWTError as e:
        logger.warning(f"Invalid token: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
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
    
    This function provides comprehensive sanitization to:
    1. Escape HTML entities to prevent XSS
    2. Remove potentially dangerous characters
    3. Limit string length to prevent DoS
    """
    if not input_str:
        return ""
    
    # Truncate extremely long strings (DoS protection)
    if len(input_str) > settings.MAX_INPUT_LENGTH:
        logger.warning(f"Input string truncated from {len(input_str)} to {settings.MAX_INPUT_LENGTH} characters")
        input_str = input_str[:settings.MAX_INPUT_LENGTH]
    
    # First escape HTML entities
    sanitized = html.escape(input_str)
    
    # Remove potentially dangerous patterns
    patterns = [
        r"[;<>/]",              # Basic dangerous characters
        r"javascript:",         # JavaScript protocol
        r"data:",               # Data URI
        r"vbscript:",           # VBScript protocol
        r"on\w+\s*=",           # Event handlers (e.g., onclick=)
        r"(?:--|\+\+)",         # SQL injection patterns 
        r";\s*(?:drop|alter)"   # More SQL injection patterns
    ]
    
    for pattern in patterns:
        sanitized = re.sub(pattern, "", sanitized, flags=re.IGNORECASE)
    
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
        logger.warning("Request without User-Agent header detected")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="User-Agent header is required")
    
    # Check if user agent is excessively long (potential attack)
    if len(user_agent) > 500:
        logger.warning(f"Excessively long User-Agent detected: {len(user_agent)} chars")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="User-Agent header too long")
    
    # Check for suspicious parameters
    suspicious_params: List[str] = []
    for param, value in request.query_params.items():
        # Check param length
        if len(param) > 100:
            suspicious_params.append(param)
            continue
            
        # Check for dangerous characters
        if re.search(r"[;<>']", param) or re.search(r"[;<>']", value):
            suspicious_params.append(param)
            continue
            
        # Check for SQL injection patterns
        if re.search(r"(?:--|;|//|#|/\*)", value):
            suspicious_params.append(param)
            continue
            
    if suspicious_params:
        logger.warning(f"Suspicious parameters detected: {', '.join(suspicious_params)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail=f"Invalid characters in query parameters: {', '.join(suspicious_params)}"
        )
    
    return None 