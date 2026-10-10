"""
Compatibility entry point for tests and legacy docs.

The canonical FastAPI application lives in ``backend/main.py`` (import as ``main``).
"""

from main import app

__all__ = ["app"]
