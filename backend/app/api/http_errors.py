"""Safe HTTP error responses for API clients."""
import uuid
import logging

from fastapi import HTTPException

from app.core.api_errors import error_body, error_code_for_status
from app.core.config import settings
from app.core.security import sanitize_error_message

logger = logging.getLogger(__name__)


def http_exception_body(exc: HTTPException) -> dict:
    detail = exc.detail
    if isinstance(detail, dict):
        message = str(detail.get("error") or detail.get("message") or detail)
    else:
        message = str(detail)
    return error_body(error_code_for_status(exc.status_code), message, detail)


def raise_safe_http_exception(
    status_code: int,
    exc: Exception,
    *,
    public_message: str = "An unexpected error occurred",
) -> None:
    error_id = str(uuid.uuid4())
    logger.exception(
        "%s (error_id=%s)",
        public_message,
        error_id,
        exc_info=exc,
    )
    if settings.DEBUG_MODE:
        detail = {
            "error": sanitize_error_message(str(exc)),
            "error_id": error_id,
        }
    else:
        detail = {"error": public_message, "error_id": error_id}
    raise HTTPException(status_code=status_code, detail=detail)
