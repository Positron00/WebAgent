"""Shared API error JSON shape (IMP-082)."""
from typing import Any

_STATUS_CODES: dict[int, str] = {
    400: "bad_request",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    409: "conflict",
    413: "too_large",
    415: "unsupported_media_type",
    422: "validation_error",
    429: "rate_limited",
    500: "internal_error",
}


def error_code_for_status(status_code: int) -> str:
    return _STATUS_CODES.get(status_code, "error")


def error_body(code: str, message: str, detail: Any = None) -> dict:
    body: dict = {"error": {"code": code, "message": message}}
    if detail is not None:
        body["detail"] = detail
    return body
