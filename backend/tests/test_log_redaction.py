"""Credential query parameters never reach the logs (recurring-research RNFR-03, RNFR-06)."""
import logging

from fastapi.testclient import TestClient

from app.core.log_context import RedactAccessLogFilter, redact_url


def test_redact_url_hides_token_values_only():
    url = "http://h/api/notify/unsubscribe?token=abc.def&x=1&API_KEY=s3cret#frag"
    assert redact_url(url) == "http://h/api/notify/unsubscribe?token=[redacted]&x=1&API_KEY=[redacted]#frag"
    assert redact_url("/api/topics?limit=5") == "/api/topics?limit=5"


def test_uvicorn_access_log_args_are_redacted():
    record = logging.LogRecord("uvicorn.access", logging.INFO, __file__, 1, '%s - "%s %s HTTP/%s" %d',
                               ("1.2.3.4:5", "POST", "/api/notify/unsubscribe?token=abc", "1.1", 200), None)
    assert RedactAccessLogFilter().filter(record)
    assert "abc" not in record.getMessage() and "token=[redacted]" in record.getMessage()


def test_request_log_line_has_no_token(caplog, monkeypatch):
    import main
    from app.core.config import settings

    monkeypatch.setattr(settings, "API_KEY_ENABLED", False)
    with caplog.at_level(logging.INFO):
        TestClient(main.app).get("/api/notify/lookup?token=secret-token-value")
    assert "secret-token-value" not in caplog.text
    assert "token=[redacted]" in caplog.text


def test_main_installs_the_access_log_filter():
    import main  # noqa: F401

    assert any(isinstance(f, RedactAccessLogFilter) for f in logging.getLogger("uvicorn.access").filters)
