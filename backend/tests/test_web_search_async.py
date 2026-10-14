"""Tavily is a blocking client: every call, including the retry, must run off the event loop."""
import threading
import time

import pytest

from app.core.config import settings
from app.services.web_search import WebSearchService


class _FakeTavily:
    def __init__(self):
        self.calls = []

    def search(self, **kwargs):
        self.calls.append((kwargs["query"], threading.current_thread() is threading.main_thread()))
        if len(self.calls) == 1:
            raise RuntimeError("400 Bad Request")
        return {"results": [{"title": "t", "url": "https://example.com", "content": "c"}]}


@pytest.mark.asyncio
async def test_retry_after_400_runs_in_a_worker_thread():
    service = WebSearchService()
    fake = _FakeTavily()
    service.tavily_client = fake
    service.initialized = True

    results = await service.search("one two three four five six seven")

    assert len(results) == 1
    assert len(fake.calls) == 2
    assert fake.calls[1][0] == "one two three four five"
    assert [on_main for _query, on_main in fake.calls] == [False, False]


class _SlowTavily:
    """Blocks like a hung Tavily request; the first call may fail with a 400 to force the retry."""

    def __init__(self, fail_first: bool = False):
        self.fail_first = fail_first
        self.calls = 0

    def search(self, **kwargs):
        self.calls += 1
        if self.fail_first and self.calls == 1:
            raise RuntimeError("400 Bad Request")
        time.sleep(0.5)
        return {"results": [{"title": "late", "url": "https://example.com", "content": "c"}]}


def _service(client):
    service = WebSearchService()
    service.tavily_client = client
    service.initialized = True
    return service


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_first", [False, True], ids=["first call", "retry"])
async def test_slow_search_times_out_with_a_warning_and_no_results(fail_first, monkeypatch, caplog):
    monkeypatch.setattr(settings, "SEARCH_TIMEOUT_SECONDS", 0.05)
    client = _SlowTavily(fail_first=fail_first)

    started = time.monotonic()
    with caplog.at_level("WARNING"):
        results = await _service(client).search("one two three four five six seven")

    assert results == []
    assert time.monotonic() - started < 0.4
    assert client.calls == (2 if fail_first else 1)
    assert "timed out after 0.05 s" in caplog.text
