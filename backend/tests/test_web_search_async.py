"""Tavily is a blocking client: every call, including the retry, must run off the event loop."""
import threading

import pytest

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
