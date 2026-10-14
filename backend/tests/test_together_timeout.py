"""Async Together calls time out like sync ones (IMP-040), tested against a slow local server."""
import asyncio
import time

import pytest
from aiohttp import web

from app.services import together_llm
from app.services.llm import _is_transient_error
from app.services.together_llm import TogetherWithUsage

REPLY = {"choices": [{"text": "ok"}], "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}


async def _start_server(delay_seconds: float):
    async def completions(request):
        await asyncio.sleep(delay_seconds)
        return web.json_response(REPLY)

    app = web.Application()
    app.router.add_post("/v1/completions", completions)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    return runner, f"http://127.0.0.1:{port}/v1/completions"


def _model(url):
    return TogetherWithUsage(model="meta-models/Muse-Glimmer-30B", max_tokens=5,
                             together_api_key="test-key", base_url=url)


@pytest.mark.asyncio
async def test_slow_server_times_out_and_counts_as_transient(monkeypatch):
    monkeypatch.setattr(together_llm, "REQUEST_TIMEOUT_SECONDS", 0.3)
    runner, url = await _start_server(delay_seconds=5)
    try:
        started = time.monotonic()
        with pytest.raises(asyncio.TimeoutError) as caught:
            await _model(url).ainvoke("hello")
        assert time.monotonic() - started < 3
        assert _is_transient_error(caught.value)
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_fast_server_still_answers(monkeypatch):
    monkeypatch.setattr(together_llm, "REQUEST_TIMEOUT_SECONDS", 2)
    runner, url = await _start_server(delay_seconds=0)
    try:
        assert await _model(url).ainvoke("hello") == "ok"
    finally:
        await runner.cleanup()
