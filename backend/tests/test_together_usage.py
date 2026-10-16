"""Tests that the Together model reports token usage, with HTTP replies faked."""
import pytest
from langchain_core.callbacks import BaseCallbackHandler

from app.services import together_llm
from app.services.llm import _is_transient_error
from app.services.together_llm import TogetherWithUsage

REPLY = {
    "choices": [{"text": "Tuples are immutable."}],
    "usage": {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17, "prompt_tokens_details": {}},
}


class FakeResponse:
    """Like requests.Response."""

    def __init__(self, status=200, data=REPLY):
        self.status_code = status
        self._data = data
        self.text = "error body"

    def json(self):
        return self._data


class FakeAsyncResponse:
    """Like aiohttp.ClientResponse."""

    def __init__(self, status=200, data=REPLY):
        self.status = status
        self._data = data

    async def text(self):
        return "error body"

    async def json(self):
        return self._data

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class FakeSession:
    def __init__(self, response):
        self.response = response

    def post(self, *args, **kwargs):
        return self.response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class UsageRecorder(BaseCallbackHandler):
    def __init__(self):
        self.llm_outputs = []

    def on_llm_end(self, response, **kwargs):
        self.llm_outputs.append(response.llm_output)


def make_llm():
    return TogetherWithUsage(model="meta-models/Muse-Glimmer-30B", max_tokens=50, together_api_key="test-key")


def test_sync_call_returns_text_and_reports_usage(monkeypatch):
    monkeypatch.setattr(together_llm.requests, "post", lambda *a, **k: FakeResponse())
    recorder = UsageRecorder()

    text = make_llm().invoke("Lists vs tuples?", config={"callbacks": [recorder]})

    assert text == "Tuples are immutable."
    assert recorder.llm_outputs == [
        {
            "token_usage": {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17},
            "model_name": "meta-models/Muse-Glimmer-30B",
        }
    ]


async def test_async_call_returns_text_and_reports_usage(monkeypatch):
    monkeypatch.setattr(together_llm, "ClientSession", lambda **kw: FakeSession(FakeAsyncResponse()))
    recorder = UsageRecorder()

    text = await make_llm().ainvoke("Lists vs tuples?", config={"callbacks": [recorder]})

    assert text == "Tuples are immutable."
    assert recorder.llm_outputs[0]["token_usage"]["total_tokens"] == 17


def test_missing_usage_is_reported_as_empty(monkeypatch):
    monkeypatch.setattr(together_llm.requests, "post", lambda *a, **k: FakeResponse(data={"choices": [{"text": "x"}]}))
    recorder = UsageRecorder()

    make_llm().invoke("q", config={"callbacks": [recorder]})

    assert recorder.llm_outputs[0]["token_usage"] == {}


def test_server_errors_keep_the_text_the_retry_logic_looks_for(monkeypatch):
    monkeypatch.setattr(together_llm.requests, "post", lambda *a, **k: FakeResponse(status=503))

    with pytest.raises(Exception, match="Together Server: Error 503") as error:
        make_llm().invoke("q")
    assert _is_transient_error(error.value)


async def test_bad_requests_are_not_retried(monkeypatch):
    monkeypatch.setattr(together_llm, "ClientSession", lambda **kw: FakeSession(FakeAsyncResponse(status=400)))

    with pytest.raises(ValueError, match="invalid payload: error body") as error:
        await make_llm().ainvoke("q")
    assert not _is_transient_error(error.value)


def test_reply_cut_off_at_the_token_limit_is_logged(monkeypatch, caplog):
    cut_off = {"choices": [{"text": "partial", "finish_reason": "length"}], "usage": {}}
    monkeypatch.setattr(together_llm.requests, "post", lambda *a, **k: FakeResponse(data=cut_off))

    with caplog.at_level("WARNING", logger=together_llm.logger.name):
        make_llm().invoke("Long question")

    assert "stopped at the max_tokens limit (50)" in caplog.text


def test_complete_reply_is_not_flagged(monkeypatch, caplog):
    done = {"choices": [{"text": "full", "finish_reason": "stop"}], "usage": {}}
    monkeypatch.setattr(together_llm.requests, "post", lambda *a, **k: FakeResponse(data=done))

    with caplog.at_level("WARNING", logger=together_llm.logger.name):
        make_llm().invoke("Short question")

    assert "max_tokens limit" not in caplog.text


async def test_rate_limits_are_retried(monkeypatch):
    # Seen on the 3.5.2 stack: Together answered 429 and the direct answer failed at once,
    # reported as "invalid payload", because 4xx counted as a bad request.
    monkeypatch.setattr(together_llm, "ClientSession", lambda **kw: FakeSession(FakeAsyncResponse(status=429)))

    with pytest.raises(Exception, match="Error 429") as error:
        await make_llm().ainvoke("q")
    assert _is_transient_error(error.value)
    assert "invalid payload" not in str(error.value)


def _requests(status):
    from app.core.metrics import WEBAGENT_REGISTRY

    return WEBAGENT_REGISTRY.get_sample_value(
        "llm_requests_total", {"provider": "together", "model": make_llm().model, "status": status}
    ) or 0


def test_each_call_is_counted_with_its_tokens(monkeypatch):
    # Before 3.5.3 llm_requests_total and llm_tokens_total were never written in production.
    from app.core.metrics import WEBAGENT_REGISTRY

    monkeypatch.setattr(together_llm.requests, "post", lambda *a, **k: FakeResponse())
    tokens = {"provider": "together", "model": make_llm().model, "type": "completion"}
    before_ok = _requests("success")
    before_tokens = WEBAGENT_REGISTRY.get_sample_value("llm_tokens_total", tokens) or 0
    make_llm().invoke("q")
    assert _requests("success") == before_ok + 1
    assert (WEBAGENT_REGISTRY.get_sample_value("llm_tokens_total", tokens) or 0) > before_tokens


async def test_failed_calls_are_counted_as_errors(monkeypatch):
    monkeypatch.setattr(together_llm, "ClientSession", lambda **kw: FakeSession(FakeAsyncResponse(status=429)))
    before = _requests("error")
    with pytest.raises(Exception):
        await make_llm().ainvoke("q")
    assert _requests("error") == before + 1

