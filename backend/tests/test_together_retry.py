"""Tests for retrying temporary Together AI failures."""
import pytest

from app.services import llm
from app.services.llm import TogetherWrapper


class FlakyLLM:
    """Stand-in Together client that fails a set number of times, then answers."""

    def __init__(self, failures, error_message="Together Server: Error 503"):
        self.failures = failures
        self.error_message = error_message
        self.calls = 0

    def _call(self):
        self.calls += 1
        if self.calls <= self.failures:
            raise Exception(self.error_message)
        return "answer"

    def invoke(self, input, config=None):
        return self._call()

    async def ainvoke(self, input, config=None):
        return self._call()


def make_wrapper(fake):
    wrapper = object.__new__(TogetherWrapper)
    wrapper.llm = fake
    return wrapper


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    async def instant_sleep(_seconds):
        return None

    monkeypatch.setattr(llm, "TOGETHER_RATE_LIMIT_BUFFER", 0.0)
    monkeypatch.setattr(llm.asyncio, "sleep", instant_sleep)
    monkeypatch.setattr(llm.time, "sleep", lambda _seconds: None)


async def test_async_call_retries_temporary_errors():
    fake = FlakyLLM(failures=2)
    assert await make_wrapper(fake).ainvoke("q") == "answer"
    assert fake.calls == 3


def test_sync_call_retries_temporary_errors():
    fake = FlakyLLM(failures=1)
    assert make_wrapper(fake).invoke("q") == "answer"
    assert fake.calls == 2


async def test_gives_up_after_max_attempts():
    fake = FlakyLLM(failures=10)
    with pytest.raises(Exception, match="Error 503"):
        await make_wrapper(fake).ainvoke("q")
    assert fake.calls == llm.TOGETHER_MAX_ATTEMPTS


async def test_does_not_retry_permanent_errors():
    fake = FlakyLLM(failures=10, error_message="Together Server: Error 401 invalid API key")
    with pytest.raises(Exception, match="Error 401"):
        await make_wrapper(fake).ainvoke("q")
    assert fake.calls == 1
