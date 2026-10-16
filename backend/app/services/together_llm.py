"""Together AI completions model that also reports token usage.

`langchain_together.Together` returns only the generated text and drops the `usage`
block Together sends back, so LangSmith shows 0 tokens. This subclass makes the same
request and returns the token counts in `llm_output["token_usage"]`, where LangChain
callbacks and LangSmith read them.
"""
import logging
import os
import time
from typing import Any, Dict, List, Optional

import requests
from aiohttp import ClientSession, ClientTimeout
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.outputs import Generation, LLMResult
from langchain_together import Together
from pydantic import Field

from app.services import llm_call_metrics, task_budget

logger = logging.getLogger(__name__)

USAGE_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens")
REQUEST_TIMEOUT_SECONDS = 120


# Base of Together's API; the client posts to <base>/completions (OBS-021). A test or the local
# check can point it at another server, e.g. scripts/fake_together.py.
TOGETHER_API_BASE_DEFAULT = "https://api.together.xyz/v1"


def together_api_base() -> str:
    return (os.getenv("TOGETHER_API_BASE") or TOGETHER_API_BASE_DEFAULT).rstrip("/")


class TogetherServerError(Exception):
    """A 5xx or 429 reply: temporary, retried by llm._is_transient_error ("Error 5xx" / "Error 429")."""

    def __init__(self, message: str, status: int):
        super().__init__(message)
        self.status = status


class TogetherRequestError(ValueError):
    """Any other 4xx reply: a bad request, not retried."""

    def __init__(self, message: str, status: int):
        super().__init__(message)
        self.status = status


def _check_status(status: int, body: str) -> None:
    # Same messages as langchain_together; llm._is_transient_error matches "Error 5xx".
    if status >= 500:
        raise TogetherServerError(f"Together Server: Error {status}", status)
    if status == 429:
        # A rate limit is temporary: "Error 429" makes llm._is_transient_error retry it. It used
        # to fall into the 4xx branch below as "invalid payload" and fail the task at once.
        raise TogetherServerError(f"Together Server: Error 429 (rate limited): {body[:300]}", status)
    if status >= 400:
        raise TogetherRequestError(f"Together received an invalid payload: {body}", status)
    if status != 200:
        raise Exception(f"Together returned an unexpected response with status {status}: {body}")


def _warn_if_cut_off(data: Dict[str, Any], model: str, max_tokens: Any) -> None:
    """A reply that stopped at the token limit is incomplete; say so instead of failing silently."""
    for choice in data.get("choices") or []:
        if choice.get("finish_reason") == "length":
            logger.warning(
                f"Together reply from {model} stopped at the max_tokens limit ({max_tokens}); "
                "the text is cut off. Raise max_tokens for this agent if this happens often."
            )
            return


def _add_usage(total: Dict[str, int], data: Dict[str, Any]) -> None:
    """Add one reply's usage to `total` and to the run's token budget."""
    usage = data.get("usage") or {}
    for key in USAGE_KEYS:
        value = usage.get(key)
        if isinstance(value, int):
            total[key] = total.get(key, 0) + value
    task_budget.add(usage.get("total_tokens", 0))


class TogetherWithUsage(Together):
    """Drop-in replacement for `Together` that reports prompt/completion/total tokens."""

    # Who made the call (an agent name, "judge", "triage" or "other"): keys the drift samples.
    call_label: str = "other"
    base_url: str = Field(default_factory=lambda: f"{together_api_base()}/completions")

    def _request_parts(self, prompt: str, stop: Optional[List[str]], kwargs: Dict[str, Any]):
        headers = {
            "Authorization": f"Bearer {self.together_api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }
        stop_to_use = stop[0] if stop and len(stop) == 1 else stop
        payload = {**self.default_params, "prompt": prompt, "stop": stop_to_use, **kwargs}
        return headers, {k: v for k, v in payload.items() if v is not None}

    def _result(self, texts: List[str], usage: Dict[str, int]) -> LLMResult:
        logger.debug(f"Together usage for {self.model}: {usage}")
        return LLMResult(
            generations=[[Generation(text=text)] for text in texts],
            llm_output={"token_usage": usage, "model_name": self.model},
        )

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        texts, usage = [], {}
        for prompt in prompts:
            task_budget.check()
            headers, payload = self._request_parts(prompt, stop, kwargs)
            started = time.perf_counter()
            try:
                response = requests.post(
                    self.base_url, json=payload, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS
                )
                _check_status(response.status_code, response.text)
                data = response.json()
            except Exception as error:
                llm_call_metrics.record("together", self.model, time.perf_counter() - started, ok=False,
                                        call_label=self.call_label, status_code=getattr(error, "status", None))
                raise
            llm_call_metrics.record("together", self.model, time.perf_counter() - started, ok=True,
                                    usage=data.get("usage"), call_label=self.call_label)
            _warn_if_cut_off(data, self.model, self.max_tokens)
            texts.append(self._format_output(data))
            _add_usage(usage, data)
        return self._result(texts, usage)

    async def _agenerate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        texts, usage = [], {}
        async with ClientSession(timeout=ClientTimeout(total=REQUEST_TIMEOUT_SECONDS)) as session:
            for prompt in prompts:
                task_budget.check()
                headers, payload = self._request_parts(prompt, stop, kwargs)
                started = time.perf_counter()
                try:
                    async with session.post(self.base_url, json=payload, headers=headers) as response:
                        _check_status(response.status, await response.text())
                        data = await response.json()
                except Exception as error:
                    llm_call_metrics.record("together", self.model, time.perf_counter() - started, ok=False,
                                            call_label=self.call_label, status_code=getattr(error, "status", None))
                    raise
                llm_call_metrics.record("together", self.model, time.perf_counter() - started, ok=True,
                                        usage=data.get("usage"), call_label=self.call_label)
                _warn_if_cut_off(data, self.model, self.max_tokens)
                texts.append(self._format_output(data))
                _add_usage(usage, data)
        return self._result(texts, usage)
