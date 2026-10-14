"""Together AI completions model that also reports token usage.

`langchain_together.Together` returns only the generated text and drops the `usage`
block Together sends back, so LangSmith shows 0 tokens. This subclass makes the same
request and returns the token counts in `llm_output["token_usage"]`, where LangChain
callbacks and LangSmith read them.
"""
import logging
from typing import Any, Dict, List, Optional

import requests
from aiohttp import ClientSession, ClientTimeout
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.outputs import Generation, LLMResult
from langchain_together import Together

logger = logging.getLogger(__name__)

USAGE_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens")
REQUEST_TIMEOUT_SECONDS = 120


def _check_status(status: int, body: str) -> None:
    # Same messages as langchain_together; llm._is_transient_error matches "Error 5xx".
    if status >= 500:
        raise Exception(f"Together Server: Error {status}")
    if status >= 400:
        raise ValueError(f"Together received an invalid payload: {body}")
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
    usage = data.get("usage") or {}
    for key in USAGE_KEYS:
        value = usage.get(key)
        if isinstance(value, int):
            total[key] = total.get(key, 0) + value


class TogetherWithUsage(Together):
    """Drop-in replacement for `Together` that reports prompt/completion/total tokens."""

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
            headers, payload = self._request_parts(prompt, stop, kwargs)
            response = requests.post(
                self.base_url, json=payload, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS
            )
            _check_status(response.status_code, response.text)
            data = response.json()
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
                headers, payload = self._request_parts(prompt, stop, kwargs)
                async with session.post(self.base_url, json=payload, headers=headers) as response:
                    _check_status(response.status, await response.text())
                    data = await response.json()
                _warn_if_cut_off(data, self.model, self.max_tokens)
                texts.append(self._format_output(data))
                _add_usage(usage, data)
        return self._result(texts, usage)
