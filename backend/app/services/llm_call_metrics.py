"""Prometheus metrics for each model call, recorded where the call happens.

Before 3.5.3 these came from a tracker wrapped around building the LLM client, so the
"latency" was client set-up time, tokens were always 0, and `llm_requests_total`,
`llm_request_duration_seconds` and `llm_tokens_total` were never written: the Grafana
model panels stayed empty. Now the Together client (and the self-hosted model) call
`record` once per request, success or failure.
"""
import logging
from typing import Any, Dict, Optional

from app.core.metrics import LLM_REQUEST_COUNT, LLM_REQUEST_LATENCY, LLM_TOKEN_COUNT
from app.core.observability import MODEL_PREDICTION_COUNT, record_model_latency, record_model_token_usage

logger = logging.getLogger(__name__)


def record(provider: str, model: str, seconds: float, ok: bool, usage: Optional[Dict[str, Any]] = None) -> None:
    """Count one model request; never raises (metrics must not break a call)."""
    try:
        status = "success" if ok else "error"
        model = model or "unknown"
        LLM_REQUEST_COUNT.labels(provider=provider, model=model, status=status).inc()
        LLM_REQUEST_LATENCY.labels(provider=provider, model=model).observe(seconds)
        MODEL_PREDICTION_COUNT.labels(model_name=provider, task_type="chat", status=status).inc()
        if not ok:
            return
        record_model_latency(provider, seconds, task_type="chat")
        usage = usage or {}
        prompt = usage.get("prompt_tokens") if isinstance(usage.get("prompt_tokens"), int) else 0
        completion = usage.get("completion_tokens") if isinstance(usage.get("completion_tokens"), int) else 0
        if prompt:
            LLM_TOKEN_COUNT.labels(provider=provider, model=model, type="prompt").inc(prompt)
        if completion:
            LLM_TOKEN_COUNT.labels(provider=provider, model=model, type="completion").inc(completion)
        record_model_token_usage(provider, prompt, completion)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Model call metrics not recorded: %s", exc)
