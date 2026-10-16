"""Observability alerts and figures computed from drift samples (OBS-012, OBS-013).

Pure functions over the samples and detector results the drift store reads; the store calls them
once per compute tick and keeps the result in ``state.json``.
"""
from collections import Counter
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, NamedTuple, Optional

ALERTS = {  # name -> (severity, service)
    "model_drift": ("warning", "llm"),
    "model_errors": ("critical", "llm"),
    "rate_limited": ("warning", "llm"),
    "token_usage": ("warning", "llm"),
    "document_drift": ("warning", "document_processing"),
    "document_errors": ("critical", "document_processing"),
}
MODEL_MIN_REQUESTS = 20
MODEL_ERROR_SHARE = 0.05
RATE_LIMITED_SHARE = 0.20
DOC_MIN_LOADS = 10
DOC_ERROR_SHARE = 0.10


class Sample(NamedTuple):
    ts: datetime
    detector: str
    value: Any = None  # a number or short string; None for an embedding aggregate
    agg: Optional[Dict[str, Any]] = None


def _since(samples: Iterable[Sample], detector: str, start: datetime) -> List[Sample]:
    return [s for s in samples if s.detector == detector and s.ts > start]


def figures(samples: List[Sample], now: datetime) -> Dict[str, Any]:
    """Model requests over the last hour, tokens over the last day, document loads over the last day."""
    hour, day = now - timedelta(hours=1), now - timedelta(days=1)
    outcomes = Counter(s.value for s in _since(samples, "llm_outcome", hour))
    by_label: Dict[str, Dict[str, int]] = {}
    tokens_24h = 0
    for s in samples:
        if s.ts <= day:
            continue
        label = s.detector.split(":", 1)[1] if ":" in s.detector else None
        if s.detector.startswith("llm_tokens:"):
            tokens_24h += int(s.value)
            by_label.setdefault(label, {"requests_24h": 0, "tokens_24h": 0})
            by_label[label]["tokens_24h"] += int(s.value)
            by_label[label]["requests_24h"] += 1
    docs = Counter(s.value for s in _since(samples, "doc_outcome", day))
    return {
        "llm": {
            "window": "1h",
            "requests": sum(outcomes.values()),
            "errors": outcomes.get("error", 0),
            "rate_limited": outcomes.get("rate_limited", 0),
            "tokens_24h": tokens_24h,
            "by_label": by_label,
        },
        "doc": {"window": "24h", "loads": sum(docs.values()), "errors": docs.get("error", 0)},
    }


def _pct(share: float) -> str:
    return f"{share:.0%}"


def evaluate(figs: Dict[str, Any], detectors: Dict[str, Dict[str, Any]], token_limit: int) -> List[Dict[str, str]]:
    """Active alerts (name, severity, service, message), without `since`."""
    active: List[Dict[str, str]] = []

    def fire(name: str, message: str) -> None:
        severity, service = ALERTS[name]
        active.append({"name": name, "severity": severity, "service": service, "message": message})

    drifting = sorted(name for name, d in detectors.items() if d.get("drift"))
    model = [n for n in drifting if n.startswith(("llm_speed:", "llm_tokens:", "embeddings:"))]
    docs = [n for n in drifting if n.startswith("doc_")]
    if model:
        fire("model_drift", f"Model behaviour changed ({', '.join(model)}). Check for a provider, model "
                            "or prompt change, then reset the LLM baseline once it's understood.")
    llm = figs["llm"]
    counted = llm["requests"] - llm["rate_limited"]
    if counted >= MODEL_MIN_REQUESTS and llm["errors"] / counted > MODEL_ERROR_SHARE:
        fire("model_errors", f"{_pct(llm['errors'] / counted)} of model requests failed in the last hour "
                             f"({counted} requests, rate limits excluded).")
    if llm["requests"] >= MODEL_MIN_REQUESTS and llm["rate_limited"] / llm["requests"] > RATE_LIMITED_SHARE:
        fire("rate_limited", f"{_pct(llm['rate_limited'] / llm['requests'])} of model requests were "
                             "rate-limited in the last hour. Together is busy or the account is near its limits.")
    if token_limit > 0 and llm["tokens_24h"] > token_limit:
        fire("token_usage", f"{llm['tokens_24h']} tokens used in the last 24 hours (alert set at {token_limit}).")
    if docs:
        fire("document_drift", f"Documents changed ({', '.join(docs)}). Check what is being uploaded, then "
                               "reset the document baseline once it's understood.")
    doc = figs["doc"]
    if doc["loads"] >= DOC_MIN_LOADS and doc["errors"] / doc["loads"] > DOC_ERROR_SHARE:
        fire("document_errors", f"{_pct(doc['errors'] / doc['loads'])} of document loads failed in the last "
                                f"24 hours ({doc['loads']} loads).")
    return active
