"""LLM drift status for /api/v1/observability (a facade over the shared drift store, 3.6.0).

The drift itself comes from real model calls (``llm_call_metrics.record``) and embeddings
(``drift_embeddings``); this module only shapes the stored state the way 3.5.x clients expect.
"""
from typing import Any, Dict, List, Optional

from app.core import drift_store
from app.core.config import settings
from app.core.drift_reference import group_of, public


def _detectors(group: str) -> Dict[str, Dict[str, Any]]:
    if not settings.DRIFT_ENABLED:
        return {}
    return {n: d for n, d in drift_store.load_state().get("detectors", {}).items() if group_of(n) == group}


def _worst(detectors: Dict[str, Dict[str, Any]], prefixes: tuple) -> Optional[Dict[str, Any]]:
    """The detector with the highest score among those named with one of ``prefixes``, or None."""
    matching: List[tuple] = [(n, d) for n, d in detectors.items() if n.startswith(prefixes)]
    if not matching:
        return None
    name, det = max(matching, key=lambda item: item[1].get("score") or 0.0)
    return {"detector": name, **public(det)}


def drift_summary(group: str, kinds: Dict[str, tuple]) -> Dict[str, Any]:
    """Shared shape of the two facades: baseline status plus the worst detector per kind."""
    detectors = _detectors(group)
    return {
        "collecting_baseline": not detectors or any("reference" not in d and "reference_centroid" not in d
                                                    for d in detectors.values()),
        "baseline_count": max((d.get("reference_count", 0) for d in detectors.values()), default=0),
        **{key: _worst(detectors, prefixes) for key, prefixes in kinds.items()},
    }


class LLMService:
    """Drift status and baseline reset for the model detectors (group ``llm``)."""

    def __init__(self, model_name: str = ""):
        self.model_name = model_name or settings.LLM_DEFAULT_PROVIDER

    def get_drift_status(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            **drift_summary("llm", {"latency_drift": ("llm_speed:",), "token_usage_drift": ("llm_tokens:",),
                                    "embedding_drift": ("embeddings:",)}),
            "performance_drift": None,  # no quality signal is measured (decision 9)
            "retrain_recommended": False,
        }

    def reset_drift_baselines(self) -> None:
        drift_store.reset("llm")


llm_service = LLMService()
