"""Observability API endpoints, read from the shared drift store (OBS-013).

Every worker answers from ``data/drift/state.json``, so all workers and restarts agree. With
``WEBAGENT_DRIFT_ENABLED=false`` the routes answer as if nothing had been recorded.
"""
from datetime import datetime
from typing import Any, Dict, List

from fastapi import APIRouter
from starlette.concurrency import run_in_threadpool

from app.core import drift_store
from app.core.config import settings
from app.core.drift_reference import public
from app.core.observability import get_observability_stats
from app.services.document_ingestion_service import doc_ingestion_service
from app.services.llm_service import llm_service

router = APIRouter(
    prefix="/observability",
    tags=["observability"],
    responses={404: {"description": "Not found"}},
)

def _state() -> Dict[str, Any]:
    if not settings.DRIFT_ENABLED:
        return drift_store._empty_state()
    return drift_store.load_state()


def _model_figures(figures: Dict[str, Any]) -> Dict[str, Any]:
    llm = figures.get("llm") or {}
    return {
        "provider": settings.LLM_DEFAULT_PROVIDER,
        "window": llm.get("window", "1h"),
        "requests": llm.get("requests", 0),
        "errors": llm.get("errors", 0),
        "rate_limited": llm.get("rate_limited", 0),
        "tokens_24h": llm.get("tokens_24h", 0),
        "by_label": llm.get("by_label", {}),
    }


def _document_figures(figures: Dict[str, Any]) -> Dict[str, Any]:
    doc = figures.get("doc") or {}
    return {"window": doc.get("window", "24h"), "loads": doc.get("loads", 0), "errors": doc.get("errors", 0)}


def _status(alerts: List[Dict[str, Any]]) -> str:
    severities = {a["severity"] for a in alerts}
    return "unhealthy" if "critical" in severities else "degraded" if "warning" in severities else "healthy"


@router.get("/metrics", summary="Get all observability metrics")
async def get_metrics() -> Dict[str, Any]:
    """Model and document figures from the drift store (Prometheus remains the source of truth)."""
    figures = _state().get("figures", {})
    return {
        "model_metrics": _model_figures(figures),
        "document_metrics": _document_figures(figures),
        "pipeline_metrics": {},
        "data_quality_metrics": {},
        "system_metrics": get_observability_stats(),
        "timestamp": datetime.now().isoformat(),
    }


@router.get("/drift", summary="Get all drift detection statistics")
async def get_drift_stats() -> Dict[str, Any]:
    """Every detector (no stored reference values), plus the 3.5.x keys."""
    state = _state()
    detectors = {name: public(det) for name, det in state.get("detectors", {}).items()}
    return {
        "computed_at": state.get("computed_at"),
        "detectors": detectors,
        "global_drift_stats": detectors,
        "llm_service_drift": llm_service.get_drift_status(),
        "doc_service_drift": doc_ingestion_service.get_drift_status(),
        "retrain_recommendations": [],  # kept for existing clients; nothing is retrained
    }


@router.post("/llm/reset_baseline", summary="Reset LLM drift detection baseline")
async def reset_llm_baseline() -> Dict[str, Any]:
    """Start new references for the model detectors from the next samples."""
    await run_in_threadpool(llm_service.reset_drift_baselines)
    return {"status": "success", "message": "LLM drift detection baseline reset"}


@router.post("/doc/reset_baseline", summary="Reset document processing drift detection baseline")
async def reset_doc_baseline() -> Dict[str, Any]:
    """Start new references for the document detectors from the next samples."""
    await run_in_threadpool(doc_ingestion_service.reset_drift_baselines)
    return {"status": "success", "message": "Document processing drift detection baseline reset"}


@router.get("/alerts", summary="Get current alerts based on observability metrics")
async def get_alerts() -> List[Dict[str, Any]]:
    """Active alerts: name, severity, service, message, since (and timestamp, equal to since)."""
    return [
        {"name": a["name"], "severity": a["severity"], "service": a["service"], "message": a["message"],
         "since": a.get("since"), "timestamp": a.get("since")}
        for a in _state().get("alerts", [])
    ]


@router.get("/health", summary="Get system health status based on observability metrics")
async def get_health() -> Dict[str, Any]:
    """Overall status from the active alerts (critical → unhealthy, warning → degraded)."""
    alerts = await get_alerts()
    figures = _state().get("figures", {})
    counts = {sev: sum(a["severity"] == sev for a in alerts) for sev in ("critical", "warning")}
    services = {
        "llm": {"status": _status([a for a in alerts if a["service"] == "llm"]),
                "metrics": _model_figures(figures)},
        "document_processing": {"status": _status([a for a in alerts if a["service"] == "document_processing"]),
                                "metrics": _document_figures(figures)},
    }
    return {
        "overall_status": _status(alerts),
        # high/medium/low are the 3.5.x names for critical/warning/(none)
        "alert_counts": {**counts, "high": counts["critical"], "medium": counts["warning"], "low": 0,
                         "total": len(alerts)},
        "services": services,
        "components": {  # 3.5.x shape
            "llm_service": services["llm"],
            "document_service": services["document_processing"],
            "system": {"status": "healthy", "metrics": get_observability_stats()},
        },
    }
