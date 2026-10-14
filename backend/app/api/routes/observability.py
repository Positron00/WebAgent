"""Observability API endpoints."""

from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException, Depends

from app.core.observability import get_all_metrics
from app.core.drift_detection import get_all_drift_stats, should_retrain_any_model
from app.services.llm_service import llm_service
from app.services.document_ingestion_service import doc_ingestion_service

router = APIRouter(
    prefix="/observability",
    tags=["observability"],
    responses={404: {"description": "Not found"}},
)

@router.get("/metrics", summary="Get all observability metrics")
async def get_metrics() -> Dict[str, Any]:
    """
    Retrieve all current observability metrics.
    
    Returns:
        Dictionary with metrics collected across the system
    """
    return get_all_metrics()

@router.get("/drift", summary="Get all drift detection statistics")
async def get_drift_stats() -> Dict[str, Any]:
    """
    Retrieve all current drift detection statistics.
    
    Returns:
        Dictionary with drift statistics collected across the system
    """
    # Get global drift stats
    global_stats = get_all_drift_stats()
    
    # Get service-specific drift stats
    llm_drift = llm_service.get_drift_status()
    doc_drift = doc_ingestion_service.get_drift_status()
    
    # Check if any models should be retrained
    retrain_recommendations = should_retrain_any_model()
    
    return {
        "global_drift_stats": global_stats,
        "llm_service_drift": llm_drift,
        "doc_service_drift": doc_drift,
        "retrain_recommendations": retrain_recommendations,
    }

@router.post("/llm/reset_baseline", summary="Reset LLM drift detection baseline")
async def reset_llm_baseline() -> Dict[str, Any]:
    """
    Reset the baseline for LLM drift detection.
    
    Returns:
        Confirmation message
    """
    llm_service.reset_drift_baselines()
    return {"status": "success", "message": "LLM drift detection baseline reset"}

@router.post("/doc/reset_baseline", summary="Reset document processing drift detection baseline")
async def reset_doc_baseline() -> Dict[str, Any]:
    """
    Reset the baseline for document processing drift detection.
    
    Returns:
        Confirmation message
    """
    doc_ingestion_service.reset_drift_baselines()
    return {"status": "success", "message": "Document processing drift detection baseline reset"}

@router.get("/alerts", summary="Get current alerts based on observability metrics")
async def get_alerts() -> List[Dict[str, Any]]:
    """
    Get a list of current alerts based on observability metrics.
    
    Returns:
        List of alerts with severity, message, and timestamp
    """
    alerts = []
    
    # Get drift stats
    drift_stats = get_all_drift_stats()
    
    # Check LLM service
    llm_drift = llm_service.get_drift_status()
    if llm_drift.get("retrain_recommended", False):
        alerts.append({
            "severity": "high",
            "service": "llm",
            "message": f"Model {llm_drift.get('model_name', 'unknown')} requires retraining due to performance drift",
            "timestamp": llm_drift.get("performance_drift", {}).get("last_drift_time")
        })
    
    # Check document service
    doc_drift = doc_ingestion_service.get_drift_status()
    if doc_drift.get("pipeline_adjustments_recommended", False):
        alerts.append({
            "severity": "medium",
            "service": "document_processing",
            "message": "Document processing pipeline requires adjustments due to quality drift",
            "timestamp": doc_drift.get("doc_quality_drift", {}).get("last_drift_time")
        })
    
    # Get global metrics
    metrics = get_all_metrics()
    
    # Check for high error rates in LLM service
    llm_error_rate = metrics.get("model_metrics", {}).get("gpt-4_error_rate", 0)
    if llm_error_rate > 0.05:  # More than 5% error rate
        alerts.append({
            "severity": "high",
            "service": "llm",
            "message": f"High LLM error rate detected: {llm_error_rate:.2%}",
            "timestamp": None
        })
    
    # Check for high token usage
    token_usage = metrics.get("model_metrics", {}).get("gpt-4_total_tokens", 0)
    if token_usage > 1000000:  # Arbitrary high token usage threshold
        alerts.append({
            "severity": "medium",
            "service": "llm",
            "message": f"High token usage detected: {token_usage} tokens",
            "timestamp": None
        })
    
    # Check for high document processing error rate
    doc_error_rate = metrics.get("document_metrics", {}).get("document_processing_error_rate", 0)
    if doc_error_rate > 0.1:  # More than 10% error rate
        alerts.append({
            "severity": "high",
            "service": "document_processing",
            "message": f"High document processing error rate detected: {doc_error_rate:.2%}",
            "timestamp": None
        })
    
    return alerts

@router.get("/health", summary="Get system health status based on observability metrics")
async def get_health() -> Dict[str, Any]:
    """
    Get the current health status of the system based on observability metrics.
    
    Returns:
        Health status with overall status and component statuses
    """
    metrics = get_all_metrics()
    alerts = await get_alerts()
    
    # Determine severity levels
    high_severity_count = sum(1 for alert in alerts if alert["severity"] == "high")
    medium_severity_count = sum(1 for alert in alerts if alert["severity"] == "medium")
    low_severity_count = sum(1 for alert in alerts if alert["severity"] == "low")
    
    # Determine overall health status
    if high_severity_count > 0:
        overall_status = "unhealthy"
    elif medium_severity_count > 0:
        overall_status = "degraded"
    else:
        overall_status = "healthy"
    
    # Get model metrics
    llm_metrics = {k: v for k, v in metrics.get("model_metrics", {}).items() if k.startswith("gpt-4")}
    
    # Get document metrics
    doc_metrics = metrics.get("document_metrics", {})
    
    # Get system metrics
    system_metrics = metrics.get("system_metrics", {})
    
    return {
        "overall_status": overall_status,
        "alert_counts": {
            "high": high_severity_count,
            "medium": medium_severity_count,
            "low": low_severity_count,
            "total": len(alerts)
        },
        "components": {
            "llm_service": {
                "status": "unhealthy" if any(alert["service"] == "llm" and alert["severity"] == "high" for alert in alerts) else 
                          "degraded" if any(alert["service"] == "llm" and alert["severity"] == "medium" for alert in alerts) else 
                          "healthy",
                "metrics": llm_metrics
            },
            "document_service": {
                "status": "unhealthy" if any(alert["service"] == "document_processing" and alert["severity"] == "high" for alert in alerts) else 
                          "degraded" if any(alert["service"] == "document_processing" and alert["severity"] == "medium" for alert in alerts) else 
                          "healthy",
                "metrics": doc_metrics
            },
            "system": {
                "status": "healthy",  # Assume system is healthy by default
                "metrics": system_metrics
            }
        }
    } 