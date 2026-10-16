"""Simulated drift and unused builders are gone; the self-hosted server is an example (OBS-019)."""
import re
from pathlib import Path

from app.core import drift_detection
from app.services import llm, llm_service

APP = Path(__file__).resolve().parents[1] / "app"


def test_no_simulation_left_in_the_services():
    assert not hasattr(llm_service.LLMService, "generate_response")
    assert not hasattr(llm_service.LLMService, "get_embedding")
    from app.services.document_ingestion_service import DocumentProcessingObservability

    assert not hasattr(DocumentProcessingObservability, "process_document")
    for path in APP.rglob("*.py"):
        if "examples" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        assert "simulated accuracy" not in text and "Simulated latency" not in text, path


def test_detector_registry_and_unused_builder_are_removed():
    for name in ("register_drift_detector", "get_all_drift_stats", "should_retrain_any_model",
                 "create_numerical_drift_detector", "NumericalDriftDetector"):
        assert not hasattr(drift_detection, name), name
    assert not hasattr(llm, "_build_llm_without_tags")
    assert not re.search(r'record_data_drift\(\s*"llm_response"', (APP / "services/llm.py").read_text())


def test_self_hosted_server_is_an_example():
    assert not (APP / "services/self_hosted_llm_service.py").exists()
    assert (APP / "services/examples/self_hosted_llm_service.py").exists()
    assert "services/examples/self_hosted_llm_service.py" in (APP / "services/examples/model_fine_tuning.py").read_text()
