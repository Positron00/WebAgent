"""Settings that must be right before the server accepts traffic in uat/prod.

Run as `python -m app.core.startup_checks` before starting uvicorn (the Docker image does this):
with several workers, uvicorn restarts a worker that fails at startup, so a failure inside the
app alone would loop forever instead of stopping the container.
"""
import logging
import os
import sys

from app.core.config import settings
from app.core.security import INSECURE_SECRET_DEFAULT
from app.security.context_defender import (
    encoder_files_present as defender_encoder_files_present,
    gate_passed as defender_gate_passed,
    model_files_present as defender_model_files_present,
)
from app.services.triage_model import encoder_available, gate_passed, model_files_present
from app.utils.code_sandbox import sandbox_mode
from app.security.decision import PolicyLoadError, get_engine


def server_code_execution_enabled() -> bool:
    """The API server runs generated code only when explicitly enabled."""
    return os.getenv("WEBAGENT_ENABLE_CODE_EXECUTION", "false").lower() == "true"


def check_triage_files() -> None:
    """In uat/prod with triage enabled, the classifier and encoder must be on disk."""
    cfg = getattr(settings, "TRIAGE", None)
    if cfg is None or not cfg.enabled:
        return
    if settings.WEBAGENT_ENV not in ("uat", "prod"):
        return
    if not model_files_present():
        raise RuntimeError(
            "Triage classifier files missing in uat/prod (models/triage/classifier.joblib, "
            "metadata.json). Set WEBAGENT_TRIAGE_ENABLED=false or ship the model."
        )
    if not gate_passed():
        raise RuntimeError(
            "The triage classifier in models/triage has not passed the release gate "
            "(metadata.json test.passed is not true; run scripts/eval_triage.py --write-metadata). "
            "Set WEBAGENT_TRIAGE_ENABLED=false or ship a version that passes."
        )
    if not encoder_available():
        raise RuntimeError(
            "The triage encoder is missing: the fine-tuned folder named by metadata.json \"encoder_path\" "
            f"(models/triage/encoder/), or for a frozen classifier {cfg.encoder_name} in "
            "SENTENCE_TRANSFORMERS_HOME or the Hugging Face cache. Set WEBAGENT_TRIAGE_ENABLED=false or ship it."
        )


def check_security_guard() -> None:
    """In uat/prod with the guard enabled, Cedar policies must load and pass tests."""
    cfg = getattr(settings, "SECURITY_GUARD", None)
    if cfg is None or not cfg.enabled:
        return
    if settings.WEBAGENT_ENV not in ("uat", "prod"):
        return
    try:
        get_engine(force_reload=True)
    except PolicyLoadError as exc:
        raise RuntimeError(f"Security guard policies failed to load: {exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"Security guard policy check failed: {exc}") from exc


def check_security_guard_files() -> None:
    """In uat/prod with the guard enabled, policies are required; model gate when artifacts exist."""
    cfg = getattr(settings, "SECURITY_GUARD", None)
    if cfg is None or not cfg.enabled:
        return
    if settings.WEBAGENT_ENV not in ("uat", "prod"):
        return
    get_engine(force_reload=True)
    if not defender_model_files_present():
        return
    import json

    from app.security.context_defender import finetuned_encoder_present, model_dir

    try:
        meta = json.loads((model_dir() / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        meta = {}
    # Missing encoder files are a broken image in any mode: the model would silently score nothing.
    if meta.get("finetuned_encoder") and not finetuned_encoder_present():
        raise RuntimeError(
            "The ContextDefender fine-tuned encoder is missing (models/security/encoder; run "
            "`git lfs pull` before building). Set WEBAGENT_GUARD_ENABLED=false or ship the encoder."
        )
    if meta.get("use_minilm") and not defender_encoder_files_present():
        raise RuntimeError(
            "The ContextDefender base encoder is missing (sentence-transformers/all-MiniLM-L6-v2 in "
            "SENTENCE_TRANSFORMERS_HOME or the Hugging Face cache). Set WEBAGENT_GUARD_ENABLED=false "
            "or bake the encoder into the image."
        )
    if cfg.judge and getattr(cfg, "jailbreak_judge", "llm") == "prompt_guard":
        from app.security import prompt_guard

        if not prompt_guard.available():
            raise RuntimeError(
                "jailbreak_judge is prompt_guard but models/security/prompt_guard has no model "
                "(scripts/fetch_security_sources.py --prompt-guard 86M, then `git lfs pull` before building)."
            )
    if not defender_gate_passed():
        if cfg.input_mode != "act":
            # Shadow mode exists to collect data with an unproven model; only acting needs the gate.
            logging.getLogger(__name__).warning(
                "ContextDefender has not passed its release gate; checkpoint 1 stays in shadow mode"
            )
            return
        raise RuntimeError(
            "The ContextDefender model has not passed the release gate "
            "(models/security/metadata.json test.passed is not true; run "
            "scripts/eval_context_defender.py --write-metadata). "
            "Set WEBAGENT_GUARD_ENABLED=false or ship a version that passes."
        )


def check_production_settings(code_execution_enabled: bool) -> None:
    """Raise RuntimeError if uat/prod settings are unsafe. Does nothing in dev."""
    check_triage_files()
    check_security_guard_files()
    check_security_guard()
    if settings.WEBAGENT_ENV not in ("uat", "prod"):
        return
    if settings.SECRET_KEY == INSECURE_SECRET_DEFAULT:
        raise RuntimeError(
            "SECRET_KEY must be set to a non-default value in uat/prod (WEBAGENT_SECURITY_SECRET_KEY)"
        )
    if settings.API_KEY_ENABLED and not os.getenv("WEBAGENT_API_KEY"):
        raise RuntimeError("WEBAGENT_API_KEY must be set when API key auth is enabled in uat/prod")
    if code_execution_enabled and sandbox_mode() != "docker":
        raise RuntimeError(
            "Code execution in uat/prod requires the container sandbox (WEBAGENT_CODE_SANDBOX=docker)"
        )


def prepare_mlflow_store() -> None:
    """Create the MLflow store and experiment once, before uvicorn starts its workers.

    On a fresh data volume every worker otherwise creates `mlruns/0/meta.yaml` at import at the same
    moment and one crashes ("Yaml file ... exists"); uvicorn restarts it, but the traceback alarms.
    Best effort: MLflow problems never block startup.
    """
    try:
        from app.core.mlflow_utils import get_or_create_experiment

        get_or_create_experiment("rag_agent_performance")
    except Exception as exc:  # noqa: BLE001
        print(f"MLflow store not prepared (workers will try): {exc}", file=sys.stderr)


if __name__ == "__main__":
    try:
        check_production_settings(server_code_execution_enabled())
    except RuntimeError as e:
        print(f"Startup check failed: {e}", file=sys.stderr)
        sys.exit(1)
    prepare_mlflow_store()
