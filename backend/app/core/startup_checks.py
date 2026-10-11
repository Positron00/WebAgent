"""Settings that must be right before the server accepts traffic in uat/prod.

Run as `python -m app.core.startup_checks` before starting uvicorn (the Docker image does this):
with several workers, uvicorn restarts a worker that fails at startup, so a failure inside the
app alone would loop forever instead of stopping the container.
"""
import os
import sys

from app.core.config import settings
from app.core.security import INSECURE_SECRET_DEFAULT
from app.services.triage_model import encoder_files_present, gate_passed, model_files_present
from app.utils.code_sandbox import sandbox_mode


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
    if not encoder_files_present():
        raise RuntimeError(
            f"Triage encoder {cfg.encoder_name} is not downloaded (looked in SENTENCE_TRANSFORMERS_HOME "
            "and the Hugging Face cache). Set WEBAGENT_TRIAGE_ENABLED=false or bake it into the image."
        )


def check_production_settings(code_execution_enabled: bool) -> None:
    """Raise RuntimeError if uat/prod settings are unsafe. Does nothing in dev."""
    check_triage_files()
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


if __name__ == "__main__":
    try:
        check_production_settings(server_code_execution_enabled())
    except RuntimeError as e:
        print(f"Startup check failed: {e}", file=sys.stderr)
        sys.exit(1)
