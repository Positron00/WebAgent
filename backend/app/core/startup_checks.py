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


LOOPBACK = {"127.0.0.1", "::1", "localhost"}


def check_dev_exposure(host: str) -> None:
    """In dev with API keys off, refuse a host other machines can reach."""
    if settings.WEBAGENT_ENV != "dev" or settings.API_KEY_ENABLED or settings.ALLOW_OPEN_DEV:
        return
    if host not in LOOPBACK:
        raise SystemExit(
            f"Dev backend would listen on {host} with API keys off. "
            "Set api.host to 127.0.0.1, turn on API keys, or set WEBAGENT_ALLOW_OPEN_DEV=true."
        )


def check_task_timeouts() -> None:
    """Warn when the stale-record timeout could mark a run that is still allowed to run as interrupted."""
    # Same parsing as task_manager.TASK_TIMEOUT_MINUTES, which this module must not import (heavy).
    try:
        stale_minutes = int(os.getenv("WEBAGENT_TASK_TIMEOUT_MINUTES", "30"))
    except ValueError:
        return
    if stale_minutes * 60 <= settings.TASK_MAX_SECONDS + 60:
        logging.getLogger(__name__).warning(
            "WEBAGENT_TASK_TIMEOUT_MINUTES (%s min) should be more than WEBAGENT_TASK_MAX_SECONDS "
            "(%s s) plus a minute; running tasks may be reported as interrupted",
            stale_minutes,
            settings.TASK_MAX_SECONDS,
        )


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


def check_alignment_model_gate() -> None:
    """checkpoint2-intent SEC-126: an alignment classifier that ships must have passed its gate.

    A missing folder is fine (phase A signals are used). Called by check_security_guard_files, which
    has already checked uat/prod and that the guard is enabled."""
    from app.security import alignment_model

    d = alignment_model.model_dir()
    if not (d / "metadata.json").is_file():
        return
    if not alignment_model.gate_passed(alignment_model.read_metadata(d)):
        raise RuntimeError(
            f"The alignment classifier in {d} has not passed the release gate (metadata.json "
            "test.passed is not true; run scripts/eval_alignment_model.py --write-metadata). "
            "Remove it, set security_guard.alignment_model_dir to a gated model, or set "
            "WEBAGENT_GUARD_ENABLED=false."
        )


def check_security_guard_files() -> None:
    """In uat/prod with the guard enabled, policies are required; model gate when artifacts exist."""
    cfg = getattr(settings, "SECURITY_GUARD", None)
    if cfg is None or not cfg.enabled:
        return
    if settings.WEBAGENT_ENV not in ("uat", "prod"):
        return
    get_engine(force_reload=True)
    check_alignment_model_gate()
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
