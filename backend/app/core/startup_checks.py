"""Settings that must be right before the server accepts traffic in uat/prod.

Run as `python -m app.core.startup_checks` before starting uvicorn (the Docker image does this):
with several workers, uvicorn restarts a worker that fails at startup, so a failure inside the
app alone would loop forever instead of stopping the container.
"""
import os
import sys

from app.core.config import settings
from app.core.security import INSECURE_SECRET_DEFAULT
from app.utils.code_sandbox import sandbox_mode


def server_code_execution_enabled() -> bool:
    """The API server runs generated code only when explicitly enabled."""
    return os.getenv("WEBAGENT_ENABLE_CODE_EXECUTION", "false").lower() == "true"


def check_production_settings(code_execution_enabled: bool) -> None:
    """Raise RuntimeError if uat/prod settings are unsafe. Does nothing in dev."""
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
