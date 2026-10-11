"""Run model-written Python with as little access to the host as the mode allows.

Modes (WEBAGENT_CODE_SANDBOX):
- "local" (default): a child process with no API keys or other secrets in its environment,
  and CPU, memory and file-size limits. Fine for local demos; not isolation.
- "docker": a throwaway container with no network, a read-only filesystem and resource
  limits. Required for code execution in uat/prod. Build the image from backend/sandbox/.
"""
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

SANDBOX_MODES = ("local", "docker")

# Environment variables the child process keeps; everything else (API keys, tokens) is dropped.
_SAFE_ENV_NAMES = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "SYSTEMROOT")


def sandbox_mode() -> str:
    mode = os.getenv("WEBAGENT_CODE_SANDBOX", "local").lower()
    return mode if mode in SANDBOX_MODES else "local"


def scrubbed_env() -> Dict[str, str]:
    env = {name: os.environ[name] for name in _SAFE_ENV_NAMES if name in os.environ}
    env["MPLBACKEND"] = "Agg"  # no display
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _limit_resources(cpu_seconds: int) -> None:
    """Runs in the child before exec (POSIX only). Unsupported limits are skipped."""
    import resource

    limits = [
        (resource.RLIMIT_CPU, cpu_seconds),
        (resource.RLIMIT_FSIZE, 100 * 1024 * 1024),  # 100 MB per file
    ]
    if sys.platform.startswith("linux"):
        limits.append((resource.RLIMIT_AS, 2 * 1024 * 1024 * 1024))  # 2 GB address space
    for limit, value in limits:
        try:
            resource.setrlimit(limit, (value, value))
        except (ValueError, OSError):
            pass


def docker_command(script_path: str) -> List[str]:
    script = Path(script_path)
    image = os.getenv("WEBAGENT_SANDBOX_IMAGE", "webagent/code-sandbox:latest")
    return [
        "docker", "run", "--rm",
        "--network", "none",
        "--memory", os.getenv("WEBAGENT_SANDBOX_MEMORY", "512m"),
        "--cpus", "1",
        "--pids-limit", "128",
        "--read-only",
        "--tmpfs", "/tmp",
        "--security-opt", "no-new-privileges",
        "-e", "MPLBACKEND=Agg",
        "-e", "MPLCONFIGDIR=/tmp/matplotlib",  # the home folder is read-only
        "-v", f"{script.parent}:/code:ro",
        "-w", "/tmp",
        image,
        "python", f"/code/{script.name}",
    ]


def run_python_file(script_path: str, timeout_seconds: int) -> subprocess.CompletedProcess:
    """Run a Python file in the configured sandbox. Raises subprocess.TimeoutExpired on timeout."""
    if sandbox_mode() == "docker":
        return subprocess.run(
            docker_command(script_path),
            capture_output=True,
            text=True,
            timeout=timeout_seconds + 15,  # container start-up
        )

    preexec = None
    if os.name == "posix":
        preexec = lambda: _limit_resources(timeout_seconds)  # noqa: E731
    return subprocess.run(
        [sys.executable, script_path],
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        env=scrubbed_env(),
        preexec_fn=preexec,
    )
