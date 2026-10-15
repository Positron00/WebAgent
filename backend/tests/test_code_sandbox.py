"""Model-written code must not see the server's secrets, and docker mode must be isolated."""
import os
import subprocess
import sys

import pytest

from app.utils import code_sandbox


def _script(tmp_path, body: str) -> str:
    path = tmp_path / "generated.py"
    path.write_text(body)
    return str(path)


def test_local_mode_hides_api_keys(tmp_path, monkeypatch):
    monkeypatch.setenv("WEBAGENT_CODE_SANDBOX", "local")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-not-leak")
    monkeypatch.setenv("WEBAGENT_API_KEY", "should-not-leak")
    script = _script(
        tmp_path,
        "import os\n"
        "print(os.environ.get('OPENAI_API_KEY'), os.environ.get('WEBAGENT_API_KEY'))\n",
    )

    result = code_sandbox.run_python_file(script, timeout_seconds=30)

    assert result.returncode == 0
    assert result.stdout.strip() == "None None"


def test_local_mode_enforces_timeout(tmp_path, monkeypatch):
    monkeypatch.setenv("WEBAGENT_CODE_SANDBOX", "local")
    script = _script(tmp_path, "while True:\n    pass\n")

    with pytest.raises(subprocess.TimeoutExpired):
        code_sandbox.run_python_file(script, timeout_seconds=1)


@pytest.mark.skipif(os.name != "posix", reason="CPU limits are POSIX only")
def test_cpu_limit_stop_is_reported_as_a_timeout(tmp_path, monkeypatch):
    # On Linux CI the CPU limit ended the busy loop before the wall-clock timeout, and the run
    # returned (killed by SIGXCPU) instead of raising. Both must look the same to callers.
    import signal

    monkeypatch.setenv("WEBAGENT_CODE_SANDBOX", "local")
    monkeypatch.setattr(
        code_sandbox.subprocess,
        "run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, -signal.SIGXCPU, "", ""),
    )
    with pytest.raises(subprocess.TimeoutExpired):
        code_sandbox.run_python_file(_script(tmp_path, "print('x')\n"), timeout_seconds=1)


def test_docker_mode_runs_without_network_and_with_limits(tmp_path, monkeypatch):
    monkeypatch.setenv("WEBAGENT_CODE_SANDBOX", "docker")
    calls = []
    monkeypatch.setattr(
        code_sandbox.subprocess,
        "run",
        lambda cmd, **kwargs: calls.append((cmd, kwargs)) or subprocess.CompletedProcess(cmd, 0, "ok", ""),
    )
    script = _script(tmp_path, "print('hi')\n")

    code_sandbox.run_python_file(script, timeout_seconds=5)

    cmd, kwargs = calls[0]
    joined = " ".join(cmd)
    assert cmd[:3] == ["docker", "run", "--rm"]
    assert "--network none" in joined
    assert "--read-only" in cmd
    assert "--memory" in cmd and "--pids-limit" in cmd
    assert f"{tmp_path}:/code:ro" in cmd
    assert cmd[-2:] == ["python", "/code/generated.py"]
    assert "env" not in kwargs  # nothing from the host environment is passed in


def test_unknown_mode_falls_back_to_local(monkeypatch):
    monkeypatch.setenv("WEBAGENT_CODE_SANDBOX", "chroot")
    assert code_sandbox.sandbox_mode() == "local"


def test_scrubbed_env_keeps_only_safe_names(monkeypatch):
    monkeypatch.setenv("TOGETHER_API_KEY", "secret")
    env = code_sandbox.scrubbed_env()
    assert "TOGETHER_API_KEY" not in env
    assert env["MPLBACKEND"] == "Agg"
    assert set(env) <= set(code_sandbox._SAFE_ENV_NAMES) | {"MPLBACKEND", "PYTHONIOENCODING"}
