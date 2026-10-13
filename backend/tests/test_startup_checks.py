"""uat/prod must refuse to start with unsafe settings; dev must never be blocked."""
import subprocess
import sys
from pathlib import Path

import pytest

from app.core import startup_checks
from app.core.config import settings
from app.core.config_def import TriageConfig
from app.core.security import INSECURE_SECRET_DEFAULT

BACKEND_DIR = Path(__file__).resolve().parents[1]


@pytest.fixture
def prod(monkeypatch):
    monkeypatch.setattr(settings, "WEBAGENT_ENV", "prod")
    monkeypatch.setattr(settings, "API_KEY_ENABLED", True)
    monkeypatch.setattr(settings, "SECRET_KEY", "a-real-secret")
    monkeypatch.setenv("WEBAGENT_API_KEY", "client-key")
    monkeypatch.setenv("WEBAGENT_CODE_SANDBOX", "local")
    monkeypatch.setattr(settings, "TRIAGE", TriageConfig(enabled=False))


def test_good_prod_settings_pass(prod):
    startup_checks.check_production_settings(code_execution_enabled=False)


def test_prod_loads_security_policies(prod, monkeypatch):
    from app.core.config_def import SecurityGuardConfig

    monkeypatch.setattr(settings, "SECURITY_GUARD", SecurityGuardConfig(enabled=True))
    startup_checks.check_security_guard()


def test_default_secret_is_rejected(prod, monkeypatch):
    monkeypatch.setattr(settings, "SECRET_KEY", INSECURE_SECRET_DEFAULT)
    with pytest.raises(RuntimeError, match="WEBAGENT_SECURITY_SECRET_KEY"):
        startup_checks.check_production_settings(code_execution_enabled=False)


def test_missing_api_key_is_rejected(prod, monkeypatch):
    monkeypatch.delenv("WEBAGENT_API_KEY")
    with pytest.raises(RuntimeError, match="WEBAGENT_API_KEY"):
        startup_checks.check_production_settings(code_execution_enabled=False)


def test_code_execution_without_container_sandbox_is_rejected(prod):
    with pytest.raises(RuntimeError, match="WEBAGENT_CODE_SANDBOX=docker"):
        startup_checks.check_production_settings(code_execution_enabled=True)


def test_dev_is_never_blocked(monkeypatch):
    monkeypatch.setattr(settings, "WEBAGENT_ENV", "dev")
    monkeypatch.setattr(settings, "SECRET_KEY", INSECURE_SECRET_DEFAULT)
    startup_checks.check_production_settings(code_execution_enabled=True)


def test_prod_without_triage_files_is_rejected(prod, monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "TRIAGE", TriageConfig(enabled=True, model_dir=str(tmp_path)))
    with pytest.raises(RuntimeError, match="Triage classifier files missing"):
        startup_checks.check_production_settings(code_execution_enabled=False)


def test_prod_with_classifier_that_failed_the_gate_is_rejected(prod, monkeypatch, tmp_path):
    (tmp_path / "classifier.joblib").write_bytes(b"x")
    (tmp_path / "metadata.json").write_text('{"test": {"passed": false}}')
    monkeypatch.setattr(settings, "TRIAGE", TriageConfig(enabled=True, model_dir=str(tmp_path)))
    with pytest.raises(RuntimeError, match="release gate"):
        startup_checks.check_production_settings(code_execution_enabled=False)


def test_prod_without_encoder_is_rejected(prod, monkeypatch, tmp_path):
    (tmp_path / "classifier.joblib").write_bytes(b"x")
    (tmp_path / "metadata.json").write_text('{"test": {"passed": true}}')
    monkeypatch.setattr(settings, "TRIAGE", TriageConfig(enabled=True, model_dir=str(tmp_path)))
    monkeypatch.setattr(startup_checks, "encoder_available", lambda: False)
    with pytest.raises(RuntimeError, match="encoder"):
        startup_checks.check_production_settings(code_execution_enabled=False)


def test_prod_with_triage_disabled_needs_no_triage_files(prod, monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "TRIAGE", TriageConfig(enabled=False, model_dir=str(tmp_path)))
    startup_checks.check_triage_files()


def test_dev_without_triage_files_starts(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "WEBAGENT_ENV", "dev")
    monkeypatch.setattr(settings, "TRIAGE", TriageConfig(enabled=True, model_dir=str(tmp_path)))
    startup_checks.check_triage_files()


def test_command_line_check_exits_with_a_message():
    # The Docker image runs this before uvicorn, so a bad config stops the container.
    # Code execution without the container sandbox fails whatever .env.local provides.
    env = {
        "PATH": "/usr/bin:/bin",
        "WEBAGENT_ENV": "prod",
        "WEBAGENT_SECURITY_SECRET_KEY": "a-real-secret",
        "WEBAGENT_API_KEY": "client-key",
        "WEBAGENT_ENABLE_CODE_EXECUTION": "true",
        "WEBAGENT_CODE_SANDBOX": "local",
        "LANGSMITH_TRACING_ENABLED": "false",
        "WEBAGENT_TRIAGE_ENABLED": "false",
    }
    out = subprocess.run(
        [sys.executable, "-m", "app.core.startup_checks"],
        cwd=BACKEND_DIR, env=env, capture_output=True, text=True, timeout=120,
    )
    assert out.returncode == 1
    assert "Startup check failed" in out.stderr
    assert "WEBAGENT_CODE_SANDBOX=docker" in out.stderr


def test_encoder_available_checks_the_fine_tuned_folder(tmp_path):
    from app.services.triage_model import encoder_available

    (tmp_path / "metadata.json").write_text('{"encoder_name": "x/y", "encoder_path": "encoder"}')
    assert not encoder_available(tmp_path)
    (tmp_path / "encoder").mkdir()
    (tmp_path / "encoder" / "config.json").write_text("{}")
    (tmp_path / "encoder" / "model.safetensors").write_bytes(b"0")
    assert encoder_available(tmp_path)


def _defender_files(tmp_path, meta):
    import json

    (tmp_path / "heads.joblib").write_bytes(b"x")
    (tmp_path / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")


def _shadow_guard(monkeypatch, tmp_path):
    from app.core.config_def import SecurityGuardConfig
    from app.security import context_defender

    monkeypatch.setattr(settings, "SECURITY_GUARD", SecurityGuardConfig(enabled=True, input_mode="shadow"))
    monkeypatch.setattr(context_defender, "model_dir", lambda: tmp_path)
    monkeypatch.setattr(startup_checks, "defender_encoder_files_present", lambda *a: True)


def test_shadow_mode_starts_with_a_model_that_failed_the_gate(prod, monkeypatch, tmp_path):
    _shadow_guard(monkeypatch, tmp_path)
    _defender_files(tmp_path, {"use_minilm": True, "test": {"passed": False}})
    startup_checks.check_security_guard_files()


def test_missing_fine_tuned_security_encoder_is_rejected_even_in_shadow_mode(prod, monkeypatch, tmp_path):
    _shadow_guard(monkeypatch, tmp_path)
    _defender_files(tmp_path, {"use_minilm": True, "finetuned_encoder": True, "test": {"passed": False}})
    with pytest.raises(RuntimeError, match="fine-tuned encoder"):
        startup_checks.check_security_guard_files()
    (tmp_path / "encoder").mkdir()
    (tmp_path / "encoder" / "config.json").write_text("{}")
    (tmp_path / "encoder" / "model.safetensors").write_bytes(b"x")
    startup_checks.check_security_guard_files()


def test_prompt_guard_judge_without_the_model_is_rejected(prod, monkeypatch, tmp_path):
    from app.core.config_def import SecurityGuardConfig
    from app.security import context_defender, prompt_guard

    monkeypatch.setattr(settings, "SECURITY_GUARD", SecurityGuardConfig(
        enabled=True, input_mode="shadow", judge=True, jailbreak_judge="prompt_guard"))
    monkeypatch.setattr(context_defender, "model_dir", lambda: tmp_path)
    monkeypatch.setattr(startup_checks, "defender_encoder_files_present", lambda *a: True)
    _defender_files(tmp_path, {"use_minilm": True, "test": {"passed": False}})
    monkeypatch.setattr(prompt_guard, "MODEL_DIR", tmp_path / "prompt_guard")
    with pytest.raises(RuntimeError, match="prompt_guard"):
        startup_checks.check_security_guard_files()
    (tmp_path / "prompt_guard").mkdir()
    (tmp_path / "prompt_guard" / "config.json").write_text("{}")
    (tmp_path / "prompt_guard" / "model.safetensors").write_bytes(b"x")
    startup_checks.check_security_guard_files()


def test_mlflow_store_preparation_never_blocks_startup(monkeypatch, capsys):
    import app.core.mlflow_utils as mu

    def boom(name):
        raise RuntimeError("store unavailable")

    monkeypatch.setattr(mu, "get_or_create_experiment", boom)
    startup_checks.prepare_mlflow_store()  # must not raise
    assert "not prepared" in capsys.readouterr().err
