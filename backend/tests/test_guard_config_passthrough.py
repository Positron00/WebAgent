"""Every security_guard field set in YAML reaches settings.SECURITY_GUARD (IMP-023)."""
from pathlib import Path

import pytest
import yaml

from app.core import config
from app.core.config_def import SecurityGuardConfig

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"
GUARD_ENV = ("ENABLED", "INPUT_MODE", "TOOL_MODE", "JUDGE", "COLLECT")


def _yaml_guard(env: str) -> SecurityGuardConfig:
    # Read the file directly: load_config() also exports WEBAGENT_* variables as a side effect.
    data = yaml.safe_load((CONFIG_DIR / f"{env}.yaml").read_text(encoding="utf-8"))
    return SecurityGuardConfig(**(data.get("security_guard") or {}))


@pytest.fixture(autouse=True)
def _no_guard_env(monkeypatch):
    for name in GUARD_ENV:
        monkeypatch.delenv(f"WEBAGENT_GUARD_{name}", raising=False)


@pytest.mark.parametrize("env", ["dev", "uat", "prod"])
def test_every_yaml_field_passes_through(monkeypatch, env):
    expected = _yaml_guard(env)

    class FakeAppConfig:
        security_guard = expected

    monkeypatch.setattr(config, "app_config", FakeAppConfig())
    built = config._build_security_guard_settings()

    for field in SecurityGuardConfig.model_fields:
        assert getattr(built, field) == getattr(expected, field), f"{env}: {field}"


def test_non_default_judge_fields_are_kept(monkeypatch):
    expected = SecurityGuardConfig(
        harm_judge="off", jailbreak_judge="prompt_guard", prompt_guard_threshold=0.8
    )

    class FakeAppConfig:
        security_guard = expected

    monkeypatch.setattr(config, "app_config", FakeAppConfig())
    built = config._build_security_guard_settings()

    assert (built.harm_judge, built.jailbreak_judge, built.prompt_guard_threshold) == (
        "off",
        "prompt_guard",
        0.8,
    )


def test_env_override_leaves_other_fields_alone(monkeypatch):
    expected = SecurityGuardConfig(jailbreak_judge="off", input_mode="shadow")

    class FakeAppConfig:
        security_guard = expected

    monkeypatch.setattr(config, "app_config", FakeAppConfig())
    monkeypatch.setenv("WEBAGENT_GUARD_INPUT_MODE", "act")
    built = config._build_security_guard_settings()

    assert (built.input_mode, built.jailbreak_judge) == ("act", "off")
