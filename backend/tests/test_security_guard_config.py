"""WEBAGENT_GUARD_* environment overrides on top of the YAML security_guard section."""
import pytest

from app.core import config
from app.core.config_def import SecurityGuardConfig

YAML = SecurityGuardConfig(
    enabled=True,
    input_mode="shadow",
    tool_mode="shadow",
    judge=False,
    collect=True,
)


@pytest.fixture(autouse=True)
def yaml_guard(monkeypatch):
    class FakeAppConfig:
        security_guard = YAML

    monkeypatch.setattr(config, "app_config", FakeAppConfig())
    for name in ("ENABLED", "INPUT_MODE", "TOOL_MODE", "JUDGE", "COLLECT"):
        monkeypatch.delenv(f"WEBAGENT_GUARD_{name}", raising=False)


def test_yaml_values_used_without_env():
    cfg = config._build_security_guard_settings()
    assert (cfg.enabled, cfg.input_mode, cfg.tool_mode, cfg.collect, cfg.judge) == (
        True,
        "shadow",
        "shadow",
        True,
        False,
    )


def test_each_env_override(monkeypatch):
    monkeypatch.setenv("WEBAGENT_GUARD_ENABLED", "false")
    monkeypatch.setenv("WEBAGENT_GUARD_INPUT_MODE", "act")
    monkeypatch.setenv("WEBAGENT_GUARD_TOOL_MODE", "act")
    monkeypatch.setenv("WEBAGENT_GUARD_JUDGE", "true")
    monkeypatch.setenv("WEBAGENT_GUARD_COLLECT", "false")
    cfg = config._build_security_guard_settings()
    assert (cfg.enabled, cfg.input_mode, cfg.tool_mode, cfg.judge, cfg.collect) == (
        False,
        "act",
        "act",
        True,
        False,
    )


@pytest.mark.parametrize(
    "name,value",
    [
        ("INPUT_MODE", "loud"),
        ("TOOL_MODE", "maybe"),
        ("ENABLED", "yes"),
        ("COLLECT", "maybe"),
    ],
)
def test_bad_values_warn_and_keep_yaml(monkeypatch, caplog, name, value):
    monkeypatch.setenv(f"WEBAGENT_GUARD_{name}", value)
    cfg = config._build_security_guard_settings()
    assert (cfg.enabled, cfg.input_mode, cfg.tool_mode, cfg.collect) == (True, "shadow", "shadow", True)
    assert f"WEBAGENT_GUARD_{name}" in caplog.text


def test_bad_yaml_mode_falls_back_to_shadow(monkeypatch, caplog):
    # model_construct skips validation, so the bad modes reach the builder as they would from loose YAML.
    class FakeAppConfig:
        security_guard = SecurityGuardConfig.model_construct(
            **{**YAML.model_dump(), "input_mode": "loud", "tool_mode": "nope"}
        )

    monkeypatch.setattr(config, "app_config", FakeAppConfig())
    cfg = config._build_security_guard_settings()
    assert cfg.input_mode == cfg.tool_mode == "shadow"
    assert "security_guard.input_mode" in caplog.text
