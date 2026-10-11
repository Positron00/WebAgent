"""WEBAGENT_TRIAGE_* environment overrides on top of the YAML `triage:` section (task 2.1)."""
import pytest

from app.core import config
from app.core.config_def import TriageConfig

YAML = TriageConfig(enabled=True, mode="act", collect=True, llm_tiebreak=False)


@pytest.fixture(autouse=True)
def yaml_triage(monkeypatch):
    class FakeAppConfig:
        triage = YAML

    monkeypatch.setattr(config, "app_config", FakeAppConfig())
    for name in ("ENABLED", "MODE", "THRESHOLD", "LLM_TIEBREAK", "COLLECT"):
        monkeypatch.delenv(f"WEBAGENT_TRIAGE_{name}", raising=False)


def test_yaml_values_used_without_env():
    cfg = config._build_triage_settings()
    assert (cfg.enabled, cfg.mode, cfg.collect, cfg.threshold) == (True, "act", True, None)


def test_each_env_override(monkeypatch):
    monkeypatch.setenv("WEBAGENT_TRIAGE_ENABLED", "FALSE")
    monkeypatch.setenv("WEBAGENT_TRIAGE_MODE", "shadow")
    monkeypatch.setenv("WEBAGENT_TRIAGE_THRESHOLD", "0.42")
    monkeypatch.setenv("WEBAGENT_TRIAGE_LLM_TIEBREAK", "true")
    monkeypatch.setenv("WEBAGENT_TRIAGE_COLLECT", "false")
    cfg = config._build_triage_settings()
    assert (cfg.enabled, cfg.mode, cfg.threshold, cfg.llm_tiebreak, cfg.collect) == (
        False, "shadow", 0.42, True, False,
    )


@pytest.mark.parametrize(
    "name,value",
    [("MODE", "loud"), ("THRESHOLD", "abc"), ("THRESHOLD", "1.5"), ("ENABLED", "yes"), ("COLLECT", "maybe")],
)
def test_bad_values_are_ignored_with_a_warning(monkeypatch, caplog, name, value):
    monkeypatch.setenv(f"WEBAGENT_TRIAGE_{name}", value)
    cfg = config._build_triage_settings()
    assert (cfg.enabled, cfg.mode, cfg.threshold, cfg.collect) == (True, "act", None, True)
    assert f"WEBAGENT_TRIAGE_{name}" in caplog.text


def test_bad_yaml_mode_falls_back_to_act(monkeypatch, caplog):
    class FakeAppConfig:
        triage = TriageConfig(mode="loud")

    monkeypatch.setattr(config, "app_config", FakeAppConfig())
    assert config._build_triage_settings().mode == "act"
    assert "triage.mode" in caplog.text
