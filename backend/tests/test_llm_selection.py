"""Tests for LLM provider selection, fallback and Together AI rate limiting (no network)."""
from types import SimpleNamespace

import pytest

from app.services import llm


@pytest.mark.parametrize(
    "provider, model, expected",
    [
        ("together", "meta-models/Muse-Glimmer-30B", "meta-models/Muse-Glimmer-30B"),
        ("together", "gpt-4o-mini", None),
        ("openai", "gpt-4o-mini", "gpt-4o-mini"),
        ("openai", "o3-mini", "o3-mini"),
        ("openai", "meta-models/Muse-Glimmer-30B", None),
        ("anthropic", "claude-3-5-sonnet", "claude-3-5-sonnet"),
        ("gemini", "gemini-1.5-flash", "gemini-1.5-flash"),
        ("groq", "llama-3.1-8b-instant", "llama-3.1-8b-instant"),
        ("groq", "meta-models/Muse-Glimmer-30B", None),
        ("together", None, None),
    ],
)
def test_model_only_goes_to_its_own_provider(provider, model, expected):
    assert llm._model_for_provider(provider, model) == expected


@pytest.fixture
def provider(monkeypatch):
    """Pick the configured provider and stop any real client from being created."""

    def use(name):
        monkeypatch.setattr(llm, "get_llm_config", lambda: SimpleNamespace(provider=name))
        monkeypatch.delenv("CONFIRM_LLM_SELECTION", raising=False)
        monkeypatch.delenv("WEBAGENT_ALLOW_FAKE_LLM", raising=False)
        for factory in (
            "get_together_llm", "get_groq_llm", "get_openai_llm",
            "get_anthropic_llm", "get_gemini_llm", "get_self_hosted_llm",
        ):
            monkeypatch.setattr(llm, factory, _failing(factory))

    return use


def _failing(name):
    def factory(model_name=None):
        raise ValueError(f"{name} unavailable")

    return factory


def test_uses_configured_provider_with_matching_model(provider, monkeypatch):
    provider("together")
    calls = []
    monkeypatch.setattr(llm, "get_together_llm", lambda model_name=None: calls.append(model_name) or "together-llm")

    assert llm._build_llm("meta-models/Muse-Glimmer-30B") == "together-llm"
    assert calls == ["meta-models/Muse-Glimmer-30B"]


def test_falls_back_to_next_provider_without_its_model(provider, monkeypatch):
    provider("together")
    calls = []
    monkeypatch.setattr(llm, "get_groq_llm", lambda model_name=None: calls.append(model_name) or "groq-llm")

    # Together fails; Groq is next and must not receive a Together-only model name
    assert llm._build_llm("meta-models/Muse-Glimmer-30B") == "groq-llm"
    assert calls == [None]


def test_all_providers_failing_raises_with_details(provider):
    provider("together")
    with pytest.raises(RuntimeError) as error:
        llm._build_llm()
    message = str(error.value)
    for name in ("together", "groq", "openai", "anthropic", "gemini", "self"):
        assert f"- {name}:" in message


def test_fake_llm_only_when_explicitly_allowed(provider, monkeypatch):
    provider("together")
    monkeypatch.setenv("WEBAGENT_ALLOW_FAKE_LLM", "true")
    fake = llm._build_llm()
    assert "unavailable" in fake.invoke("hi")


def test_agent_uses_its_yaml_model(monkeypatch):
    agents = SimpleNamespace(team_manager=SimpleNamespace(model="org/team-model"))
    monkeypatch.setattr(llm, "get_agents_config", lambda: agents)
    monkeypatch.setattr(llm, "_build_llm_for_agent", lambda agent, model=None, provider=None: f"llm:{model}")

    guarded = llm.get_agent_llm("team_manager")
    assert getattr(guarded, "inner", guarded) == "llm:org/team-model"
    unknown = llm.get_agent_llm("unknown_agent")
    assert getattr(unknown, "inner", unknown) == "llm:None"


def test_rate_limit_queues_back_to_back_calls(monkeypatch):
    monkeypatch.setattr(llm, "TOGETHER_RATE_LIMIT_BUFFER", 2.0)
    monkeypatch.setattr(llm, "_last_together_api_call", 0.0)
    monkeypatch.setattr(llm.time, "time", lambda: 1000.0)

    # Three calls at the same moment get slots 0s, 2s and 4s away
    assert [llm._reserve_together_slot() for _ in range(3)] == [0.0, 2.0, 4.0]


def test_rate_limit_does_not_wait_after_a_pause(monkeypatch):
    monkeypatch.setattr(llm, "TOGETHER_RATE_LIMIT_BUFFER", 2.0)
    monkeypatch.setattr(llm, "_last_together_api_call", 1000.0)
    monkeypatch.setattr(llm.time, "time", lambda: 1005.0)

    assert llm._reserve_together_slot() == 0.0


@pytest.mark.parametrize(
    "message, transient",
    [
        ("Together Server: Error 503", True),
        ("Together Server: Error 429", True),
        ("Read timed out", True),
        ("Connection reset by peer", True),
        ("Together Server: Error 401", False),
        ("Together Server: Error 400 bad request", False),
    ],
)
def test_transient_error_detection(message, transient):
    assert llm._is_transient_error(Exception(message)) is transient
