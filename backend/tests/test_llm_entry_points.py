"""SEC-001 / SEC-002: LLM entry points stay in services/llm.py."""
from pathlib import Path

import pytest

AGENTS_DIR = Path(__file__).resolve().parents[1] / "app" / "agents"
ALLOWED_TRUSTED = {
    Path(__file__).resolve().parents[1] / "app" / "services" / "triage.py",
    Path(__file__).resolve().parents[1] / "app" / "security" / "judge.py",
    Path(__file__).resolve().parents[1] / "app" / "utils" / "diagnostics.py",
}


def test_agents_do_not_import_direct_model_clients():
    banned = ("ChatOpenAI", "get_llm_without_tags", "openai.OpenAI", "from openai import")
    for path in AGENTS_DIR.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        for token in banned:
            assert token not in text, f"{path.name} imports forbidden {token}"


def test_agents_make_no_raw_model_api_calls():
    """No agent posts to a model provider itself: every call goes through services/llm.py (SEC-001)."""
    providers = ("api.together.xyz", "api.openai.com", "api.anthropic.com", "api.groq.com",
                 "generativelanguage.googleapis.com", "/v1/completions", "/v1/chat/completions")
    for path in AGENTS_DIR.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        for token in providers:
            assert token not in text, f"{path.name} calls a model API directly ({token})"


def test_coding_assistant_uses_only_the_guarded_model():
    from app.agents.coding_assistant import CodingAssistantAgent

    assert not hasattr(CodingAssistantAgent, "_try_direct_api")


def test_get_trusted_llm_only_in_allowed_modules():
    root = Path(__file__).resolve().parents[1] / "app"
    for path in root.rglob("*.py"):
        if "test_" in path.name:
            continue
        text = path.read_text(encoding="utf-8")
        if "get_trusted_llm" not in text:
            continue
        assert path.resolve() in ALLOWED_TRUSTED or path.name == "llm.py", (
            f"get_trusted_llm referenced from unexpected file: {path}"
        )


def test_coding_agents_use_their_configured_model_not_an_implicit_openai_switch(monkeypatch):
    from app.core.config import settings
    from app.services import llm

    monkeypatch.setattr(settings, "OPENAI_API_KEY", "sk-placeholder")
    called = {}
    monkeypatch.setattr(llm, "get_openai_llm", lambda model=None: called.setdefault("openai", model))
    monkeypatch.setattr(llm, "_build_llm", lambda model=None, call_label="other": called.setdefault("built", (model, call_label)))
    llm._build_llm_for_agent("coding_assistant", "meta-models/Muse-Glimmer-30B", None)
    assert called == {"built": ("meta-models/Muse-Glimmer-30B", "coding_assistant")}
    called.clear()
    llm._build_llm_for_agent("coding_assistant", None, "openai")
    assert called == {"openai": settings.OPENAI_MODEL}
