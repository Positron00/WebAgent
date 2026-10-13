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
