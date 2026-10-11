"""The web research prompt must be fully filled in and end with a clear cue to write the answer.

A real run in 3.0.2 sent the template with literal "{query}"/"{search_results}" placeholders and
ended on raw search results, so the completion model kept writing (inventing sources) until it hit
16,384 tokens, about three minutes per call.
"""
import re
import zipfile
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from app.agents import web_research as web_research_module
from app.models.task import WorkflowState
from app.services.llm import apply_max_tokens

SEARCH_RESULTS = [
    {"title": "LangGraph docs", "url": "https://example.com/langgraph", "content": "LangGraph uses state graphs."},
    {"title": "CrewAI docs", "url": "https://example.com/crewai", "content": "CrewAI uses role-based crews."},
]
UNFILLED = re.compile(r"\{[a-z_]+\}")


class _RecordingLLM:
    def __init__(self, reply="## Comparison\nLangGraph uses graphs [1]; CrewAI uses crews [2]."):
        self.prompts = []
        self.reply = reply

    async def ainvoke(self, prompt, *args, **kwargs):
        self.prompts.append(prompt)
        return self.reply


async def _fake_search(**kwargs):
    return list(SEARCH_RESULTS)


def _agent(llm):
    with patch.object(web_research_module, "get_agent_llm", return_value=llm):
        return web_research_module.WebResearchAgent()


@pytest.mark.asyncio
async def test_first_pass_prompt_is_filled_and_ends_with_the_answer_cue():
    llm = _RecordingLLM()
    agent = _agent(llm)
    state = WorkflowState(
        query="LangGraph vs CrewAI?",
        context={"research_plan": {"requires_web_search": True, "questions": ["How do they differ?"]}},
    )

    with patch.object(web_research_module, "perform_web_search", _fake_search):
        state = await agent.run(state)

    prompt = llm.prompts[0]
    assert isinstance(prompt, str)
    assert not UNFILLED.search(prompt), UNFILLED.findall(prompt)
    assert "LangGraph vs CrewAI?" in prompt
    assert "- How do they differ?" in prompt
    assert "SOURCE 1:" in prompt and "SOURCE 2:" in prompt
    assert prompt.rstrip().endswith("ANALYSIS:")
    assert state.reports["web_research"]["analysis"].startswith("## Comparison")


@pytest.mark.asyncio
async def test_follow_up_prompt_is_filled_too():
    llm = _RecordingLLM()
    agent = _agent(llm)
    state = WorkflowState(
        query="LangGraph vs CrewAI?",
        context={
            "research_plan": {"requires_web_search": True},
            "research_iteration": 2,
            "research_feedback": {
                "missing_information": ["pricing"],
                "research_questions": ["What do they cost in 2026?"],
            },
        },
    )

    with patch.object(web_research_module, "perform_web_search", _fake_search):
        await agent.run(state)

    prompt = llm.prompts[0]
    assert not UNFILLED.search(prompt), UNFILLED.findall(prompt)
    assert "ADDITIONAL RESEARCH REQUESTED - ITERATION 2" in prompt
    assert "- What do they cost in 2026?" in prompt
    assert prompt.rstrip().endswith("ANALYSIS:")


def test_reasoning_before_the_answer_marker_is_dropped():
    agent = _agent(_RecordingLLM())
    raw = (
        "We must ensure citations correspond. Let's craft final answer.\n\n"
        "assistant to=user## LangGraph vs CrewAI\n\nLangGraph uses graphs [1]."
    )

    cleaned = agent._clean_analysis(raw)

    assert cleaned.startswith("## LangGraph vs CrewAI")
    assert "craft final answer" not in cleaned


def test_agent_output_limit_lowers_but_never_raises():
    inner = SimpleNamespace(max_tokens=16384)
    wrapper = SimpleNamespace(max_tokens=16384, llm=inner)
    apply_max_tokens(wrapper, 3000)
    assert (wrapper.max_tokens, inner.max_tokens) == (3000, 3000)

    small = SimpleNamespace(max_tokens=1000)
    apply_max_tokens(small, 3000)
    assert small.max_tokens == 1000


def test_web_research_limit_is_configured_in_every_environment():
    import yaml
    from pathlib import Path

    config_dir = Path(__file__).resolve().parents[1] / "config"
    for env in ("dev", "uat", "prod"):
        agents = yaml.safe_load((config_dir / f"{env}.yaml").read_text())["agents"]
        assert agents["web_research"]["max_tokens"] == 4096, env


def test_corpus_installed_only_as_zip_counts_as_installed(tmp_path, monkeypatch):
    import nltk
    from app.utils import nltk_data

    corpora = tmp_path / "corpora"
    corpora.mkdir()
    with zipfile.ZipFile(corpora / "fakecorpus.zip", "w") as zf:
        zf.writestr("fakecorpus/README", "data")
    monkeypatch.setattr(nltk.data, "path", [str(tmp_path)])

    assert nltk_data._is_installed("corpora/fakecorpus")
    assert not nltk_data._is_installed("corpora/missingcorpus")
