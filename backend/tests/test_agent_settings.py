"""Each agent uses its own YAML settings (IMP-030): loop limit, temperature (only when set) and caps.

The global llm.temperature (0.7 dev / 0.5 uat / 0.3 prod) is what every Together agent ran at
before; agents set 0.0-0.4, so the test checks the agent value wins only when the YAML has it.
"""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml

from app.agents import internal_research, senior_research, web_research
from app.core.config_def import AgentConfig, AgentsConfig
from app.models.task import WorkflowState
from app.services import llm as llm_module, task_budget

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"
ENVS = ("dev", "uat", "prod")
LLM_AGENTS = ("supervisor", "web_research", "internal_research", "senior_research",
              "data_analysis", "coding_assistant", "team_manager")


def _yaml(env):
    return yaml.safe_load((CONFIG_DIR / f"{env}.yaml").read_text())


def _fake_together(global_temperature, global_max_tokens):
    """Shaped like TogetherWrapper: the wrapper and its inner client both carry the settings."""
    def build(agent_name, model, provider):
        inner = SimpleNamespace(temperature=global_temperature, max_tokens=global_max_tokens)
        return SimpleNamespace(temperature=global_temperature, max_tokens=global_max_tokens, llm=inner)
    return build


class _NoRetriever:
    def __init__(self, *args, **kwargs):
        self.calls = []

    async def retrieve(self, query, k):
        self.calls.append(k)
        return []


@pytest.fixture
def env_config(request, monkeypatch):
    """Agents built from one YAML file, with a fake model instead of Together."""
    data = _yaml(request.param)
    agents = AgentsConfig(**data["agents"])
    for module in (llm_module, senior_research, web_research, internal_research):
        monkeypatch.setattr(module, "get_agents_config", lambda: agents)
    monkeypatch.setattr(llm_module, "_build_llm_for_agent",
                        _fake_together(data["llm"]["temperature"], data["llm"]["max_tokens"]))
    monkeypatch.setattr(internal_research, "RAGRetriever", _NoRetriever)
    monkeypatch.setattr(internal_research, "RAGReranker", lambda *a, **k: None)
    return data


@pytest.mark.parametrize("env_config", ENVS, indirect=True)
def test_agent_temperature_and_output_limit_follow_the_yaml(env_config):
    for name in LLM_AGENTS:
        section = env_config["agents"]["coding" if name == "coding_assistant" else name]
        model = llm_module.get_agent_llm(name).inner
        expected = section.get("temperature", env_config["llm"]["temperature"])
        assert (model.temperature, model.llm.temperature) == (expected, expected), name
        limit = min(section.get("max_tokens") or env_config["llm"]["max_tokens"], env_config["llm"]["max_tokens"])
        assert (model.max_tokens, model.llm.max_tokens) == (limit, limit), name


@pytest.mark.parametrize("env_config", ENVS, indirect=True)
def test_agent_without_temperature_keeps_the_global_one(env_config, monkeypatch):
    agents_yaml = dict(env_config["agents"])
    agents_yaml["team_manager"] = {"model": agents_yaml["team_manager"]["model"]}
    agents = AgentsConfig(**agents_yaml)
    monkeypatch.setattr(llm_module, "get_agents_config", lambda: agents)

    model = llm_module.get_agent_llm("team_manager").inner

    assert model.temperature == model.llm.temperature == env_config["llm"]["temperature"]


@pytest.mark.parametrize("env_config", ENVS, indirect=True)
def test_senior_research_loop_limit_and_prompt(env_config):
    limit = env_config["agents"]["senior_research"]["max_follow_up_questions"]
    agent = senior_research.SeniorResearchAgent()

    prompt = agent.prompt.format(query="q", requirements="r", web_research="w",
                                 internal_research="i", research_iteration=1)

    assert agent.max_iterations == limit
    assert f"(Maximum: {limit})" in prompt
    assert f"already reached {limit}," in prompt


@pytest.mark.parametrize("env_config", ENVS, indirect=True)
def test_web_and_internal_research_caps(env_config):
    agents = env_config["agents"]
    assert web_research.WebResearchAgent().max_searches == agents["web_research"]["max_searches_per_task"]
    assert internal_research.InternalResearchAgent().max_documents == agents["internal_research"]["max_documents"]


@pytest.mark.asyncio
@pytest.mark.parametrize("env_config", ["dev"], indirect=True)
async def test_web_research_pass_stops_at_the_search_cap(env_config):
    calls = []

    async def fake_search(**kwargs):
        calls.append(kwargs["query"])
        return [{"title": "t", "url": f"https://example.com/{len(calls)}", "content": "c"}]

    agent = web_research.WebResearchAgent()
    agent.llm = SimpleNamespace(ainvoke=_async_reply("## Answer"))
    state = WorkflowState(query="q", context={
        "research_plan": {"requires_web_search": True},
        "research_iteration": 2,
        "research_feedback": {"research_questions": ["first question?", "second question?"]},
    })

    # The cap is per task now (REC-109): the run's counter in task_budget holds it.
    token = task_budget.start(0, search_cap=1)
    try:
        with patch.object(web_research, "perform_web_search", fake_search):
            await agent.run(state)
    finally:
        task_budget.reset(token)

    assert calls == ["first question?"]


@pytest.mark.asyncio
@pytest.mark.parametrize("env_config", ["dev"], indirect=True)
async def test_internal_research_never_asks_for_more_than_the_cap(env_config):
    agent = internal_research.InternalResearchAgent()
    agent.max_documents = 4
    state = WorkflowState(query="q", context={
        "research_plan": {"requires_internal_knowledge": True},
        "research_iteration": 2,
        "research_feedback": {"research_questions": ["a?", "b?", "c?"]},
    })

    await agent.run(state)

    assert agent.retriever.calls == [4, 4, 4]


def test_apply_agent_settings_sets_temperature_only_when_given():
    def model():
        return SimpleNamespace(temperature=0.7, max_tokens=8000, llm=SimpleNamespace(temperature=0.7, max_tokens=8000))

    unset = llm_module.apply_agent_settings(model(), AgentConfig())
    assert (unset.temperature, unset.llm.temperature) == (0.7, 0.7)

    given = llm_module.apply_agent_settings(model(), AgentConfig(temperature=0.1, max_tokens=4096))
    assert (given.temperature, given.llm.temperature) == (0.1, 0.1)
    assert (given.max_tokens, given.llm.max_tokens) == (4096, 4096)

    assert llm_module.apply_agent_settings(model(), None).temperature == 0.7


def _async_reply(text):
    async def reply(*args, **kwargs):
        return text
    return reply
