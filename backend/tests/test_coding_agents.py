"""Tests for the Coding Assistant and Coding Review agents, with model calls replaced by fixed replies."""
import json

import pytest

from app.agents import coding_assistant as assistant_module
from app.agents import coding_review as review_module
from app.agents.coding_assistant import CodingAssistantAgent
from app.agents.coding_review import CodingReviewAgent
from app.core.config import settings
from app.models.task import WorkflowState

CODE = "import matplotlib.pyplot as plt\nplt.plot([1, 2, 3], [4, 5, 6])\nplt.title('Growth')\n"


@pytest.fixture(autouse=True)
def no_real_models(monkeypatch):
    """Build the agents without OpenAI clients or any network access."""
    monkeypatch.setattr(settings, "OPENAI_API_KEY", None)
    monkeypatch.setattr(assistant_module, "get_agent_llm", lambda _name: object())
    monkeypatch.setattr(review_module, "get_agent_llm", lambda _name: object())


def state_with(**reports):
    state = WorkflowState(query="Plot growth")
    state.reports.update(reports)
    return state


# ---------- Coding Assistant ----------

async def test_assistant_reads_code_from_json_reply(monkeypatch):
    agent = CodingAssistantAgent()
    reply = json.dumps({"explanation": "A line chart", "visualizations": ["line"], "code": CODE})

    async def fake_generate(prompt):
        return reply

    monkeypatch.setattr(agent, "_generate_with_retry", fake_generate)
    state = await agent.run(state_with())

    report = state.reports["coding_assistant"]
    assert report["code"] == CODE
    assert report["explanation"] == "A line chart"
    assert state.error is None


async def test_assistant_extracts_code_block_from_markdown_reply(monkeypatch):
    agent = CodingAssistantAgent()

    async def fake_generate(prompt):
        return f"Here is the chart:\n```python\n{CODE}```\nEnjoy."

    monkeypatch.setattr(agent, "_generate_with_retry", fake_generate)
    state = await agent.run(state_with())

    assert state.reports["coding_assistant"]["code"].strip() == CODE.strip()


async def test_assistant_tries_next_method_after_a_failure(monkeypatch):
    agent = CodingAssistantAgent()
    calls = []

    async def broken(prompt):
        calls.append("broken")
        raise RuntimeError("model unavailable")

    async def working(prompt):
        calls.append("working")
        return "ok"

    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr("asyncio.sleep", no_sleep)
    monkeypatch.setattr(agent, "_try_langchain_agenerate", broken)
    monkeypatch.setattr(agent, "_try_langchain_invoke", working)

    assert await agent._generate_with_retry("prompt") == "ok"
    assert calls == ["broken", "working"]


async def test_assistant_returns_error_json_when_every_method_fails(monkeypatch):
    agent = CodingAssistantAgent()

    async def broken(prompt):
        raise RuntimeError("down")

    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr("asyncio.sleep", no_sleep)
    for name in ("_try_langchain_agenerate", "_try_langchain_invoke", "_try_fallback"):
        monkeypatch.setattr(agent, name, broken)

    reply = json.loads(await agent._generate_with_retry("prompt"))
    assert reply["explanation"] == "Error generating code"


# ---------- Coding Review: running code ----------

def test_code_does_not_run_when_execution_is_disabled(monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_CODE_EXECUTION", False)
    success, output = CodingReviewAgent().execute_python_code("print('hi')")
    assert success is False
    assert "disabled" in output


def test_code_runs_and_returns_output(monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_CODE_EXECUTION", True)
    success, output = CodingReviewAgent().execute_python_code("print('hello')")
    assert success is True
    assert output == "hello"


def test_failing_code_reports_the_error(monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_CODE_EXECUTION", True)
    success, output = CodingReviewAgent().execute_python_code("raise ValueError('bad input')")
    assert success is False
    assert "ValueError" in output and "bad input" in output


def test_slow_code_times_out(monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_CODE_EXECUTION", True)
    agent = CodingReviewAgent()
    agent.timeout_seconds = 1
    success, output = agent.execute_python_code("import time\ntime.sleep(5)")
    assert success is False
    assert "timed out" in output


# ---------- Coding Review: reading the review ----------

REVIEW_REPLY = """REVIEW:
The plot is missing axis labels.

EXECUTION ANALYSIS:
The code raised a NameError.

```python
print('short')
```

```python
import math
print(math.sqrt(16))
```

IMPROVEMENT EXPLANATION:
Imported math before using it.
"""


def test_review_sections_and_longest_code_block_are_extracted():
    agent = CodingReviewAgent()
    assert agent.extract_section(REVIEW_REPLY, "REVIEW") == "The plot is missing axis labels."
    assert agent.extract_section(REVIEW_REPLY, "EXECUTION ANALYSIS") == "The code raised a NameError."
    assert agent.extract_code(REVIEW_REPLY) == "import math\nprint(math.sqrt(16))"


async def test_review_runs_the_fixed_code_when_the_original_failed(monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_CODE_EXECUTION", True)
    agent = CodingReviewAgent()

    async def fake_review(prompt_dict):
        assert "NameError" in prompt_dict["execution_result"]
        return REVIEW_REPLY

    monkeypatch.setattr(agent, "_generate_review", fake_review)
    state = await agent.run(state_with(coding_assistant={"code": "print(math.sqrt(16))"}))

    report = state.reports["coding_review"]
    assert report["original_execution"]["success"] is False
    assert report["fixed_execution"] == {"success": True, "result": "4.0"}
    assert report["improvement_explanation"] == "Imported math before using it."
