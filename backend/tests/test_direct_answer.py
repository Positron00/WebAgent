"""The backend direct path: a prompt the text model answers, and no reasoning in the reply."""
import pytest

from app.agents.supervisor import SupervisorAgent
from app.utils.reasoning import strip_reasoning

# Shape of a real Together reply to the direct path before the fix (3.1.0 smoke test).
THINKING_REPLY = (
    " To the nearest thousand.\nassistant to=selfAnd what is its population?\n\n"
    "We need population of Canberra. Probably 463,000.\n\nFinal.assistant to=user"
    "Canberra's population was about 464,000 in mid-2023."
)


class _FakeLLM:
    def __init__(self, reply):
        self.reply = reply
        self.prompts = []

    async def ainvoke(self, prompt, *args, **kwargs):
        self.prompts.append(prompt)
        return self.reply


def _agent(reply):
    agent = SupervisorAgent.__new__(SupervisorAgent)
    agent.llm = _FakeLLM(reply)
    return agent


@pytest.mark.asyncio
async def test_direct_answer_drops_reasoning_and_keeps_the_answer():
    agent = _agent(THINKING_REPLY)
    answer = await agent.answer_directly(
        "And what is its population?",
        [{"role": "user", "content": "What is the capital of Australia?"},
         {"role": "assistant", "content": "Canberra."}],
    )
    assert answer == "Canberra's population was about 464,000 in mid-2023."


@pytest.mark.asyncio
async def test_direct_prompt_ends_where_the_answer_starts_and_carries_history():
    agent = _agent("Paris.")
    await agent.answer_directly("And France?", [{"role": "user", "content": "Capital of Italy?"}])
    prompt = agent.llm.prompts[0]
    assert prompt.rstrip().endswith("Answer:")
    assert "user: Capital of Italy?" in prompt
    assert "Current query: And France?" in prompt


def test_strip_reasoning_leaves_plain_answers_alone():
    assert strip_reasoning("  Plain answer.\n") == "Plain answer."
    assert strip_reasoning("thinking...assistant to=user") == "thinking...assistant to=user"


@pytest.mark.asyncio
async def test_direct_prompt_says_the_image_cannot_be_seen():
    agent = _agent("I can't view images here.")
    await agent.answer_directly("describe this picture", [], image_attached=True)
    assert "cannot see it" in agent.llm.prompts[0]
    await agent.answer_directly("describe this picture", [])
    assert "cannot see it" not in agent.llm.prompts[1]
