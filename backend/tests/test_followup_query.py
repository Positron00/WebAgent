"""Follow-up questions are searched as the supervisor's standalone rewrite, not as raw text."""
from langchain.prompts import ChatPromptTemplate

from app.agents.supervisor import SUPERVISOR_PROMPT
from app.models.research_plan import normalize_research_plan, research_query
from app.models.task import WorkflowState


def test_research_uses_standalone_rewrite_of_a_follow_up():
    state = WorkflowState(
        query="what about Europe?",
        context={
            "research_plan": {
                "requires_web_search": True,
                "standalone_query": "Electric vehicle adoption rates in Europe",
            }
        },
    )
    assert research_query(state) == "Electric vehicle adoption rates in Europe"


def test_research_falls_back_to_the_raw_query():
    state = WorkflowState(query="EV adoption in Norway", context={"research_plan": {}})
    assert research_query(state) == "EV adoption in Norway"


def test_blank_rewrite_is_ignored():
    plan = normalize_research_plan({"standalone_query": "   "})
    assert plan.standalone_query == ""


def test_supervisor_prompt_asks_for_the_rewrite_and_still_takes_only_the_query():
    assert '"standalone_query"' in SUPERVISOR_PROMPT
    assert ChatPromptTemplate.from_template(SUPERVISOR_PROMPT).input_variables == ["query"]
