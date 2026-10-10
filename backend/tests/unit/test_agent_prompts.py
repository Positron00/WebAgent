"""Ensure agent prompt templates only declare expected input variables."""
from langchain.prompts import ChatPromptTemplate

from app.agents.supervisor import SUPERVISOR_PROMPT
from app.agents.team_manager import TEAM_MANAGER_PROMPT
from app.agents.data_analysis import DATA_ANALYSIS_PROMPT


def test_supervisor_prompt_variables():
    variables = ChatPromptTemplate.from_template(SUPERVISOR_PROMPT).input_variables
    assert variables == ["query"]


def test_team_manager_prompt_variables():
    variables = ChatPromptTemplate.from_template(TEAM_MANAGER_PROMPT).input_variables
    assert set(variables) == {
        "query",
        "research_synthesis",
        "data_analysis",
        "visualization_code",
    }


def test_data_analysis_prompt_variables():
    variables = ChatPromptTemplate.from_template(DATA_ANALYSIS_PROMPT).input_variables
    assert set(variables) == {"query", "research_findings"}
