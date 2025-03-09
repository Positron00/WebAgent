"""
Agent modules for the WebAgent backend.
"""
from app.agents.supervisor import get_supervisor_agent
from app.agents.web_research import get_web_research_agent
from app.agents.internal_research import get_internal_research_agent
from app.agents.senior_research import get_senior_research_agent
from app.agents.data_analysis import get_data_analysis_agent
from app.agents.coding_assistant import get_coding_assistant_agent
from app.agents.team_manager import get_team_manager_agent

__all__ = [
    "get_supervisor_agent",
    "get_web_research_agent",
    "get_internal_research_agent",
    "get_senior_research_agent",
    "get_data_analysis_agent",
    "get_coding_assistant_agent",
    "get_team_manager_agent"
]
