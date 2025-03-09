"""
Agent modules for the WebAgent backend.
"""
from app.agents.supervisor import get_supervisor_agent
from app.agents.web_research import get_web_research_agent
from app.agents.internal_research import get_internal_research_agent
from app.agents.senior_research import get_senior_research_agent

__all__ = [
    "get_supervisor_agent",
    "get_web_research_agent",
    "get_internal_research_agent",
    "get_senior_research_agent"
]
