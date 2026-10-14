"""Progress of a running workflow, saved in its result file (IMP-052).

`save` is called once per finished graph node. Any worker answering a status poll reads
the saved `progress` from the file.
"""
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.services import result_store

logger = logging.getLogger(__name__)

MAX_LOG_LINES = 5

STEP_LABELS = {
    "supervisor": "Planning the research",
    "research": "Web and internal research",
    "senior_research": "Checking the findings",
    "team_manager": "Writing the report",
    "data_analysis": "Analysing data",
    "coding_assistant": "Writing code",
    "coding_review": "Reviewing code",
    "document_extraction": "Reading the document",
}


def step_label(node: Optional[str]) -> str:
    if not node:
        return "Working"
    return STEP_LABELS.get(node, node.replace("_", " "))


def logs_of(state: Any) -> List[str]:
    """The workflow log lines in a state or node update (a WorkflowState or a dict)."""
    logs = state.get("logs") if isinstance(state, dict) else getattr(state, "logs", None)
    return [str(line) for line in logs] if isinstance(logs, list) else []


def build(node: Optional[str], logs: List[str]) -> Dict[str, Any]:
    return {
        "step": step_label(node),
        "logs": list(logs[-MAX_LOG_LINES:]),
        "updated_at": datetime.now().isoformat(),
    }


def save(task_id: str, node: Optional[str], logs: List[str]) -> bool:
    """Save progress for a processing task. A failure is logged and never stops the run."""
    try:
        return result_store.save_progress(task_id, build(node, logs))
    except Exception as e:
        logger.warning(f"Could not save progress for task {task_id}: {e}")
        return False
