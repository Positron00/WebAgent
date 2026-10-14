"""User question text must not appear in logs outside dev (IMP-070)."""
import ast
import logging
from pathlib import Path
from uuid import uuid4

import pytest

from app.core.config import settings
from app.services import task_manager as task_manager_module
from app.services.task_manager import TaskManager
from tests.fakes.workflow import StreamingFakeWorkflow

MARKER = "marker-question-7f3a9b2c"


class _CollectHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


# Variables that hold the user's question or text derived from it (follow-up questions,
# rewritten search queries). The 3.5.0 review found `{question}` logged by internal research.
RAW_TEXT_NAMES = (
    "query", "search_query", "question", "user_query", "sanitized_query", "simplified_query",
    "enhanced_query", "standalone_query", "transcript",
)


def _forbidden_query_log_snippets() -> list[str]:
    """Static scan: logger calls that still embed state.query or raw query variables."""
    app_dir = Path(__file__).resolve().parents[1] / "app"
    bad: list[str] = []
    for path in sorted(app_dir.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (
                isinstance(func, ast.Attribute)
                and func.attr in ("info", "warning", "error", "exception")
                and isinstance(func.value, ast.Name)
                and func.value.id == "logger"
            ):
                continue
            for arg in (*node.args, *(kw.value for kw in node.keywords)):
                if isinstance(arg, ast.JoinedStr):
                    for part in arg.values:
                        if isinstance(part, ast.FormattedValue):
                            src = ast.get_source_segment(source, part.value) or ""
                            if "describe_text" in src:
                                continue
                            if src.endswith(".query") or src.strip() in RAW_TEXT_NAMES:
                                bad.append(f"{path.relative_to(app_dir.parent)}:{node.lineno}")
    return bad


def test_no_logger_lines_embed_raw_query_text():
    assert _forbidden_query_log_snippets() == []


@pytest.mark.asyncio
async def test_a_full_run_never_logs_the_question_in_prod(monkeypatch, caplog):
    monkeypatch.setattr(settings, "WEBAGENT_ENV", "prod")

    class FakeWorkflow(StreamingFakeWorkflow):
        async def ainvoke(self, state):
            logging.getLogger("test.marker").info("agent step")
            state.mark_completed({"content": "done"})
            return state

    monkeypatch.setattr(task_manager_module, "get_agent_workflow", lambda: FakeWorkflow())
    task_id = str(uuid4())

    with caplog.at_level(logging.INFO):
        await TaskManager().run_workflow(task_id, MARKER, {})

    combined = "\n".join(caplog.messages)
    assert MARKER not in combined
