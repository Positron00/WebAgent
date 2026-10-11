"""API test: a task posted to /api/task runs in the background and leaves "processing"."""
from unittest.mock import patch

from fastapi.testclient import TestClient

from main import app


class FakeWorkflow:
    """Stands in for the compiled LangGraph; LangGraph 0.2 returns a dict of state fields."""

    async def ainvoke(self, state):
        return {
            "query": state.query,
            "context": state.context,
            "final_report": {"title": "Report", "content": "Scripted answer"},
        }


class EmptyWorkflow:
    async def ainvoke(self, state):
        return {"query": state.query, "context": state.context, "error": "Supervisor failed"}


def _poll(client: TestClient, task_id: str) -> dict:
    for _ in range(20):
        body = client.get(f"/api/task/{task_id}").json()
        if body["status"] != "processing":
            return body
    return body


def test_task_completes_with_report():
    with patch("app.services.task_manager.get_agent_workflow", return_value=FakeWorkflow()):
        client = TestClient(app)
        created = client.post("/api/task", json={"query": "What is an example?", "agentic": True})
        assert created.status_code == 200
        task_id = created.json()["task_id"]

        body = _poll(client, task_id)

    assert body["status"] == "completed"
    assert body["result"]["content"] == "Scripted answer"


def test_task_without_report_is_reported_as_failed():
    with patch("app.services.task_manager.get_agent_workflow", return_value=EmptyWorkflow()):
        client = TestClient(app)
        task_id = client.post("/api/task", json={"query": "q", "agentic": True}).json()["task_id"]

        body = _poll(client, task_id)

    assert body["status"] == "failed"
    assert body["error"] == "Supervisor failed"
