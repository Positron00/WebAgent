from unittest.mock import patch

from fastapi.testclient import TestClient

from main import app


class CaptureWorkflow:
    def __init__(self):
        self.last_state = None

    async def ainvoke(self, state):
        self.last_state = state
        return {"final_report": {"title": "R", "content": "ok"}}


def test_previous_messages_reach_workflow_context():
    workflow = CaptureWorkflow()
    with patch("app.services.task_manager.get_agent_workflow", return_value=workflow):
        client = TestClient(app)
        r = client.post(
            "/api/task",
            json={
                "query": "what about Europe?",
                "agentic": True,
                "previous_messages": [
                    {"role": "user", "content": "Compare US healthcare"},
                    {"role": "assistant", "content": "US summary"},
                ],
            },
        )
        assert r.status_code == 200
        assert workflow.last_state is not None
        prev = workflow.last_state.context["previous_messages"]
        assert prev[0]["content"] == "Compare US healthcare"


def test_too_many_previous_messages_rejected():
    client = TestClient(app)
    body = {
        "query": "x",
        "previous_messages": [{"role": "user", "content": f"m{i}"} for i in range(9)],
    }
    r = client.post("/api/task", json=body)
    assert r.status_code == 422
