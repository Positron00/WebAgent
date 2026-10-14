"""Progress of a running workflow is saved per node and readable from any worker (IMP-052)."""
from uuid import uuid4

import pytest
from langgraph.graph import END, StateGraph

from app.api.endpoints import task as task_endpoint
from app.models.task import WorkflowState
from app.services import progress, result_store, task_manager as task_manager_module
from tests.fakes.workers import separate_task_manager

NODES = ["supervisor", "research", "team_manager"]


def _three_node_graph(on_node=None):
    """A real LangGraph graph with three fake nodes, each adding one workflow log line."""
    graph = StateGraph(WorkflowState)

    def make_node(name):
        async def node(state: WorkflowState) -> WorkflowState:
            if on_node is not None:
                await on_node(name)
            state.add_log(f"{name} finished")
            if name == NODES[-1]:
                state.mark_completed({"title": "R", "content": "done"})
            return state

        return node

    for name in NODES:
        graph.add_node(name, make_node(name))
    graph.set_entry_point(NODES[0])
    for current, following in zip(NODES, NODES[1:]):
        graph.add_edge(current, following)
    graph.add_edge(NODES[-1], END)
    return graph.compile()


@pytest.mark.asyncio
async def test_astream_yields_mode_and_chunk_tuples():
    items = [item async for item in _three_node_graph().astream(
        WorkflowState(query="q"), stream_mode=["updates", "values"]
    )]

    assert all(isinstance(item, tuple) and len(item) == 2 for item in items)
    updates = [chunk for mode, chunk in items if mode == "updates"]
    assert [next(iter(chunk)) for chunk in updates] == NODES
    assert {mode for mode, _ in items} == {"updates", "values"}
    assert items[-1][0] == "values"


@pytest.mark.asyncio
async def test_each_node_saves_progress_that_another_worker_reads(monkeypatch):
    runner, other = separate_task_manager(), separate_task_manager()
    monkeypatch.setattr(task_endpoint, "task_manager", other)
    task_id = str(uuid4())
    seen_by_other = []

    async def poll_from_other_worker(_name):
        seen_by_other.append(await task_endpoint.task_status(task_id))

    saves = []
    real_save = progress.save
    monkeypatch.setattr(progress, "save", lambda *args: saves.append(args) or real_save(*args))
    monkeypatch.setattr(task_manager_module, "get_agent_workflow", lambda: _three_node_graph(poll_from_other_worker))

    runner.register_pending(task_id, "a question")
    await runner.run_workflow(task_id, "a question", {})

    assert [node for _task, node, _logs in saves] == NODES
    assert [body["status"] for body in seen_by_other] == ["processing"] * 3
    assert "progress" not in seen_by_other[0]
    assert seen_by_other[1]["progress"]["step"] == "Planning the research"
    assert seen_by_other[2]["progress"]["step"] == "Web and internal research"
    assert seen_by_other[2]["progress"]["logs"] == ["supervisor finished", "research finished"]

    saved = result_store.load_result(task_id)
    assert saved["status"] == "completed"
    assert "progress" not in saved


def test_progress_keeps_the_last_five_log_lines_and_labels_unknown_nodes():
    built = progress.build("data_cleanup", [f"line {i}" for i in range(8)])
    assert built["step"] == "data cleanup"
    assert built["logs"] == ["line 3", "line 4", "line 5", "line 6", "line 7"]
    assert progress.step_label("senior_research") == "Checking the findings"


def test_progress_is_not_saved_once_the_task_is_final():
    task_id = str(uuid4())
    result_store.save_result(task_id, {"task_id": task_id, "status": "completed"})
    assert progress.save(task_id, "research", ["x"]) is False
    assert "progress" not in result_store.load_result(task_id)
