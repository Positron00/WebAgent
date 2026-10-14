"""Base class for fake workflows that only define `ainvoke`.

The task manager runs a graph with `astream(state, stream_mode=["updates", "values"])`, which
yields `(mode, chunk)` tuples. This base turns one `ainvoke` call into one node update
followed by the final values, the same shape LangGraph 0.2 produces.
"""


class StreamingFakeWorkflow:
    node_name = "fake_node"

    async def ainvoke(self, state):  # pragma: no cover - subclasses override
        raise NotImplementedError

    async def astream(self, state, stream_mode=None):
        result = await self.ainvoke(state)
        yield "updates", {self.node_name: result}
        yield "values", result
