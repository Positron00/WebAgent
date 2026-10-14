"""IMP-081: every workflow node is reachable from __start__."""
from langgraph.graph import END

from app.graph.workflows import _build_agent_workflow_impl


def _edge_endpoints(edge):
    if isinstance(edge, tuple):
        return edge[0], edge[1]
    return edge.source, edge.target


def _reachable_from_start(graph) -> set[str]:
    successors: dict[str, set[str]] = {}
    for edge in graph.edges:
        src, dst = _edge_endpoints(edge)
        if dst is END:
            continue
        successors.setdefault(src, set()).add(dst)

    seen = {"__start__"}
    queue = ["__start__"]
    while queue:
        node = queue.pop(0)
        for nxt in successors.get(node, ()):
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return seen


def test_all_workflow_nodes_reachable():
    compiled = _build_agent_workflow_impl()
    graph = compiled.get_graph()
    node_names = {n for n in graph.nodes if n not in ("__start__", END)}
    reachable = _reachable_from_start(graph)
    unreachable = node_names - reachable
    assert not unreachable, f"Unreachable nodes: {sorted(unreachable)}"
