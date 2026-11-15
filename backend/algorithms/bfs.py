from collections import deque
from graph.graph import Graph


def shortest_path(graph: Graph, source: str, destination: str) -> dict:
    if source not in graph.adjacency:
        raise KeyError(f"Unknown source node: {source}")
    if destination not in graph.adjacency:
        raise KeyError(f"Unknown destination node: {destination}")
    if not graph.has_uniform_nonnegative_weights():
        raise ValueError("BFS requires all edges to have the same nonnegative weight.")

    queue = deque([source])
    parent: dict[str, str | None] = {source: None}
    explored_nodes = 0
    edge_scans = 0

    while queue:
        current = queue.popleft()
        explored_nodes += 1

        if current == destination:
            break

        for neighbor, _ in graph.neighbors(current):
            edge_scans += 1
            if neighbor not in parent:
                parent[neighbor] = current
                queue.append(neighbor)

    if destination not in parent:
        return {"path": None, "cost": None, "explored_nodes": explored_nodes,
                "edge_scans": edge_scans}

    path = []
    current = destination
    while current is not None:
        path.append(current)
        current = parent[current]

    path.reverse()

    return {
        "path": path,
        "cost": (len(path) - 1) * (graph.uniform_edge_weight() or 0.0),
        "explored_nodes": explored_nodes,
        "edge_scans": edge_scans,
    }
