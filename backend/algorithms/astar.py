import heapq
import math
from itertools import count

from graph.graph import Graph


def _heuristic_scale(graph: Graph) -> float:
    return graph.heuristic_scale()


def shortest_path(graph: Graph, source: str, destination: str) -> dict:
    if source not in graph.adjacency:
        raise KeyError(f"Unknown source node: {source}")
    if destination not in graph.adjacency:
        raise KeyError(f"Unknown destination node: {destination}")
    if graph.has_negative_weights():
        raise ValueError("A* cannot handle negative edge weights.")

    scale = _heuristic_scale(graph)

    def heuristic(node: str) -> float:
        x1, y1 = graph.positions[node]
        x2, y2 = graph.positions[destination]
        return scale * math.hypot(x2 - x1, y2 - y1)

    distances = {node: float("inf") for node in graph.adjacency}
    parent: dict[str, str | None] = {source: None}
    distances[source] = 0.0

    tie_breaker = count()
    queue = [(heuristic(source), next(tie_breaker), 0.0, source)]
    explored_nodes = 0
    edge_scans = 0

    while queue:
        _, _, current_cost, current = heapq.heappop(queue)

        if current_cost > distances[current]:
            continue

        explored_nodes += 1

        if current == destination:
            break

        for neighbor, weight in graph.neighbors(current):
            edge_scans += 1
            new_cost = current_cost + weight

            if new_cost < distances[neighbor]:
                distances[neighbor] = new_cost
                parent[neighbor] = current
                priority = new_cost + heuristic(neighbor)
                heapq.heappush(
                    queue,
                    (priority, next(tie_breaker), new_cost, neighbor),
                )

    if distances[destination] == float("inf"):
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
        "cost": distances[destination],
        "explored_nodes": explored_nodes,
        "edge_scans": edge_scans,
    }
