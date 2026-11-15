import heapq
from itertools import count
from graph.graph import Graph


def shortest_path(graph: Graph, source: str, destination: str) -> dict:
    if source not in graph.adjacency:
        raise KeyError(f"Unknown source node: {source}")
    if destination not in graph.adjacency:
        raise KeyError(f"Unknown destination node: {destination}")
    if graph.has_negative_weights():
        raise ValueError("Dijkstra cannot handle negative edge weights.")

    distances = {node: float("inf") for node in graph.adjacency}
    parent: dict[str, str | None] = {source: None}
    distances[source] = 0

    tie_breaker = count()
    priority_queue = [(0, next(tie_breaker), source)]
    explored_nodes = 0
    edge_scans = 0

    while priority_queue:
        current_distance, _, current = heapq.heappop(priority_queue)

        # Ignore outdated entries in the priority queue.
        if current_distance > distances[current]:
            continue

        explored_nodes += 1

        if current == destination:
            break

        for neighbor, weight in graph.neighbors(current):
            edge_scans += 1
            new_distance = current_distance + weight

            if new_distance < distances[neighbor]:
                distances[neighbor] = new_distance
                parent[neighbor] = current
                heapq.heappush(
                    priority_queue,
                    (new_distance, next(tie_breaker), neighbor),
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
