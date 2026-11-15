from graph.graph import Graph


def shortest_path(graph: Graph, source: str, destination: str) -> dict:
    if source not in graph.adjacency:
        raise KeyError(f"Unknown source node: {source}")
    if destination not in graph.adjacency:
        raise KeyError(f"Unknown destination node: {destination}")

    nodes = list(graph.adjacency)
    edges = [
        (node, neighbor, weight)
        for node, neighbors in graph.adjacency.items()
        for neighbor, weight in neighbors
    ]

    distances = {node: float("inf") for node in nodes}
    parent: dict[str, str | None] = {source: None}
    distances[source] = 0
    passes_executed = 0
    relaxation_attempts = 0
    successful_relaxations = 0

    for _ in range(len(nodes) - 1):
        passes_executed += 1
        changed = False

        for start, end, weight in edges:
            relaxation_attempts += 1
            if distances[start] != float("inf"):
                candidate = distances[start] + weight
                if candidate < distances[end]:
                    distances[end] = candidate
                    parent[end] = start
                    changed = True
                    successful_relaxations += 1

        if not changed:
            break

    for start, end, weight in edges:
        relaxation_attempts += 1
        if distances[start] != float("inf") and distances[start] + weight < distances[end]:
            raise ValueError("A reachable negative-weight cycle exists.")

    if distances[destination] == float("inf"):
        return {"path": None, "cost": None, "explored_nodes": None,
                "passes_executed": passes_executed,
                "relaxation_attempts": relaxation_attempts,
                "successful_relaxations": successful_relaxations}

    path = []
    current = destination
    while current is not None:
        path.append(current)
        current = parent[current]

    path.reverse()

    return {
        "path": path,
        "cost": distances[destination],
        "explored_nodes": None,
        "passes_executed": passes_executed,
        "relaxation_attempts": relaxation_attempts,
        "successful_relaxations": successful_relaxations,
    }
