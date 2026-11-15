from graph.graph import Graph


def validate_algorithm_input(
    graph: Graph,
    source: str,
    destination: str,
    algorithm: str,
) -> None:
    if source not in graph.adjacency:
        raise ValueError(f"Unknown source node: {source}")
    if destination not in graph.adjacency:
        raise ValueError(f"Unknown destination node: {destination}")

    supported = {"bfs", "dijkstra", "astar", "bellman_ford"}
    if algorithm not in supported:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    if algorithm == "bfs" and not graph.has_uniform_nonnegative_weights():
        raise ValueError("BFS requires all edges to have the same nonnegative weight.")

    if algorithm in {"dijkstra", "astar"} and graph.has_negative_weights():
        raise ValueError(f"{algorithm} cannot handle negative edge weights.")
