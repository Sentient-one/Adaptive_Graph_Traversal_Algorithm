
from graph.graph import Graph


SUPPORTED_ALGORITHMS = {
    "bfs",
    "dijkstra",
    "astar",
    "bellman_ford",
}


def validate_algorithm_input(
    graph: Graph,
    source: str,
    destination: str,
    algorithm: str,
) -> None:
    """
    Validate whether an algorithm can correctly process
    the given graph and source-destination pair.

    Raises:
        ValueError: If the input or algorithm is invalid.
    """

    if source not in graph.adjacency:
        raise ValueError(f"Unknown source node: {source}")

    if destination not in graph.adjacency:
        raise ValueError(f"Unknown destination node: {destination}")

    if algorithm not in SUPPORTED_ALGORITHMS:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    # BFS is valid when all edges have the same
    # non-negative weight.
    if algorithm == "bfs":
        if not graph.has_uniform_nonnegative_weights():
            raise ValueError(
                "BFS requires all edges to have the same "
                "nonnegative weight."
            )

    # Dijkstra cannot process negative edge weights.
    if algorithm == "dijkstra":
        if graph.has_negative_weights():
            raise ValueError(
                "Dijkstra cannot handle negative edge weights."
            )

    # A* requires non-negative weights and coordinates
    # for every graph node.
    if algorithm == "astar":
        if graph.has_negative_weights():
            raise ValueError(
                "A* cannot handle negative edge weights."
            )

        if not graph.has_positions_for_all_nodes():
            raise ValueError(
                "A* requires coordinates for every graph node."
            )

    # Bellman-Ford supports negative edge weights
    # and detects reachable negative-weight cycles.
    # No additional graph restrictions are required.
