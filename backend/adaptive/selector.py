
from collections.abc import Callable

from adaptive.features import extract_features
from adaptive.validity import (
    SUPPORTED_ALGORITHMS,
    validate_algorithm_input,
)
from graph.graph import Graph

from algorithms import astar, bfs, bellman_ford, dijkstra


ALGORITHM_IMPLEMENTATIONS = {
    "bfs": bfs.shortest_path,
    "dijkstra": dijkstra.shortest_path,
    "astar": astar.shortest_path,
    "bellman_ford": bellman_ford.shortest_path,
}


def eligible_algorithms(
    graph: Graph,
    source: str,
    destination: str,
    features: dict | None = None,
) -> dict[str, Callable]:
    """
    Return all algorithms valid for the given graph.
    """

    if features is None:
        features = extract_features(graph)

    candidates = {}

    # Sorted ordering ensures deterministic candidate selection.
    for algorithm in sorted(SUPPORTED_ALGORITHMS):
        try:
            validate_algorithm_input(
                graph,
                source,
                destination,
                algorithm,
            )
        except ValueError:
            continue

        candidates[algorithm] = ALGORITHM_IMPLEMENTATIONS[algorithm]

    return candidates


def choose_algorithm(
    graph: Graph,
    source: str | None = None,
    destination: str | None = None,
) -> str:
    """
    Basic rule-based adaptive selection.
    """

    if source is None:
        source = next(iter(graph.adjacency), None)

    if destination is None:
        destination = source

    if source is None or destination is None:
        raise ValueError(
            "Cannot select an algorithm for an empty graph."
        )

    features = extract_features(graph)

    candidates = eligible_algorithms(
        graph,
        source,
        destination,
        features=features,
    )

    if not candidates:
        raise ValueError(
            "No valid shortest-path algorithm available."
        )

    if "bfs" in candidates:
        return "bfs"

    if "astar" in candidates:
        scale = features["heuristic_scale"]

        if (
            scale is not None
            and scale > 0
            and source != destination
        ):
            return "astar"

    if "dijkstra" in candidates:
        return "dijkstra"

    if "bellman_ford" in candidates:
        return "bellman_ford"

    raise ValueError("No suitable algorithm found.")


def find_shortest_path(
    graph: Graph,
    source: str,
    destination: str,
    performance_model=None,
) -> dict:
    """
    Select and execute a valid shortest-path algorithm.

    Supports rule-based and benchmark-trained selection.
    """

    if source not in graph.adjacency:
        raise ValueError(f"Unknown source node: {source}")

    if destination not in graph.adjacency:
        raise ValueError(f"Unknown destination node: {destination}")

    if performance_model is None:
        algorithm = choose_algorithm(
            graph,
            source,
            destination,
        )
    else:
        algorithm = performance_model.choose_algorithm(
            graph,
            source,
            destination,
        )

    # Validate the selected algorithm before execution.
    validate_algorithm_input(
        graph,
        source,
        destination,
        algorithm,
    )

    implementation = ALGORITHM_IMPLEMENTATIONS[algorithm]

    result = implementation(
        graph,
        source,
        destination,
    )

    result["algorithm"] = algorithm

    return result
