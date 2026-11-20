from collections.abc import Callable

from adaptive.features import extract_features
from graph.graph import Graph
from algorithms import astar, bfs, bellman_ford, dijkstra


def eligible_algorithms(
    graph: Graph, source: str, destination: str, features: dict | None = None
) -> dict[str, Callable]:
    features = features or extract_features(graph)
    candidates = {"bellman_ford": bellman_ford.shortest_path}
    if not features["has_negative_weights"]:
        candidates["dijkstra"] = dijkstra.shortest_path
        if features["positions_available"]:
            if features["heuristic_scale"] is not None and (
                features["heuristic_scale"] > 0 or source == destination
            ):
                candidates["astar"] = astar.shortest_path
    if (features["uniform_weights"] and not features["has_negative_weights"]):
        candidates["bfs"] = bfs.shortest_path
    return candidates


def choose_algorithm(graph: Graph, source: str | None = None,
                     destination: str | None = None) -> str:
    if graph.has_negative_weights():
        return "bellman_ford"

    if graph.has_uniform_nonnegative_weights():
        return "bfs"

    features = extract_features(graph)
    if (source is not None and destination is not None
            and features["positions_available"]):
        scale = features["heuristic_scale"] or 0.0
        if scale > 0 and source != destination:
            return "astar"

    return "dijkstra"


def find_shortest_path(
    graph: Graph,
    source: str,
    destination: str,
    performance_model=None,
) -> dict:
    if source not in graph.adjacency:
        raise KeyError(f"Unknown source node: {source}")
    if destination not in graph.adjacency:
        raise KeyError(f"Unknown destination node: {destination}")

    if performance_model is None:
        algorithm = choose_algorithm(graph, source, destination)
    else:
        algorithm = performance_model.choose_algorithm(graph, source, destination)

    implementations = {
        "bfs": bfs.shortest_path,
        "dijkstra": dijkstra.shortest_path,
        "astar": astar.shortest_path,
        "bellman_ford": bellman_ford.shortest_path,
    }

    result = implementations[algorithm](graph, source, destination)
    result["algorithm"] = algorithm
    return result
