from graph.graph import Graph
from algorithms import astar, bfs, bellman_ford, dijkstra


def choose_algorithm(graph: Graph, source: str | None = None,
                     destination: str | None = None) -> str:
    if graph.has_negative_weights():
        return "bellman_ford"

    if graph.has_uniform_nonnegative_weights():
        return "bfs"

    if (source is not None and destination is not None
            and graph.has_positions_for_all_nodes()):
        scale = astar._heuristic_scale(graph)
        if scale > 0 and source != destination:
            return "astar"

    return "dijkstra"


def find_shortest_path(graph: Graph, source: str, destination: str) -> dict:
    if source not in graph.adjacency:
        raise KeyError(f"Unknown source node: {source}")
    if destination not in graph.adjacency:
        raise KeyError(f"Unknown destination node: {destination}")

    algorithm = choose_algorithm(graph, source, destination)

    implementations = {
        "bfs": bfs.shortest_path,
        "dijkstra": dijkstra.shortest_path,
        "astar": astar.shortest_path,
        "bellman_ford": bellman_ford.shortest_path,
    }

    result = implementations[algorithm](graph, source, destination)
    result["algorithm"] = algorithm
    return result
