from graph.graph import Graph


def extract_features(graph: Graph, include_heuristic: bool = True) -> dict:
    node_count = len(graph.adjacency)
    degrees = [len(edges) for edges in graph.adjacency.values()]
    positions_available = graph.has_positions_for_all_nodes()
    heuristic_scale = None

    if include_heuristic and positions_available and not graph.has_negative_weights():
        heuristic_scale = graph.heuristic_scale()

    possible_edges = (
        node_count * (node_count - 1)
        if graph.directed
        else node_count * (node_count - 1) / 2
    )
    return {
        "node_count": node_count,
        "edge_count": graph.edge_count,
        "density": graph.edge_count / possible_edges if possible_edges else 0.0,
        "directed": graph.directed,
        "all_unit_weights": graph.has_only_unit_weights(),
        "uniform_weights": graph.has_uniform_weights(),
        "has_negative_weights": graph.has_negative_weights(),
        "minimum_weight": graph.minimum_weight,
        "maximum_weight": graph.maximum_weight,
        "mean_weight": graph.mean_weight,
        "weight_stddev": graph.weight_stddev,
        "unique_weight_ratio": graph.unique_weight_ratio,
        "average_degree": sum(degrees) / node_count if node_count else 0.0,
        "minimum_degree": min(degrees) if degrees else 0,
        "maximum_degree": max(degrees) if degrees else 0,
        "positions_available": positions_available,
        "heuristic_scale": heuristic_scale,
    }
