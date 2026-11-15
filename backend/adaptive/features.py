from graph.graph import Graph


def extract_features(graph: Graph) -> dict:
    node_count = len(graph.adjacency)
    edge_count = sum(len(edges) for edges in graph.adjacency.values())

    # Each undirected edge appears twice in the adjacency list.
    if not graph.directed:
        edge_count //= 2

    possible_edges = (
        node_count * (node_count - 1)
        if graph.directed
        else node_count * (node_count - 1) / 2
    )

    weights = [
        weight
        for edges in graph.adjacency.values()
        for _, weight in edges
    ]
    degrees = [len(edges) for edges in graph.adjacency.values()]

    return {
        "node_count": node_count,
        "edge_count": edge_count,
        "density": edge_count / possible_edges if possible_edges else 0.0,
        "directed": graph.directed,
        "all_unit_weights": graph.has_only_unit_weights(),
        "uniform_weights": graph.has_uniform_weights(),
        "has_negative_weights": graph.has_negative_weights(),
        "minimum_weight": min(weights) if weights else None,
        "maximum_weight": max(weights) if weights else None,
        "mean_weight": sum(weights) / len(weights) if weights else None,
        "average_degree": sum(degrees) / node_count if node_count else 0.0,
        "positions_available": graph.has_positions_for_all_nodes(),
    }
