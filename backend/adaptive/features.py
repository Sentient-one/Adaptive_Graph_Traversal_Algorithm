
from graph.graph import Graph


def _get_connectivity_features(graph: Graph) -> dict:
    """
    Calculate connectivity-related features.

    For directed graphs, component_count represents weakly
    connected components.
    """

    nodes = list(graph.adjacency)

    if not nodes:
        return {
            "component_count": 0,
            "is_connected": False,
            "is_weakly_connected": False,
            "isolated_node_count": 0,
        }

    # Build an undirected view of the graph.
    # This allows us to calculate weak connectivity.
    undirected_adjacency = {node: set() for node in nodes}

    for source, neighbors in graph.adjacency.items():
        for destination, _ in neighbors:
            undirected_adjacency[source].add(destination)
            undirected_adjacency[destination].add(source)

    visited = set()
    component_count = 0

    for node in nodes:
        if node in visited:
            continue

        component_count += 1
        stack = [node]
        visited.add(node)

        while stack:
            current = stack.pop()

            for neighbor in undirected_adjacency[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)

    isolated_node_count = sum(
        1 for node in nodes
        if len(graph.adjacency[node]) == 0
    )

    weakly_connected = component_count == 1

    return {
        "component_count": component_count,
        "is_connected": (
            weakly_connected if not graph.directed else None
        ),
        "is_weakly_connected": weakly_connected,
        "isolated_node_count": isolated_node_count,
    }


def extract_features(
    graph: Graph,
    include_heuristic: bool = True
) -> dict:
    """
    Extract structural and weight-related features from a graph.
    """

    node_count = len(graph.adjacency)
    edge_count = graph.edge_count

    # Degree statistics
    out_degrees = [
        len(edges)
        for edges in graph.adjacency.values()
    ]

    in_degrees = {node: 0 for node in graph.adjacency}

    for neighbors in graph.adjacency.values():
        for destination, _ in neighbors:
            in_degrees[destination] += 1

    in_degree_values = list(in_degrees.values())

    # Graph density
    if node_count <= 1:
        density = 0.0
    else:
        possible_edges = (
            node_count * (node_count - 1)
            if graph.directed
            else node_count * (node_count - 1) / 2
        )

        density = edge_count / possible_edges

    # Weight characteristics
    has_edges = edge_count > 0
    all_unit_weights = graph.has_only_unit_weights()

    is_weighted = has_edges and not all_unit_weights
    has_negative_weights = graph.has_negative_weights()

    # A* heuristic information
    positions_available = graph.has_positions_for_all_nodes()

    heuristic_scale = None

    if (
        include_heuristic
        and positions_available
        and not has_negative_weights
    ):
        heuristic_scale = graph.heuristic_scale()

    # Connectivity
    connectivity = _get_connectivity_features(graph)

    return {
        # Basic graph characteristics
        "node_count": node_count,
        "edge_count": edge_count,
        "density": density,
        "directed": graph.directed,
        "has_edges": has_edges,

        # Weight characteristics
        "is_weighted": is_weighted,
        "all_unit_weights": all_unit_weights,
        "uniform_weights": graph.has_uniform_weights(),
        "has_negative_weights": has_negative_weights,
        "minimum_weight": graph.minimum_weight,
        "maximum_weight": graph.maximum_weight,
        "mean_weight": graph.mean_weight,
        "weight_stddev": graph.weight_stddev,
        "unique_weight_ratio": graph.unique_weight_ratio,

        # Out-degree characteristics
        "average_degree": (
            sum(out_degrees) / node_count if node_count else 0.0
        ),
        "average_out_degree": (
            sum(out_degrees) / node_count if node_count else 0.0
        ),
        "minimum_degree": min(out_degrees) if out_degrees else 0,
        "maximum_degree": max(out_degrees) if out_degrees else 0,

        # In-degree characteristics
        "average_in_degree": (
            sum(in_degree_values) / node_count if node_count else 0.0
        ),
        "minimum_in_degree": (
            min(in_degree_values) if in_degree_values else 0
        ),
        "maximum_in_degree": (
            max(in_degree_values) if in_degree_values else 0
        ),

        # Connectivity characteristics
        **connectivity,

        # Spatial characteristics
        "positions_available": positions_available,
        "heuristic_scale": heuristic_scale,
    }
