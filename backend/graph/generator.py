
import math
import random

from graph.graph import Graph


def generate_random_graph(
    node_count: int,
    edge_probability: float = 0.2,
    weighted: bool = False,
    directed: bool = False,
    seed: int | None = None,
    weight_mode: str | None = None,
    coordinates: bool = True,
) -> Graph:
    """
    Generate a reproducible random graph.

    weight_mode may be unit, uniform, positive, or negative. Negative mode
    uses directed edges with potential-based weights, so generated cycles are
    positive and Bellman-Ford has no negative cycle to reject.
    The weighted flag remains as a compatibility shortcut for positive mode.
    """

    if node_count < 1:
        raise ValueError("node_count must be at least 1")

    if not 0 <= edge_probability <= 1:
        raise ValueError("edge_probability must be between 0 and 1")

    if weight_mode is None:
        weight_mode = "positive" if weighted else "unit"
    if weight_mode not in {"unit", "uniform", "positive", "negative"}:
        raise ValueError("weight_mode must be unit, uniform, positive, or negative")
    if weight_mode == "negative" and not directed:
        raise ValueError("negative weight mode requires a directed graph")

    rng = random.Random(seed)
    graph = Graph(directed=directed)

    nodes = [str(i) for i in range(node_count)]

    # Coordinates can be disabled to benchmark graphs without A* eligibility.
    for node in nodes:
        graph.add_node(node)
        if coordinates:
            graph.set_position(node, x=rng.uniform(0, 100), y=rng.uniform(0, 100))

    potentials = {node: rng.uniform(-8, 8) for node in nodes}
    uniform_weight = rng.uniform(1.0, 10.0) if weight_mode == "uniform" else 1.0

    # Generate candidate edges.
    if directed:
        candidate_edges = (
            (source, destination)
            for source in nodes
            for destination in nodes
            if source != destination
        )
    else:
        candidate_edges = (
            (nodes[i], nodes[j])
            for i in range(node_count)
            for j in range(i + 1, node_count)
        )

    for source, destination in candidate_edges:
        if rng.random() < edge_probability:
            if weight_mode == "positive":
                weight = rng.uniform(1.0, 10.0)
            elif weight_mode == "uniform":
                weight = uniform_weight
            elif weight_mode == "negative":
                weight = rng.uniform(0.25, 2.0) + potentials[destination] - potentials[source]
            else:
                weight = 1.0

            graph.add_edge(source, destination, weight)

    return graph
