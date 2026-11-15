
import math
import random

from graph.graph import Graph


def generate_random_graph(
    node_count: int,
    edge_probability: float = 0.2,
    weighted: bool = False,
    directed: bool = False,
    seed: int | None = None,
) -> Graph:
    """
    Generate a reproducible random graph.

    weighted=False: every edge has weight 1.
    weighted=True: every edge receives a random positive weight.
    directed=True: edges have a direction.
    """

    if node_count < 1:
        raise ValueError("node_count must be at least 1")

    if not 0 <= edge_probability <= 1:
        raise ValueError("edge_probability must be between 0 and 1")

    rng = random.Random(seed)
    graph = Graph(directed=directed)

    nodes = [str(i) for i in range(node_count)]

    # Add every node and assign coordinates for A* experiments.
    for node in nodes:
        graph.set_position(
            node,
            x=rng.uniform(0, 100),
            y=rng.uniform(0, 100),
        )

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
            if weighted:
                weight = rng.uniform(1.0, 10.0)
            else:
                weight = 1.0

            graph.add_edge(source, destination, weight)

    return graph
