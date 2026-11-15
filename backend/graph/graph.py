import math


class Graph:
    """Weighted graph stored as an adjacency list, with optional node positions."""

    def __init__(self, directed: bool = False):
        self.directed = directed
        self.adjacency: dict[str, list[tuple[str, float]]] = {}
        self.positions: dict[str, tuple[float, float]] = {}

    def add_node(self, node: str) -> None:
        self.adjacency.setdefault(node, [])

    def set_position(self, node: str, x: float, y: float) -> None:
        if not math.isfinite(float(x)) or not math.isfinite(float(y)):
            raise ValueError("Node coordinates must be finite numbers.")
        self.add_node(node)
        self.positions[node] = (float(x), float(y))

    def has_positions_for_all_nodes(self) -> bool:
        return bool(self.adjacency) and all(
            node in self.positions for node in self.adjacency
        )

    def add_edge(self, source: str, destination: str, weight: float = 1) -> None:
        try:
            numeric_weight = float(weight)
        except (TypeError, ValueError) as error:
            raise ValueError("Edge weight must be a finite number.") from error
        if not math.isfinite(numeric_weight):
            raise ValueError("Edge weight must be a finite number.")
        self.add_node(source)
        self.add_node(destination)
        self.adjacency[source].append((destination, numeric_weight))

        if not self.directed:
            self.adjacency[destination].append((source, numeric_weight))

    def neighbors(self, node: str) -> list[tuple[str, float]]:
        if node not in self.adjacency:
            raise KeyError(f"Unknown node: {node}")
        return self.adjacency[node]

    def has_only_unit_weights(self) -> bool:
        return all(
            weight == 1
            for edges in self.adjacency.values()
            for _, weight in edges
        )

    def has_uniform_nonnegative_weights(self) -> bool:
        weights = [weight for edges in self.adjacency.values() for _, weight in edges]
        return not weights or (min(weights) >= 0 and min(weights) == max(weights))

    def has_uniform_weights(self) -> bool:
        weights = [weight for edges in self.adjacency.values() for _, weight in edges]
        return not weights or min(weights) == max(weights)

    def uniform_edge_weight(self) -> float | None:
        weights = [weight for edges in self.adjacency.values() for _, weight in edges]
        return weights[0] if weights and min(weights) == max(weights) else None

    def has_negative_weights(self) -> bool:
        return any(
            weight < 0
            for edges in self.adjacency.values()
            for _, weight in edges
        )
