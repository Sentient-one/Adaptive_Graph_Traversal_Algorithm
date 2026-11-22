import math


class Graph:
    """Weighted graph stored as an adjacency list, with optional node positions."""

    def __init__(self, directed: bool = False):
        self.directed = directed
        self.adjacency: dict[str, list[tuple[str, float]]] = {}
        self.positions: dict[str, tuple[float, float]] = {}
        self._position_count = 0
        self._edge_count = 0
        self._weight_count = 0
        self._minimum_weight: float | None = None
        self._maximum_weight: float | None = None
        self._mean_weight = 0.0
        self._weight_m2 = 0.0
        self._unique_weights: set[float] = set()
        self._all_unit_weights = True
        self._has_negative_weights = False
        self._heuristic_scale_cache: float | None = None
        self._heuristic_scale_dirty = True

    def add_node(self, node: str) -> None:
        self.adjacency.setdefault(node, [])

    def set_position(self, node: str, x: float, y: float) -> None:
        try:
            x_value, y_value = float(x), float(y)
        except (TypeError, ValueError) as error:
            raise ValueError("Node coordinates must be finite numbers.") from error
        if not math.isfinite(x_value) or not math.isfinite(y_value):
            raise ValueError("Node coordinates must be finite numbers.")
        self.add_node(node)
        if node not in self.positions:
            self._position_count += 1
        self.positions[node] = (x_value, y_value)
        self._heuristic_scale_dirty = True

    def has_positions_for_all_nodes(self) -> bool:
        return bool(self.adjacency) and self._position_count == len(self.adjacency)

    def add_edge(self, source: str, destination: str, weight: float = 1) -> None:
        if source == destination:
            raise ValueError("Self-loops are not supported.")
        try:
            numeric_weight = float(weight)
        except (TypeError, ValueError) as error:
            raise ValueError("Edge weight must be a finite number.") from error
        if not math.isfinite(numeric_weight):
            raise ValueError("Edge weight must be a finite number.")
        if any(neighbor == destination for neighbor, _ in self.adjacency.get(source, [])):
            raise ValueError(f"Duplicate edge: {source!r} to {destination!r}")
        if not self.directed and any(
            neighbor == source for neighbor, _ in self.adjacency.get(destination, [])
        ):
            raise ValueError(f"Duplicate edge: {source!r} to {destination!r}")
        self.add_node(source)
        self.add_node(destination)
        self.adjacency[source].append((destination, numeric_weight))

        if not self.directed:
            self.adjacency[destination].append((source, numeric_weight))
        self._edge_count += 1
        self._heuristic_scale_dirty = True
        self._all_unit_weights = self._all_unit_weights and numeric_weight == 1
        self._has_negative_weights = self._has_negative_weights or numeric_weight < 0
        self._unique_weights.add(numeric_weight)
        copies = 1 if self.directed else 2
        for _ in range(copies):
            self._weight_count += 1
            delta = numeric_weight - self._mean_weight
            self._mean_weight += delta / self._weight_count
            self._weight_m2 += delta * (numeric_weight - self._mean_weight)
        self._minimum_weight = (
            numeric_weight if self._minimum_weight is None
            else min(self._minimum_weight, numeric_weight)
        )
        self._maximum_weight = (
            numeric_weight if self._maximum_weight is None
            else max(self._maximum_weight, numeric_weight)
        )

    def neighbors(self, node: str) -> list[tuple[str, float]]:
        if node not in self.adjacency:
            raise KeyError(f"Unknown node: {node}")
        return self.adjacency[node]

    def has_only_unit_weights(self) -> bool:
        return self._all_unit_weights

    def has_uniform_nonnegative_weights(self) -> bool:
        return (
            self._minimum_weight is None
            or (self._minimum_weight >= 0 and self._minimum_weight == self._maximum_weight)
        )

    def has_uniform_weights(self) -> bool:
        return self._minimum_weight is None or self._minimum_weight == self._maximum_weight

    def uniform_edge_weight(self) -> float | None:
        return (
            self._minimum_weight
            if self._minimum_weight is not None and self._minimum_weight == self._maximum_weight
            else None
        )

    def has_negative_weights(self) -> bool:
        return self._has_negative_weights

    @property
    def edge_count(self) -> int:
        return self._edge_count

    @property
    def mean_weight(self) -> float | None:
        return self._mean_weight if self._weight_count else None

    @property
    def minimum_weight(self) -> float | None:
        return self._minimum_weight

    @property
    def maximum_weight(self) -> float | None:
        return self._maximum_weight

    @property
    def weight_stddev(self) -> float | None:
        return (
            math.sqrt(self._weight_m2 / self._weight_count)
            if self._weight_count else None
        )

    @property
    def unique_weight_ratio(self) -> float:
        return len(self._unique_weights) / self._weight_count if self._weight_count else 0.0

    def heuristic_scale(self) -> float:
        if not self.has_positions_for_all_nodes():
            raise ValueError("A* requires coordinates for every graph node.")
        if self.has_negative_weights():
            raise ValueError("A* cannot handle negative edge weights.")
        if not self._heuristic_scale_dirty:
            return self._heuristic_scale_cache or 0.0

        scale = float("inf")
        positions = self.positions
        hypot = math.hypot
        for start, neighbors in self.adjacency.items():
            x1, y1 = positions[start]
            for end, weight in neighbors:
                x2, y2 = positions[end]
                distance = hypot(x2 - x1, y2 - y1)
                if distance > 0:
                    scale = min(scale, weight / distance)
        self._heuristic_scale_cache = 0.0 if scale == float("inf") else scale
        self._heuristic_scale_dirty = False
        return self._heuristic_scale_cache
