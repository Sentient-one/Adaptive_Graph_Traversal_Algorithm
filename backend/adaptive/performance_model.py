
"""Dependency-free ridge regression over development benchmark measurements."""

import csv
import math
from pathlib import Path
from statistics import mean

from adaptive.features import extract_selection_features
from adaptive.selector import eligible_algorithms, choose_algorithm


def _number(value, default=0.0) -> float:
    if isinstance(value, str) and value.strip().lower() in {
        "true", "false"
    }:
        return 1.0 if value.strip().lower() == "true" else 0.0

    if value is None or value == "":
        return default

    try:
        number = float(value)
    except (TypeError, ValueError):
        return default

    return number if math.isfinite(number) else default


def _vector(values: dict) -> tuple[float, ...]:
    return (
        math.log1p(
            _number(values.get("node_count", values.get("nodes")))
        ),
        _number(values.get("density")),
        _number(values.get("directed")),
        _number(values.get("uniform_weights")),
        _number(values.get("has_negative_weights")),
        _number(values.get("average_degree")),
        _number(values.get("weight_stddev")),
        _number(values.get("positions_available")),
        _number(values.get("heuristic_scale")),
    )


def _solve(
    matrix: list[list[float]],
    vector: list[float],
) -> list[float]:
    """Solve a small dense linear system with partial pivoting."""

    size = len(vector)

    augmented = [
        row[:] + [value]
        for row, value in zip(matrix, vector)
    ]

    for column in range(size):
        pivot = max(
            range(column, size),
            key=lambda row: abs(augmented[row][column]),
        )

        if abs(augmented[pivot][column]) < 1e-12:
            augmented[pivot][column] = 1e-12

        augmented[column], augmented[pivot] = (
            augmented[pivot],
            augmented[column],
        )

        divisor = augmented[column][column]

        augmented[column] = [
            value / divisor
            for value in augmented[column]
        ]

        for row in range(size):
            if row == column:
                continue

            factor = augmented[row][column]

            augmented[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(
                    augmented[row],
                    augmented[column],
                )
            ]

    return [
        augmented[row][-1]
        for row in range(size)
    ]


class BenchmarkRuntimeSelector:
    """
    Estimate eligible algorithm runtimes from development
    benchmark rows using ridge regression.

    Only development benchmark data should be used for training.
    """

    def __init__(
        self,
        rows: list[dict],
        ridge: float = 1e-3,
    ):
        if not math.isfinite(ridge) or ridge <= 0:
            raise ValueError("ridge must be positive")

        self.ridge = ridge

        cases = {}

        for row in rows:
            if row.get("algorithm") == "adaptive":
                continue

            runtime = _number(
                row.get("median_seconds"),
                -1,
            )

            if runtime <= 0:
                continue

            case_id = row.get("case_id") or f"row-{len(cases)}"

            if case_id not in cases:
                cases[case_id] = [_vector(row), {}]

            cases[case_id][1][row["algorithm"]] = runtime

        examples = [
            tuple(case)
            for case in cases.values()
        ]

        if not examples:
            raise ValueError(
                "No algorithm benchmark rows available for training"
            )

        vectors = [
            example[0]
            for example in examples
        ]

        dimensions = list(zip(*vectors))

        self.centers = tuple(
            mean(column)
            for column in dimensions
        )

        self.scales = tuple(
            math.sqrt(
                mean(
                    (value - center) ** 2
                    for value in column
                )
            ) or 1.0
            for column, center in zip(
                dimensions,
                self.centers,
            )
        )

        self.models = {}

        for algorithm in {
            name
            for _, results in examples
            for name in results
        }:
            observations = [
                (
                    self._standardize(vector),
                    math.log(results[algorithm]),
                )
                for vector, results in examples
                if algorithm in results
            ]

            width = len(self.centers) + 1

            normal = [
                [0.0] * width
                for _ in range(width)
            ]

            target = [0.0] * width

            for vector, value in observations:
                design = [1.0, *vector]

                for i in range(width):
                    target[i] += design[i] * value

                    for j in range(width):
                        normal[i][j] += (
                            design[i] * design[j]
                        )

            for i in range(1, width):
                normal[i][i] += ridge

            self.models[algorithm] = _solve(
                normal,
                target,
            )

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        ridge: float = 1e-3,
    ):
        with Path(path).open(
            newline="",
            encoding="utf-8",
        ) as file:
            return cls(
                list(csv.DictReader(file)),
                ridge=ridge,
            )

    def _standardize(
        self,
        vector: tuple[float, ...],
    ) -> tuple[float, ...]:
        return tuple(
            (value - center) / scale
            for value, center, scale in zip(
                vector,
                self.centers,
                self.scales,
            )
        )

    def predict_runtimes_from_features(
        self,
        features: dict,
    ) -> dict[str, float]:

        vector = [
            1.0,
            *self._standardize(_vector(features)),
        ]

        return {
            algorithm: math.exp(
                max(
                    -700,
                    min(
                        700,
                        sum(
                            a * b
                            for a, b in zip(
                                coefficients,
                                vector,
                            )
                        ),
                    ),
                )
            )
            for algorithm, coefficients in self.models.items()
        }

    def predict_runtimes(
        self,
        graph,
        features: dict | None = None,
    ) -> dict[str, float]:

        if features is None:
            features = extract_selection_features(graph)

        return self.predict_runtimes_from_features(features)

    def choose_algorithm(
        self,
        graph,
        source: str,
        destination: str,
    ) -> str:
        """
        Select the predicted fastest eligible algorithm.

        Algorithms without trained runtime models are excluded
        from prediction-based selection.
        """

        features = extract_selection_features(graph)

        eligible = eligible_algorithms(
            graph,
            source,
            destination,
            features=features,
        )

        if not eligible:
            raise ValueError(
                "No valid shortest-path algorithm available."
            )

        predictions = self.predict_runtimes(
            graph,
            features=features,
        )

        available = [
            algorithm
            for algorithm in eligible
            if algorithm in predictions
            and math.isfinite(predictions[algorithm])
            and predictions[algorithm] > 0
        ]

        if available:
            return min(
                available,
                key=lambda algorithm: predictions[algorithm],
            )

        # No eligible algorithm has a trained model.
        # Use the deterministic rule-based fallback.
        return choose_algorithm(
            graph,
            source,
            destination,
        )
