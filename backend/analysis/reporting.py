
"""Summarize selector quality and end-to-end runtime from benchmark records."""

import csv
import statistics
from collections import defaultdict
from pathlib import Path


# A selected algorithm within 5% of the fastest measured algorithm
# is considered near-optimal.
NEAR_OPTIMAL_TOLERANCE = 1.05


def _to_float(value, default=None):
    """Safely convert benchmark values to floats."""
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _median(values):
    """Return the median of available numeric values."""
    valid_values = [
        value for value in values
        if value is not None
    ]

    if not valid_values:
        return None

    return statistics.median(valid_values)


def summarize_selection(rows: list[dict]) -> list[dict]:
    """
    Summarize adaptive selector performance.

    Measures:
    - Exact fastest-algorithm selection accuracy.
    - Near-optimal selection rate.
    - Selection regret.
    - Adaptive end-to-end runtime.
    - Adaptive runtime ratio and speedup.
    - Runtime variability.
    - Full and selection-specific feature extraction.
    - Selection, selected algorithm runtime, and estimated overhead.
    """

    # Group algorithm records by graph case.
    # Include split to avoid collisions between development and held-out cases.
    cases = defaultdict(list)

    for row in rows:
        case_id = row.get("case_id")

        if not case_id:
            continue

        split = row.get("split", "unspecified")
        cases[(split, case_id)].append(row)

    groups = defaultdict(list)

    for (split, case_id), case_rows in cases.items():

        baseline = [
            row for row in case_rows
            if row.get("algorithm") != "adaptive"
            and _to_float(row.get("median_seconds")) is not None
        ]

        adaptive = next(
            (
                row for row in case_rows
                if row.get("algorithm") == "adaptive"
            ),
            None,
        )

        if not baseline or adaptive is None:
            continue

        adaptive_time = _to_float(
            adaptive.get("median_seconds")
        )

        if adaptive_time is None or adaptive_time <= 0:
            continue

        # Find the fastest measured baseline algorithm.
        fastest = min(
            baseline,
            key=lambda row: float(row["median_seconds"]),
        )

        best_time = float(fastest["median_seconds"])

        if best_time <= 0:
            continue

        # Find the algorithm selected by the adaptive selector.
        selected_algorithm = adaptive.get("selected_algorithm")

        selected_row = next(
            (
                row for row in baseline
                if row.get("algorithm") == selected_algorithm
            ),
            None,
        )

        selected_time = (
            _to_float(selected_row.get("median_seconds"))
            if selected_row is not None
            else None
        )

        # Selection quality metrics.
        selected_best = (
            selected_algorithm == fastest.get("algorithm")
        )

        if selected_time is not None and selected_time > 0:
            selection_regret = (
                selected_time / best_time
            ) - 1.0

            selected_runtime_ratio = (
                selected_time / best_time
            )

            near_optimal = (
                selected_runtime_ratio <= NEAR_OPTIMAL_TOLERANCE
            )
        else:
            selection_regret = None
            selected_runtime_ratio = None
            near_optimal = False

        # End-to-end adaptive performance.
        runtime_ratio = adaptive_time / best_time

        speedup = best_time / adaptive_time

        # Adaptive runtime variability.
        runtime_stddev = _to_float(
            adaptive.get("runtime_stddev_seconds")
        )

        runtime_cv = (
            runtime_stddev / adaptive_time
            if runtime_stddev is not None
            else None
        )

        # Selection instrumentation.
        selection_overhead = _to_float(
            adaptive.get("estimated_selection_overhead_seconds")
        )

        # Full feature extraction.
        # Fall back to the legacy field for older benchmark CSV files.
        full_feature_extraction_time = _to_float(
            adaptive.get(
                "median_full_feature_extraction_seconds",
                adaptive.get("median_feature_extraction_seconds"),
            )
        )

        # Lightweight extraction used by the performance model.
        selection_feature_extraction_time = _to_float(
            adaptive.get("median_selection_feature_extraction_seconds")
        )

        selection_time = _to_float(
            adaptive.get("median_selection_seconds")
        )

        selected_algorithm_time = _to_float(
            adaptive.get("median_selected_algorithm_seconds")
        )

        observed_density = _to_float(
            adaptive.get("density"),
            default=0.0,
        )

        category = (
            split,
            adaptive.get("selector_policy", "rules"),
            adaptive.get("weight_mode", "unspecified"),
            adaptive.get("coordinates", "unspecified"),
            adaptive.get("directed", "unspecified"),
            adaptive.get("nodes", "unspecified"),
            adaptive.get(
                "target_density",
                adaptive.get("density", "unspecified"),
            ),
        )

        groups[category].append({
            "selected_best": selected_best,
            "near_optimal": near_optimal,
            "selection_regret": selection_regret,
            "selected_runtime_ratio": selected_runtime_ratio,
            "runtime_ratio": runtime_ratio,
            "speedup": speedup,
            "adaptive_faster": runtime_ratio < 1.0,
            "observed_density": observed_density,
            "runtime_stddev": runtime_stddev,
            "runtime_cv": runtime_cv,
            "selection_overhead": selection_overhead,
            "full_feature_extraction_time": full_feature_extraction_time,
            "selection_feature_extraction_time": (
                selection_feature_extraction_time
            ),
            "selection_time": selection_time,
            "selected_algorithm_time": selected_algorithm_time,
            "best_algorithm_time": best_time,
            "adaptive_time": adaptive_time,
        })

    summaries = []

    for category, values in sorted(groups.items()):

        (
            split,
            policy,
            weight_mode,
            coordinates,
            directed,
            nodes,
            target_density,
        ) = category

        count = len(values)

        ratios = [
            item["runtime_ratio"]
            for item in values
        ]

        regrets = [
            item["selection_regret"]
            for item in values
            if item["selection_regret"] is not None
        ]

        summaries.append({

            # Graph category information.
            "split": split,
            "selector_policy": policy,
            "weight_mode": weight_mode,
            "coordinates": coordinates,
            "directed": directed,
            "nodes": nodes,
            "target_density": target_density,

            # Dataset statistics.
            "graph_cases": count,

            "mean_observed_density": statistics.mean(
                item["observed_density"]
                for item in values
            ),

            # Algorithm selection quality.
            "selected_fastest_algorithm_rate": (
                sum(item["selected_best"] for item in values)
                / count
            ),

            "selection_within_5_percent_rate": (
                sum(item["near_optimal"] for item in values)
                / count
            ),

            "median_selection_regret": _median(regrets),

            "mean_selection_regret": (
                statistics.mean(regrets)
                if regrets else None
            ),

            "median_selected_algorithm_to_best_runtime_ratio": (
                _median([
                    item["selected_runtime_ratio"]
                    for item in values
                ])
            ),

            # End-to-end adaptive performance.
            "median_best_algorithm_runtime_seconds": _median([
                item["best_algorithm_time"]
                for item in values
            ]),

            "median_adaptive_runtime_seconds": _median([
                item["adaptive_time"]
                for item in values
            ]),

            "median_adaptive_to_best_runtime_ratio": (
                statistics.median(ratios)
            ),

            "mean_adaptive_to_best_runtime_ratio": (
                statistics.mean(ratios)
            ),

            "median_best_to_adaptive_speedup": _median([
                item["speedup"]
                for item in values
            ]),

            "adaptive_faster_than_best_rate": (
                sum(item["adaptive_faster"] for item in values)
                / count
            ),

            # Runtime variability.
            "median_adaptive_runtime_stddev_seconds": _median([
                item["runtime_stddev"]
                for item in values
            ]),

            "median_adaptive_runtime_cv": _median([
                item["runtime_cv"]
                for item in values
            ]),

            # Selection overhead and instrumentation.
            "median_full_feature_extraction_seconds": _median([
                item["full_feature_extraction_time"]
                for item in values
            ]),

            "median_selection_feature_extraction_seconds": _median([
                item["selection_feature_extraction_time"]
                for item in values
            ]),

            # Backward compatibility with older reports.
            "median_feature_extraction_seconds": _median([
                item["full_feature_extraction_time"]
                for item in values
            ]),

            "median_selection_seconds": _median([
                item["selection_time"]
                for item in values
            ]),

            "median_selected_algorithm_seconds": _median([
                item["selected_algorithm_time"]
                for item in values
            ]),

            "median_estimated_selection_overhead_seconds": _median([
                item["selection_overhead"]
                for item in values
            ]),
        })

    return summaries


def save_selection_summary(
    rows: list[dict],
    output_path: str | Path,
) -> None:

    summaries = summarize_selection(rows)

    if not summaries:
        raise ValueError(
            "No adaptive benchmark records to summarize"
        )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(summaries[0]),
        )

        writer.writeheader()
        writer.writerows(summaries)
