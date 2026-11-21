
import csv
import json
import math
import statistics
from pathlib import Path
from time import perf_counter

from algorithms import astar, bfs, dijkstra, bellman_ford
from adaptive.features import extract_features
from adaptive.selector import choose_algorithm, find_shortest_path
from analysis.metrics import measure_algorithm


ALGORITHM_IMPLEMENTATIONS = {
    "bfs": bfs.shortest_path,
    "dijkstra": dijkstra.shortest_path,
    "astar": astar.shortest_path,
    "bellman_ford": bellman_ford.shortest_path,
}


def _costs_match(first, second) -> bool:
    if first is None or second is None:
        return first is second

    return math.isclose(
        first,
        second,
        rel_tol=1e-9,
        abs_tol=1e-9,
    )


def _median(values):
    return statistics.median(values) if values else ""


def _runtime_statistics(durations):
    return {
        "median_seconds": statistics.median(durations),
        "minimum_seconds": min(durations),
        "maximum_seconds": max(durations),
        "runtime_stddev_seconds": (
            statistics.stdev(durations) if len(durations) > 1 else 0.0
        ),
    }


def _record_counters(result, counters):
    explored = result.get("explored_nodes")

    if explored is not None:
        counters["explored_nodes"].append(explored)

    counters["edge_scans"].append(result.get("edge_scans", 0))
    counters["relaxation_attempts"].append(
        result.get("relaxation_attempts", 0)
    )


def _counter_statistics(counters):
    return {
        "median_explored_nodes": _median(counters["explored_nodes"]),
        "median_edge_scans": _median(counters["edge_scans"]),
        "median_relaxation_attempts": _median(
            counters["relaxation_attempts"]
        ),
    }


def _new_counters():
    return {
        "explored_nodes": [],
        "edge_scans": [],
        "relaxation_attempts": [],
    }


def benchmark_graph(
    graph,
    source: str,
    destination: str,
    repeats: int = 7,
    performance_model=None,
    metadata: dict | None = None,
) -> list[dict]:

    if repeats < 1:
        raise ValueError("repeats must be at least 1")

    # Extract graph characteristics.
    features = extract_features(graph)

    # Identify eligible algorithms.
    algorithms = {}

    if graph.has_uniform_nonnegative_weights():
        algorithms["bfs"] = bfs.shortest_path

    if not graph.has_negative_weights():
        algorithms["dijkstra"] = dijkstra.shortest_path

    if (
        not graph.has_negative_weights()
        and graph.has_positions_for_all_nodes()
    ):
        algorithms["astar"] = astar.shortest_path

    algorithms["bellman_ford"] = bellman_ford.shortest_path

    rows = []
    expected_cost = None
    have_reference = False

    # --------------------------------------------------
    # 1. Benchmark individual algorithms
    # --------------------------------------------------

    for name, run_algorithm in algorithms.items():

        # Warm-up and correctness check.
        warmup_result = run_algorithm(
            graph, source, destination
        )

        if not have_reference:
            expected_cost = warmup_result["cost"]
            have_reference = True

        elif not _costs_match(
            expected_cost, warmup_result["cost"]
        ):
            raise AssertionError(
                f"{name} returned cost {warmup_result['cost']}; "
                f"expected {expected_cost}"
            )

        durations = []
        counters = _new_counters()
        final_result = warmup_result

        for _ in range(repeats):

            start = perf_counter()

            result = run_algorithm(
                graph, source, destination
            )

            durations.append(perf_counter() - start)

            if not _costs_match(
                expected_cost, result["cost"]
            ):
                raise AssertionError(
                    f"{name} returned inconsistent path cost "
                    f"{result['cost']}; expected {expected_cost}"
                )

            _record_counters(result, counters)
            final_result = result

        memory_result = measure_algorithm(
            run_algorithm,
            graph,
            source,
            destination,
        )

        row = {
            "algorithm": name,
            "selected_algorithm": "",
            "nodes": features["node_count"],
            "edges": features["edge_count"],
            "density": features["density"],
            "directed": features["directed"],
            "all_unit_weights": features["all_unit_weights"],
            "uniform_weights": features["uniform_weights"],
            "average_degree": features["average_degree"],
            "weight_stddev": features["weight_stddev"],
            "positions_available": features["positions_available"],
            "heuristic_scale": features["heuristic_scale"],
            "has_negative_weights": features["has_negative_weights"],
            "source": source,
            "destination": destination,
            "path_cost": final_result["cost"],
            "path": json.dumps(final_result["path"]),
            **_runtime_statistics(durations),
            **_counter_statistics(counters),
            "peak_memory_bytes": memory_result[1][
                "peak_memory_bytes"
            ],
            "repeats": repeats,
        }

        row.update(metadata or {})
        rows.append(row)

    # --------------------------------------------------
    # 2. Benchmark adaptive selection
    # --------------------------------------------------

    adaptive_function = lambda: find_shortest_path(
        graph,
        source,
        destination,
        performance_model=performance_model,
    )

    # Warm-up adaptive execution.
    warmup_adaptive = adaptive_function()

    if not _costs_match(
        expected_cost, warmup_adaptive["cost"]
    ):
        raise AssertionError(
            "Adaptive selector returned an incorrect path cost."
        )

    adaptive_durations = []
    adaptive_counters = _new_counters()
    adaptive_result = warmup_adaptive

    for _ in range(repeats):

        start = perf_counter()
        result = adaptive_function()
        elapsed = perf_counter() - start

        adaptive_durations.append(elapsed)

        if not _costs_match(
            expected_cost, result["cost"]
        ):
            raise AssertionError(
                f"Adaptive selector returned cost {result['cost']}; "
                f"expected {expected_cost}"
            )

        if result["algorithm"] != warmup_adaptive["algorithm"]:
            raise AssertionError(
                "Adaptive selector changed its algorithm "
                "between repeated executions."
            )

        _record_counters(result, adaptive_counters)
        adaptive_result = result

    selected_algorithm = adaptive_result["algorithm"]
    selected_function = ALGORITHM_IMPLEMENTATIONS[
        selected_algorithm
    ]

    # --------------------------------------------------
    # 3. Measure feature extraction separately
    # --------------------------------------------------

    feature_durations = []

    for _ in range(repeats):
        start = perf_counter()
        extract_features(graph)
        feature_durations.append(perf_counter() - start)

    # --------------------------------------------------
    # 4. Measure selection decision separately
    # --------------------------------------------------

    selection_durations = []

    for _ in range(repeats):

        start = perf_counter()

        if performance_model is None:
            selected = choose_algorithm(
                graph, source, destination
            )
        else:
            selected = performance_model.choose_algorithm(
                graph, source, destination
            )

        selection_durations.append(perf_counter() - start)

        if selected != selected_algorithm:
            raise AssertionError(
                "Standalone selection and adaptive execution "
                "produced different algorithm choices."
            )

    # --------------------------------------------------
    # 5. Measure selected algorithm execution separately
    # --------------------------------------------------

    selected_algorithm_durations = []

    for _ in range(repeats):

        start = perf_counter()

        result = selected_function(
            graph, source, destination
        )

        selected_algorithm_durations.append(
            perf_counter() - start
        )

        if not _costs_match(
            expected_cost, result["cost"]
        ):
            raise AssertionError(
                "Selected algorithm returned an incorrect path cost."
            )

    # Estimate selection overhead by comparing the median
    # end-to-end adaptive time with standalone algorithm time.
    estimated_selection_overhead = (
        statistics.median(adaptive_durations)
        - statistics.median(selected_algorithm_durations)
    )

    adaptive_memory = measure_algorithm(
        lambda g, s, d: find_shortest_path(
            g,
            s,
            d,
            performance_model=performance_model,
        ),
        graph,
        source,
        destination,
    )[1]["peak_memory_bytes"]

    # --------------------------------------------------
    # 6. Store adaptive benchmark results
    # --------------------------------------------------

    adaptive_row = {
        "algorithm": "adaptive",
        "selector_policy": (
            "ridge_regression"
            if performance_model
            else "rules"
        ),
        "ridge_alpha": (
            performance_model.ridge
            if performance_model
            else ""
        ),
        "selected_algorithm": selected_algorithm,
        "nodes": features["node_count"],
        "edges": features["edge_count"],
        "density": features["density"],
        "directed": features["directed"],
        "all_unit_weights": features["all_unit_weights"],
        "uniform_weights": features["uniform_weights"],
        "average_degree": features["average_degree"],
        "weight_stddev": features["weight_stddev"],
        "positions_available": features["positions_available"],
        "heuristic_scale": features["heuristic_scale"],
        "has_negative_weights": features["has_negative_weights"],
        "source": source,
        "destination": destination,
        "path_cost": adaptive_result["cost"],
        "path": json.dumps(adaptive_result["path"]),
        **_runtime_statistics(adaptive_durations),
        **_counter_statistics(adaptive_counters),
        "peak_memory_bytes": adaptive_memory,
        "median_feature_extraction_seconds": statistics.median(
            feature_durations
        ),
        "median_selection_seconds": statistics.median(
            selection_durations
        ),
        "median_selected_algorithm_seconds": statistics.median(
            selected_algorithm_durations
        ),
        "estimated_selection_overhead_seconds": (
            estimated_selection_overhead
        ),
        "repeats": repeats,
    }

    adaptive_row.update(metadata or {})
    rows.append(adaptive_row)

    return rows


def save_results_csv(rows: list[dict], output_path: str) -> None:

    if not rows:
        raise ValueError("No benchmark results to save")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    columns = list(
        dict.fromkeys(
            key
            for row in rows
            for key in row
        )
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=columns,
        )

        writer.writeheader()
        writer.writerows(rows)
