import csv
import json
import math
import statistics
from pathlib import Path
from time import perf_counter

from algorithms import astar, bfs, dijkstra, bellman_ford
from adaptive.features import extract_features
from adaptive.selector import find_shortest_path


def _costs_match(first, second) -> bool:
    if first is None or second is None:
        return first is second
    return math.isclose(first, second, rel_tol=1e-9, abs_tol=1e-9)


def benchmark_graph(graph, source: str, destination: str, repeats: int = 7) -> list[dict]:
    if repeats < 1:
        raise ValueError("repeats must be at least 1")

    features = extract_features(graph)
    algorithms = {}

    if graph.has_uniform_nonnegative_weights():
        algorithms["bfs"] = bfs.shortest_path

    if not graph.has_negative_weights():
        algorithms["dijkstra"] = dijkstra.shortest_path
        
    if not graph.has_negative_weights() and graph.has_positions_for_all_nodes():
        algorithms["astar"] = astar.shortest_path

    algorithms["bellman_ford"] = bellman_ford.shortest_path

    rows = []
    expected_cost = None
    have_reference = False

    for name, run_algorithm in algorithms.items():
        # Warm up and use this result to check correctness.
        warmup_result = run_algorithm(graph, source, destination)

        if not have_reference:
            expected_cost = warmup_result["cost"]
            have_reference = True
        elif not _costs_match(expected_cost, warmup_result["cost"]):
            raise AssertionError(
                f"{name} returned cost {warmup_result['cost']}; "
                f"expected {expected_cost}"
            )

        durations = []
        explored_counts = []
        edge_scan_counts = []
        relaxation_counts = []
        final_result = warmup_result

        for _ in range(repeats):
            start = perf_counter()
            result = run_algorithm(graph, source, destination)
            durations.append(perf_counter() - start)
            if result.get("explored_nodes") is not None:
                explored_counts.append(result["explored_nodes"])
            edge_scan_counts.append(result.get("edge_scans", 0))
            relaxation_counts.append(result.get("relaxation_attempts", 0))
            final_result = result

            if not _costs_match(expected_cost, result["cost"]):
                raise AssertionError(
                    f"{name} returned inconsistent path cost {result['cost']}; "
                    f"expected {expected_cost}"
                )

        rows.append({
            "algorithm": name,
            "selected_algorithm": "",
            "nodes": features["node_count"],
            "edges": features["edge_count"],
            "density": features["density"],
            "directed": features["directed"],
            "all_unit_weights": features["all_unit_weights"],
            "uniform_weights": features["uniform_weights"],
            "has_negative_weights": features["has_negative_weights"],
            "source": source,
            "destination": destination,
            "path_cost": final_result["cost"],
            "path": json.dumps(final_result["path"]),
            "median_seconds": statistics.median(durations),
            "median_explored_nodes": (
                statistics.median(explored_counts) if explored_counts else ""
            ),
            "median_edge_scans": statistics.median(edge_scan_counts),
            "median_relaxation_attempts": statistics.median(relaxation_counts),
            "repeats": repeats,
        })

    adaptive_durations = []
    adaptive_results = []
    for _ in range(repeats):
        start = perf_counter()
        adaptive_result = find_shortest_path(graph, source, destination)
        adaptive_durations.append(perf_counter() - start)
        adaptive_results.append(adaptive_result)
    adaptive_result = adaptive_results[-1]
    if not _costs_match(expected_cost, adaptive_result["cost"]):
        raise AssertionError(
            f"Adaptive selector returned cost {adaptive_result['cost']}; "
            f"expected {expected_cost}"
        )

    rows.append({
        "algorithm": "adaptive",
        "selected_algorithm": adaptive_result["algorithm"],
        "nodes": features["node_count"],
        "edges": features["edge_count"],
        "density": features["density"],
        "directed": features["directed"],
        "all_unit_weights": features["all_unit_weights"],
        "uniform_weights": features["uniform_weights"],
        "has_negative_weights": features["has_negative_weights"],
        "source": source,
        "destination": destination,
        "path_cost": adaptive_result["cost"],
        "path": json.dumps(adaptive_result["path"]),
        "median_seconds": statistics.median(adaptive_durations),
        "median_explored_nodes": adaptive_result["explored_nodes"],
        "median_edge_scans": adaptive_result.get("edge_scans", 0),
        "median_relaxation_attempts": adaptive_result.get("relaxation_attempts", 0),
        "repeats": repeats,
    })

    return rows


def save_results_csv(rows: list[dict], output_path: str) -> None:
    if not rows:
        raise ValueError("No benchmark results to save")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    columns = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
