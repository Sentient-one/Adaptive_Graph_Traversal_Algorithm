"""Measure cold/warm A* heuristic-scale cost and a narrow scan optimization.

Only writes heuristic_cache_profiling.csv. Existing benchmark and final
evaluation outputs are not read or overwritten.
"""

import csv
import math
import statistics
from pathlib import Path
from time import perf_counter_ns

from adaptive.features import extract_selection_features
from graph.generator import generate_random_graph
from graph.graph import Graph


BACKEND_DIR = Path(__file__).resolve().parents[1]
RESULTS_PATH = BACKEND_DIR / "results" / "heuristic_cache_profiling.csv"
SIZES = (100, 500, 1000, 2000)
DENSITIES = (0.002, 0.01, 0.03)
DIRECTIONS = (False, True)
SEED = 77123
PAIRED_REPEATS = 51
CACHE_REPEATS = 101


def _baseline_scan(graph):
    scale = float("inf")
    for start, neighbors in graph.adjacency.items():
        x1, y1 = graph.positions[start]
        for end, weight in neighbors:
            x2, y2 = graph.positions[end]
            distance = math.hypot(x2 - x1, y2 - y1)
            if distance > 0:
                scale = min(scale, weight / distance)
    return 0.0 if scale == float("inf") else scale


def _local_alias_scan(graph):
    """Candidate: bind stable mappings and hypot once for the O(E) scan."""
    scale = float("inf")
    positions = graph.positions
    hypot = math.hypot
    for start, neighbors in graph.adjacency.items():
        x1, y1 = positions[start]
        for end, weight in neighbors:
            x2, y2 = positions[end]
            distance = hypot(x2 - x1, y2 - y1)
            if distance > 0:
                scale = min(scale, weight / distance)
    return 0.0 if scale == float("inf") else scale


def _timed(function, graph):
    started = perf_counter_ns()
    result = function(graph)
    return perf_counter_ns() - started, result


def _copy_uncached(graph):
    fresh = Graph(directed=graph.directed)
    fresh.adjacency = {
        node: list(neighbors) for node, neighbors in graph.adjacency.items()
    }
    fresh.positions = graph.positions.copy()
    fresh._position_count = graph._position_count
    fresh._edge_count = graph._edge_count
    fresh._weight_count = graph._weight_count
    fresh._minimum_weight = graph._minimum_weight
    fresh._maximum_weight = graph._maximum_weight
    fresh._mean_weight = graph._mean_weight
    fresh._weight_m2 = graph._weight_m2
    fresh._unique_weights = graph._unique_weights.copy()
    fresh._all_unit_weights = graph._all_unit_weights
    fresh._has_negative_weights = graph._has_negative_weights
    fresh._heuristic_scale_cache = None
    fresh._heuristic_scale_dirty = True
    return fresh


def _paired_scan(graph):
    baseline, candidate = [], []
    for repeat in range(PAIRED_REPEATS):
        order = ((_baseline_scan, baseline), (_local_alias_scan, candidate))
        if repeat % 2:
            order = tuple(reversed(order))
        outputs = []
        for function, samples in order:
            elapsed, value = _timed(function, graph)
            samples.append(elapsed)
            outputs.append(value)
        if not math.isclose(outputs[0], outputs[1], rel_tol=1e-13, abs_tol=1e-15):
            raise AssertionError("Candidate heuristic scan changed its value")
    return statistics.median(baseline), statistics.median(candidate)


def _cold_public_scale(graph):
    samples = []
    expected = _baseline_scan(graph)
    for _ in range(CACHE_REPEATS):
        fresh = _copy_uncached(graph)
        elapsed, value = _timed(lambda item: item.heuristic_scale(), fresh)
        if not math.isclose(value, expected, rel_tol=1e-13, abs_tol=1e-15):
            raise AssertionError("Public cold heuristic scale mismatch")
        samples.append(elapsed)
    return statistics.median(samples)


def _cache_transition(graph):
    fresh = _copy_uncached(graph)
    cold_ns, _ = _timed(lambda item: item.heuristic_scale(), fresh)
    warm = []
    for _ in range(CACHE_REPEATS):
        elapsed, _ = _timed(lambda item: item.heuristic_scale(), fresh)
        warm.append(elapsed)
    return cold_ns, statistics.median(warm)


def _feature_transition(graph):
    cold, warm = [], []
    for _ in range(CACHE_REPEATS):
        fresh = _copy_uncached(graph)
        elapsed, _ = _timed(extract_selection_features, fresh)
        cold.append(elapsed)
    fresh = _copy_uncached(graph)
    extract_selection_features(fresh)
    for _ in range(CACHE_REPEATS):
        elapsed, _ = _timed(extract_selection_features, fresh)
        warm.append(elapsed)
    return statistics.median(cold), statistics.median(warm)


def _write(rows):
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run():
    rows = []
    for directed in DIRECTIONS:
        for node_count in SIZES:
            for density in DENSITIES:
                graph = generate_random_graph(
                    node_count=node_count,
                    edge_probability=density,
                    weight_mode="positive",
                    directed=directed,
                    coordinates=True,
                    seed=SEED + node_count + int(density * 10000) + directed,
                )
                baseline_ns, candidate_ns = _paired_scan(graph)
                cold_public_ns = _cold_public_scale(graph)
                _, warm_public_ns = _cache_transition(graph)
                cold_feature_ns, warm_feature_ns = _feature_transition(graph)
                row = {
                    "nodes": node_count,
                    "directed": directed,
                    "density_parameter": density,
                    "stored_arcs": sum(map(len, graph.adjacency.values())),
                    "logical_edges": graph.edge_count,
                    "baseline_scan_median_ns": baseline_ns,
                    "local_alias_scan_median_ns": candidate_ns,
                    "candidate_change_percent": (candidate_ns / baseline_ns - 1) * 100,
                    "public_cold_scale_median_ns": cold_public_ns,
                    "public_warm_scale_median_ns": warm_public_ns,
                    "cold_selection_feature_median_ns": cold_feature_ns,
                    "warm_selection_feature_median_ns": warm_feature_ns,
                    "cold_samples": CACHE_REPEATS,
                    "paired_scan_repeats": PAIRED_REPEATS,
                    "warm_samples": CACHE_REPEATS,
                    "scale_equivalent": True,
                    "cache_invalidation_supported_by_existing_test": True,
                }
                rows.append(row)
                _write(rows)
                print(
                    f"Profiled n={node_count}, directed={directed}, "
                    f"density={density}, arcs={row['stored_arcs']}",
                    flush=True,
                )
    _write(rows)
    print(f"Saved {len(rows)} profile cases to {RESULTS_PATH}")


if __name__ == "__main__":
    run()
