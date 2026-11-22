"""Evaluate the frozen synthetic-trained selector on DIMACS road graphs.

The evaluator downloads the public-domain DIMACS Challenge 9 distance graphs,
keeps real edge weights and coordinates, and builds a bounded connected induced
sample for the repository's pure-Python algorithms. Full-network statistics
remain recorded separately from each evaluated sample.
"""

import csv
import hashlib
import gzip
import math
import random
import statistics
from time import perf_counter
import urllib.request
from collections import defaultdict, deque
from pathlib import Path

from adaptive.features import extract_selection_features
from adaptive.performance_model import BenchmarkRuntimeSelector
from adaptive.selector import choose_algorithm, find_shortest_path
from analysis.benchmark import benchmark_graph, _costs_match
from graph.graph import Graph


BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR.parent
RESULTS_DIR = BACKEND_DIR / "results"
DATA_DIR = PROJECT_ROOT / "data" / "real_world" / "dimacs9"
DEVELOPMENT_PATH = RESULTS_DIR / "development_benchmarks.csv"
# The DIMACS host currently returns 404 for its data-file URLs. These public
# mirrors identify the original Challenge 9 source and graph dimensions. Keep
# each mirror URL in the manifest for auditability.
NETWORKS = {
    "DIMACS9-NY-distance": {
        "name": "New York City",
        "nodes": 264346,
        "arcs": 733846,
        "graph_file": "USA-road-d.NY.gr",
        "graph_url": "https://raw.githubusercontent.com/blxrua/ads_project2_ylb/main/data/dimacs/USA-road-d.NY.gr",
        "coordinate_file": "USA-road-d.NY.co",
        "coordinate_url": "https://raw.githubusercontent.com/arnauddesombre/Python-Utilities/master/USA-road-d.NY.co",
        "weight_type": "physical road distance, original DIMACS weights",
    },
    "DIMACS9-COL-distance": {
        "name": "Colorado",
        "nodes": 435666,
        "arcs": 1057066,
        "graph_file": "USA-road-d.COL.gr",
        "graph_url": "https://raw.githubusercontent.com/blxrua/ads_project2_ylb/main/data/dimacs/USA-road-d.COL.gr",
        "coordinate_file": "USA-road-d.COL.co",
        "coordinate_url": "https://raw.githubusercontent.com/wmsnp/cch-rs/master/tests/data/USA-road-d.COL.co",
        "weight_type": "physical road distance, original DIMACS weights",
    },
    "DIMACS9-COL-travel-time": {
        "name": "Colorado",
        "nodes": 435666,
        "arcs": 1057066,
        "graph_file": "USA-road-t.COL.gr",
        "graph_url": "https://raw.githubusercontent.com/wmsnp/cch-rs/master/tests/data/USA-road-t.COL.gr",
        "coordinate_file": "USA-road-d.COL.co",
        "coordinate_url": "https://raw.githubusercontent.com/wmsnp/cch-rs/master/tests/data/USA-road-d.COL.co",
        "weight_type": "transit time, original DIMACS weights",
    },
    "DIMACS9-FLA-distance": {
        "name": "Florida",
        "nodes": 1070376,
        "arcs": 2712798,
        "graph_file": "USA-road-d.FLA.gr",
        "graph_url": "https://raw.githubusercontent.com/blxrua/ads_project2_ylb/main/data/dimacs/USA-road-d.FLA.gr",
        "coordinate_file": None,
        "coordinate_url": None,
        "weight_type": "physical road distance, original DIMACS weights",
    },
    "DIMACS9-Rome99-distance": {
        "name": "Rome (1999)",
        "nodes": 3353,
        "arcs": 8870,
        "graph_file": "rome99.gr",
        "graph_url": "https://raw.githubusercontent.com/patrickenfuego/rome99-graph-traversal/master/rome99.gr",
        "coordinate_file": None,
        "coordinate_url": None,
        "weight_type": "physical road distance in meters, original DIMACS weights",
    },
}
MAX_SCAN_NODE_ID = 20000
MAX_SAMPLE_NODES = 200
QUERY_PAIRS_PER_DATASET = 20
BENCHMARK_REPEATS = 3
QUERY_SEED = 26001


def _download_if_missing(path: Path, url: str) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {path.name}")
    urllib.request.urlretrieve(url, path)


def _ensure_files(dataset: dict) -> tuple[Path, Path | None]:
    graph_path = DATA_DIR / dataset["graph_file"]
    _download_if_missing(graph_path, dataset["graph_url"])
    coordinate_path = None
    if dataset["coordinate_file"]:
        coordinate_path = DATA_DIR / dataset["coordinate_file"]
        _download_if_missing(coordinate_path, dataset["coordinate_url"])
    return graph_path, coordinate_path


def _open_text(path: Path):
    return gzip.open(path, "rt", encoding="ascii") if path.suffix == ".gz" else path.open(encoding="ascii")


def load_bounded_dimacs_graph(
    graph_path: Path,
    coordinate_path: Path | None,
    *,
    max_scan_node_id: int = MAX_SCAN_NODE_ID,
    max_nodes: int = MAX_SAMPLE_NODES,
) -> tuple[Graph, dict]:
    """Load an induced connected sample while retaining original arc weights."""
    declared_nodes = declared_arcs = None
    arcs = []
    candidate_nodes = set()
    weak = defaultdict(set)

    with _open_text(graph_path) as file:
        for line in file:
            fields = line.split()
            if not fields or fields[0] == "c":
                continue
            if fields[0] == "p" and len(fields) >= 4:
                declared_nodes, declared_arcs = int(fields[2]), int(fields[3])
            elif fields[0] == "a" and len(fields) >= 4:
                source, destination, weight = int(fields[1]), int(fields[2]), float(fields[3])
                if source <= max_scan_node_id and destination <= max_scan_node_id:
                    arcs.append((source, destination, weight))
                    candidate_nodes.update((source, destination))
                    weak[source].add(destination)
                    weak[destination].add(source)

    if declared_nodes is None or declared_arcs is None:
        raise ValueError(f"Missing DIMACS problem header in {graph_path}")
    if not arcs:
        raise ValueError(f"No arcs found in scanned node range for {graph_path}")

    components = []
    visited = set()
    for node in candidate_nodes:
        if node in visited:
            continue
        component = []
        queue = [node]
        visited.add(node)
        for current in queue:
            component.append(current)
            for neighbor in weak[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        components.append(component)
    largest = max(components, key=len)
    largest_set = set(largest)

    # Pick a deterministic connected BFS prefix. The final graph is induced on
    # these vertices, so every retained arc and weight comes from the source.
    selected = []
    selected_set = set()
    queue = deque([min(largest)])
    selected_set.add(queue[0])
    while queue and len(selected) < max_nodes:
        current = queue.popleft()
        selected.append(current)
        for neighbor in sorted(weak[current]):
            if neighbor in largest_set and neighbor not in selected_set:
                selected_set.add(neighbor)
                queue.append(neighbor)
                if len(selected_set) >= max_nodes:
                    break

    selected_set = set(selected)
    raw_selected_arcs = [
        arc for arc in arcs if arc[0] in selected_set and arc[1] in selected_set
    ]
    # Graph.add_edge rejects parallel arcs. For nonnegative shortest paths,
    # keeping only the minimum-weight parallel arc preserves every shortest
    # path cost and the supported simple-graph behavior.
    minimum_arc_weight = {}
    for source, destination, weight in raw_selected_arcs:
        key = (source, destination)
        minimum_arc_weight[key] = min(
            weight, minimum_arc_weight.get(key, float("inf"))
        )
    selected_arcs = [
        (source, destination, weight)
        for (source, destination), weight in minimum_arc_weight.items()
    ]
    coordinate_values = {}
    if coordinate_path is not None:
        with _open_text(coordinate_path) as file:
            for line in file:
                fields = line.split()
                if not fields or fields[0] == "c":
                    continue
                if fields[0] == "v" and len(fields) >= 4:
                    node = int(fields[1])
                    if node in selected_set:
                        coordinate_values[node] = (float(fields[2]), float(fields[3]))

    graph = Graph(directed=True)
    for node in selected:
        graph.add_node(str(node))
        if node in coordinate_values:
            x, y = coordinate_values[node]
            graph.set_position(str(node), x, y)
    for source, destination, weight in selected_arcs:
        graph.add_edge(str(source), str(destination), weight)

    metadata = {
        "full_node_count": declared_nodes,
        "full_arc_count": declared_arcs,
        "scanned_node_id_limit": max_scan_node_id,
        "sample_node_count": len(graph.adjacency),
        "sample_arc_count": graph.edge_count,
        "raw_induced_arc_count": len(raw_selected_arcs),
        "parallel_arcs_coalesced": len(raw_selected_arcs) - len(selected_arcs),
        "parallel_arc_policy": "minimum weight per ordered endpoint pair; preserves shortest-path costs for nonnegative weights",
        "sample_weak_component_node_count": len(largest),
        "coordinate_node_count": len(coordinate_values),
        "sample_method": "largest weak component among arcs with both IDs <= scan limit; deterministic BFS prefix; induced arcs",
        "weight_method": "original DIMACS distance arc weights, unmodified",
    }
    if coordinate_path is not None and not graph.has_positions_for_all_nodes():
        raise ValueError("DIMACS coordinate file lacks positions for sampled nodes")
    return graph, metadata


def _reachable_by_hops(graph: Graph, source: str) -> list[tuple[str, int]]:
    distances = {source: 0}
    queue = deque([source])
    while queue:
        current = queue.popleft()
        for neighbor, _ in graph.neighbors(current):
            if neighbor not in distances:
                distances[neighbor] = distances[current] + 1
                queue.append(neighbor)
    return sorted(distances.items(), key=lambda pair: (pair[1], pair[0]))


def make_queries(graph: Graph, count: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    nodes = sorted(graph.adjacency)
    queries = []
    used = set()
    attempts = 0
    while len(queries) < count and attempts < count * 50:
        attempts += 1
        source = rng.choice(nodes)
        reachable = _reachable_by_hops(graph, source)
        if len(reachable) < 3:
            continue
        # Alternate a nearby and distant reachable endpoint, as measured by
        # unweighted hop count, without replacing the dataset's edge weights.
        target_index = max(1, len(reachable) // 4) if len(queries) % 2 == 0 else len(reachable) - 1
        destination = reachable[target_index][0]
        if source == destination or (source, destination) in used:
            continue
        used.add((source, destination))
        queries.append({
            "source": source,
            "destination": destination,
            "endpoint_class": "near" if len(queries) % 2 == 0 else "far",
            "hop_distance": reachable[target_index][1],
        })
    return queries


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path | None) -> str:
    if path is None:
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _benchmark_rule_selector(graph, source, destination, expected_cost):
    """Measure rules on the exact same graph/query as the Ridge run."""
    selected = choose_algorithm(graph, source, destination)
    warm = find_shortest_path(graph, source, destination)
    if not _costs_match(expected_cost, warm["cost"]):
        raise AssertionError("Rule selector returned an incorrect path cost")

    decision_times = []
    end_to_end_times = []
    final = warm
    for _ in range(BENCHMARK_REPEATS):
        started = perf_counter()
        decision = choose_algorithm(graph, source, destination)
        decision_times.append(perf_counter() - started)
        if decision != selected:
            raise AssertionError("Rule selector changed choice between repeats")

        started = perf_counter()
        final = find_shortest_path(graph, source, destination)
        end_to_end_times.append(perf_counter() - started)
        if not _costs_match(expected_cost, final["cost"]):
            raise AssertionError("Rule selector path cost did not match baseline")

    return {
        "algorithm": "adaptive",
        "selector_policy": "rules",
        "selected_algorithm": selected,
        "median_seconds": statistics.median(end_to_end_times),
        "median_selection_seconds": statistics.median(decision_times),
        "median_selected_algorithm_seconds": "",
        "estimated_selection_overhead_seconds": "",
        "path_cost": final["cost"],
        "correctness_verified": True,
        "repeats": BENCHMARK_REPEATS,
    }


def run() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model = BenchmarkRuntimeSelector.from_csv(DEVELOPMENT_PATH)
    manifest, benchmark_rows, prediction_rows, query_rows = [], [], [], []

    for network_index, (dataset_id, dataset) in enumerate(NETWORKS.items()):
        graph_path, coordinate_path = _ensure_files(dataset)
        graph, sample_metadata = load_bounded_dimacs_graph(graph_path, coordinate_path)
        manifest.append({
            "dataset_id": dataset_id,
            "name": dataset["name"],
            "source": "DIMACS 9th Implementation Challenge, USA road networks",
            "source_url": "https://www.diag.uniroma1.it/~challenge9/download.shtml",
            "license_status": "Challenge page states files are public domain unless otherwise specified",
            "source_graph_file": graph_path.name,
            "source_coordinate_file": coordinate_path.name if coordinate_path else "not available in verified mirror",
            "mirror_graph_url": dataset["graph_url"],
            "mirror_coordinate_url": dataset["coordinate_url"] or "",
            "graph_sha256": _sha256(graph_path),
            "coordinate_sha256": _sha256(coordinate_path),
            "weight_type": dataset["weight_type"],
            "directed": True,
            "coordinates": graph.has_positions_for_all_nodes(),
            "full_node_count_reference": dataset["nodes"],
            "full_arc_count_reference": dataset["arcs"],
            **sample_metadata,
        })

        queries = make_queries(graph, QUERY_PAIRS_PER_DATASET, QUERY_SEED + network_index)
        if len(queries) < QUERY_PAIRS_PER_DATASET:
            raise ValueError(f"Only made {len(queries)} query pairs for {dataset_id}")
        features = extract_selection_features(graph)
        predictions = model.predict_runtimes_from_features(features)

        for query_index, query in enumerate(queries):
            case_id = f"{dataset_id}-q{query_index + 1:02d}"
            metadata = {
                "dataset_id": dataset_id,
                "case_id": case_id,
                "query_seed": QUERY_SEED + network_index,
                "endpoint_class": query["endpoint_class"],
                "hop_distance": query["hop_distance"],
                "source_node": query["source"],
                "destination_node": query["destination"],
                "data_sample_nodes": len(graph.adjacency),
            }
            rows = benchmark_graph(
                graph,
                query["source"],
                query["destination"],
                repeats=BENCHMARK_REPEATS,
                performance_model=model,
                metadata=metadata,
            )
            adaptive = next(row for row in rows if row["algorithm"] == "adaptive")
            algorithms = [row for row in rows if row["algorithm"] != "adaptive"]
            fastest = min(float(row["median_seconds"]) for row in algorithms)
            winners = [row["algorithm"] for row in algorithms if math.isclose(
                float(row["median_seconds"]), fastest, rel_tol=1e-9, abs_tol=1e-12
            )]
            selected_row = next(row for row in algorithms if row["algorithm"] == adaptive["selected_algorithm"])
            adaptive_runtime = float(adaptive["median_seconds"])
            rule_adaptive = _benchmark_rule_selector(
                graph, query["source"], query["destination"], adaptive["path_cost"]
            )
            for policy, chosen in (("ridge_regression", adaptive), ("rules", rule_adaptive)):
                chosen_algorithm = chosen["selected_algorithm"]
                chosen_row = next(row for row in algorithms if row["algorithm"] == chosen_algorithm)
                chosen_runtime = float(chosen_row["median_seconds"])
                decision_overhead = (
                    float(chosen["median_seconds"]) - chosen_runtime
                    if chosen["median_seconds"] != "" else ""
                )
                query_rows.append({
                    **metadata,
                    "selector_policy": policy,
                    "selected_algorithm": chosen_algorithm,
                    "oracle_algorithms": ";".join(winners),
                    "winner_correct": chosen_algorithm in winners,
                    "winner_tie_count": len(winners),
                    "eligible_algorithm_count": len(algorithms),
                    "selected_algorithm_runtime_seconds": chosen_runtime,
                    "oracle_runtime_seconds": fastest,
                    "selection_decision_seconds": chosen["median_selection_seconds"],
                    "adaptive_runtime_seconds": chosen["median_seconds"],
                    "estimated_selection_overhead_seconds": decision_overhead,
                    "selected_to_oracle_ratio": chosen_runtime / fastest,
                    "adaptive_to_oracle_ratio": float(chosen["median_seconds"]) / fastest,
                    "adaptive_runtime_regret_seconds": float(chosen["median_seconds"]) - fastest,
                    "correctness_verified": True,
                })
            for row in rows:
                row["dataset_id"] = dataset_id
                row["case_id"] = case_id
                row["correctness_verified"] = True
                benchmark_rows.append(row)
            rule_adaptive.update(metadata)
            rule_adaptive.update({
                "dataset_id": dataset_id,
                "case_id": case_id,
                "correctness_verified": True,
            })
            benchmark_rows.append(rule_adaptive)
            for algorithm, predicted_seconds in predictions.items():
                prediction_rows.append({
                    **metadata,
                    "algorithm": algorithm,
                    "predicted_runtime_seconds": predicted_seconds,
                    "eligible": algorithm in {row["algorithm"] for row in algorithms},
                })
        print(f"Evaluated {dataset_id}: {len(queries)} source/destination pairs")

    by_dataset = defaultdict(list)
    for row in query_rows:
        by_dataset[(row["dataset_id"], row["selector_policy"])].append(row)
    summary = []
    for (dataset_id, policy), rows in sorted(by_dataset.items()):
        summary.append({
            "dataset_id": dataset_id,
            "selector_policy": policy,
            "query_pairs": len(rows),
            "winner_accuracy": statistics.mean(row["winner_correct"] for row in rows),
            "nontrivial_winner_accuracy": statistics.mean(row["winner_correct"] for row in rows),
            "median_selected_to_oracle_ratio": statistics.median(row["selected_to_oracle_ratio"] for row in rows),
            "median_adaptive_to_oracle_ratio": statistics.median(row["adaptive_to_oracle_ratio"] for row in rows),
            "near_optimal_selected_rate_1_10": statistics.mean(row["selected_to_oracle_ratio"] <= 1.10 for row in rows),
            "near_optimal_adaptive_rate_1_10": statistics.mean(row["adaptive_to_oracle_ratio"] <= 1.10 for row in rows),
            "median_runtime_regret_seconds": statistics.median(row["adaptive_runtime_regret_seconds"] for row in rows),
            "all_path_costs_verified": all(row["correctness_verified"] for row in rows),
        })

    all_query_rows = query_rows
    for policy in ("ridge_regression", "rules"):
        rows = [row for row in all_query_rows if row["selector_policy"] == policy]
        summary.append({
            "dataset_id": "ALL_FIVE_GRAPH_INPUTS",
            "selector_policy": policy,
            "query_pairs": len(rows),
            "winner_accuracy": statistics.mean(row["winner_correct"] for row in rows),
            "nontrivial_winner_accuracy": statistics.mean(row["winner_correct"] for row in rows),
            "median_selected_to_oracle_ratio": statistics.median(row["selected_to_oracle_ratio"] for row in rows),
            "median_adaptive_to_oracle_ratio": statistics.median(row["adaptive_to_oracle_ratio"] for row in rows),
            "near_optimal_selected_rate_1_10": statistics.mean(row["selected_to_oracle_ratio"] <= 1.10 for row in rows),
            "near_optimal_adaptive_rate_1_10": statistics.mean(row["adaptive_to_oracle_ratio"] <= 1.10 for row in rows),
            "median_runtime_regret_seconds": statistics.median(row["adaptive_runtime_regret_seconds"] for row in rows),
            "all_path_costs_verified": all(row["correctness_verified"] for row in rows),
        })

    _write_csv(RESULTS_DIR / "real_world_dataset_manifest.csv", manifest)
    _write_csv(RESULTS_DIR / "real_world_benchmarks.csv", benchmark_rows)
    _write_csv(RESULTS_DIR / "real_world_predictions.csv", prediction_rows)
    _write_csv(RESULTS_DIR / "real_world_query_results.csv", query_rows)
    _write_csv(RESULTS_DIR / "real_world_summary.csv", summary)
    print(f"Saved real-world evaluation to {RESULTS_DIR}")


if __name__ == "__main__":
    run()
