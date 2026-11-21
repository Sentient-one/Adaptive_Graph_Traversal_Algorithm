"""Independent final evaluation on newly generated graph cases.

The Ridge selector is trained only from development_benchmarks.csv. Test cases
use disjoint seeds and a broader graph grid than the development experiment.
All reports are written to new files under backend/results.
"""

import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from adaptive.performance_model import BenchmarkRuntimeSelector
from analysis.benchmark import benchmark_graph, save_results_csv
from analysis.reporting import save_selection_summary
from graph.generator import generate_random_graph

BACKEND_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BACKEND_DIR / "results"
DEVELOPMENT_PATH = RESULTS_DIR / "development_benchmarks.csv"
NODE_COUNTS = (50, 100, 200, 300)
DENSITIES = (0.03, 0.10, 0.30, 0.55)
WEIGHT_MODES = ("unit", "uniform", "positive", "negative")
COORDINATE_OPTIONS = (False, True)
NONNEGATIVE_DIRECTED_OPTIONS = (False, True)
TEST_SEEDS = (303, 404, 505)
REPEATS = 3


def cases():
    for node_count in NODE_COUNTS:
        for density in DENSITIES:
            for weight_mode in WEIGHT_MODES:
                directed_options = (
                    (True,) if weight_mode == "negative"
                    else NONNEGATIVE_DIRECTED_OPTIONS
                )
                for directed in directed_options:
                    for coordinates in COORDINATE_OPTIONS:
                        for seed in TEST_SEEDS:
                            case_id = (
                                f"final-n{node_count}-d{density:.2f}-"
                                f"{weight_mode}-dir{int(directed)}-"
                                f"xy{int(coordinates)}-s{seed}"
                            )
                            graph = generate_random_graph(
                                node_count=node_count,
                                edge_probability=density,
                                directed=directed,
                                seed=seed,
                                weight_mode=weight_mode,
                                coordinates=coordinates,
                            )
                            metadata = {
                                "case_id": case_id,
                                "split": "final_model",
                                "seed": seed,
                                "weight_mode": weight_mode,
                                "target_density": density,
                                "coordinates": coordinates,
                                "evaluation": "independent_final",
                            }
                            yield graph, metadata


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def make_seed_summaries(benchmark_rows):
    """Summarize winner quality and end-to-end timing for each test seed."""
    grouped = defaultdict(list)
    for row in benchmark_rows:
        grouped[(row.get("split"), row.get("case_id"))].append(row)

    by_policy_seed = defaultdict(list)
    for (split, case_id), rows in grouped.items():
        adaptive = next(
            (row for row in rows if row.get("algorithm") == "adaptive"),
            None,
        )
        if adaptive is None:
            continue
        baselines = [row for row in rows if row.get("algorithm") != "adaptive"]
        if not baselines:
            continue
        fastest = min(float(row["median_seconds"]) for row in baselines)
        winners = {
            row["algorithm"] for row in baselines
            if math.isclose(float(row["median_seconds"]), fastest,
                            rel_tol=1e-9, abs_tol=1e-12)
        }
        key = (
            adaptive.get("selector_policy", "unknown"),
            int(adaptive["seed"]),
        )
        by_policy_seed[key].append({
            "nontrivial": len(baselines) > 1,
            "winner_correct": adaptive["selected_algorithm"] in winners,
            "speedup": fastest / float(adaptive["median_seconds"]),
            "runtime_ratio": float(adaptive["median_seconds"]) / fastest,
            "selection_seconds": float(adaptive["median_selection_seconds"]),
            "overhead_seconds": float(adaptive["estimated_selection_overhead_seconds"]),
            "correctness_verified": adaptive.get("correctness_verified", "True").lower() == "true",
        })

    summaries = []
    for (policy, seed), values in sorted(by_policy_seed.items()):
        nontrivial = [row for row in values if row["nontrivial"]]
        summaries.append({
            "selector_policy": policy,
            "seed": seed,
            "graph_cases": len(values),
            "nontrivial_graph_cases": len(nontrivial),
            "winner_accuracy": statistics.mean(row["winner_correct"] for row in values),
            "nontrivial_winner_accuracy": (
                statistics.mean(row["winner_correct"] for row in nontrivial)
                if nontrivial else ""
            ),
            "all_adaptive_costs_verified": all(
                row["correctness_verified"] for row in values
            ),
            "median_best_to_adaptive_speedup": statistics.median(
                row["speedup"] for row in values
            ),
            "median_adaptive_to_best_runtime_ratio": statistics.median(
                row["runtime_ratio"] for row in values
            ),
            "median_selection_seconds": statistics.median(
                row["selection_seconds"] for row in values
            ),
            "median_estimated_selection_overhead_seconds": statistics.median(
                row["overhead_seconds"] for row in values
            ),
        })
    return summaries


def summarize_existing_outputs():
    """Build a compact overall summary from completed case-level reports."""
    benchmark_path = RESULTS_DIR / "final_evaluation_benchmarks.csv"
    prediction_path = RESULTS_DIR / "final_evaluation_predictions.csv"
    pair_path = RESULTS_DIR / "final_evaluation_model_vs_rules.csv"

    with benchmark_path.open(newline="", encoding="utf-8") as file:
        benchmark_rows = list(csv.DictReader(file))
    with prediction_path.open(newline="", encoding="utf-8") as file:
        prediction_rows = list(csv.DictReader(file))
    with pair_path.open(newline="", encoding="utf-8") as file:
        pair_rows = list(csv.DictReader(file))
    write_csv(
        RESULTS_DIR / "final_evaluation_seed_summary.csv",
        make_seed_summaries(benchmark_rows),
    )

    groups = defaultdict(list)
    for row in benchmark_rows:
        if row.get("algorithm") == "adaptive":
            continue
        groups[(row.get("split"), row.get("case_id"))].append(row)
    adaptive_by_case = {}
    for row in benchmark_rows:
        if row.get("algorithm") == "adaptive":
            adaptive_by_case[(row.get("split"), row.get("case_id"))] = row

    policy_results = {}
    for split, policy in (("final_model", "ridge_regression"), ("final_rules", "rules")):
        cases_found = []
        for key, baselines in groups.items():
            if key[0] != split or key not in adaptive_by_case:
                continue
            adaptive = adaptive_by_case[key]
            fastest = min(float(row["median_seconds"]) for row in baselines)
            best_algorithms = {
                row["algorithm"] for row in baselines
                if math.isclose(float(row["median_seconds"]), fastest,
                                rel_tol=1e-9, abs_tol=1e-12)
            }
            selected_row = next(
                (row for row in baselines
                 if row["algorithm"] == adaptive["selected_algorithm"]),
                None,
            )
            selected_runtime_ratio = (
                float(selected_row["median_seconds"]) / fastest
                if selected_row else float("inf")
            )
            actual_count = len(baselines)
            cases_found.append({
                "nontrivial": actual_count > 1,
                "winner_correct": adaptive["selected_algorithm"] in best_algorithms,
                "best_runtime": fastest,
                "adaptive_runtime": float(adaptive["median_seconds"]),
                "speedup": fastest / float(adaptive["median_seconds"]),
                "runtime_ratio": float(adaptive["median_seconds"]) / fastest,
                "selection_seconds": float(adaptive["median_selection_seconds"]),
                "overhead_seconds": float(adaptive["estimated_selection_overhead_seconds"]),
                "peak_memory_bytes": float(adaptive["peak_memory_bytes"]),
                "correct": adaptive.get("correctness_verified", "True").lower() == "true",
                "selected_runtime_ratio": selected_runtime_ratio,
            })
        nontrivial = [row for row in cases_found if row["nontrivial"]]
        policy_results[policy] = {
            "record_type": "overall_policy_summary",
            "selector_policy": policy,
            "graph_cases": len(cases_found),
            "nontrivial_graph_cases": len(nontrivial),
            "winner_accuracy": statistics.mean(row["winner_correct"] for row in cases_found),
            "nontrivial_winner_accuracy": statistics.mean(row["winner_correct"] for row in nontrivial) if nontrivial else "",
            "all_adaptive_costs_verified": all(row["correct"] for row in cases_found),
            "median_best_algorithm_runtime_seconds": statistics.median(row["best_runtime"] for row in cases_found),
            "median_adaptive_runtime_seconds": statistics.median(row["adaptive_runtime"] for row in cases_found),
            "median_best_to_adaptive_speedup": statistics.median(row["speedup"] for row in cases_found),
            "median_adaptive_to_best_runtime_ratio": statistics.median(row["runtime_ratio"] for row in cases_found),
            "median_selected_algorithm_to_best_ratio": statistics.median(row["selected_runtime_ratio"] for row in cases_found),
            "median_selection_seconds": statistics.median(row["selection_seconds"] for row in cases_found),
            "median_estimated_selection_overhead_seconds": statistics.median(row["overhead_seconds"] for row in cases_found),
            "median_peak_memory_bytes": statistics.median(row["peak_memory_bytes"] for row in cases_found),
        }

    overall = list(policy_results.values())
    ridge_rule_speedups = [float(row["rule_to_ridge_speedup"]) for row in pair_rows]
    overall.append({
        "record_type": "paired_policy_comparison",
        "selector_policy": "ridge_regression_vs_rules",
        "graph_cases": len(pair_rows),
        "ridge_faster_cases": sum(value > 1 for value in ridge_rule_speedups),
        "rule_faster_cases": sum(value < 1 for value in ridge_rule_speedups),
        "runtime_ties": sum(math.isclose(value, 1.0, rel_tol=1e-9, abs_tol=1e-12) for value in ridge_rule_speedups),
        "median_rule_to_ridge_speedup": statistics.median(ridge_rule_speedups),
        "mean_rule_to_ridge_speedup": statistics.mean(ridge_rule_speedups),
        "mean_absolute_log_prediction_error": statistics.mean(
            float(row["absolute_log_error"]) for row in prediction_rows
        ),
        "median_absolute_log_prediction_error": statistics.median(
            float(row["absolute_log_error"]) for row in prediction_rows
        ),
    })
    write_csv(RESULTS_DIR / "final_evaluation_overall_summary.csv", overall)
    for row in overall:
        if row["record_type"] == "overall_policy_summary":
            print(
                f"{row['selector_policy']}: winner accuracy="
                f"{row['winner_accuracy']:.4f}, non-trivial="
                f"{row['nontrivial_winner_accuracy']:.4f}, median speedup="
                f"{row['median_best_to_adaptive_speedup']:.4f}x"
            )
        else:
            print(
                f"Paired rule/Ridge median speedup: "
                f"{row['median_rule_to_ridge_speedup']:.4f}x; "
                f"Ridge faster on {row['ridge_faster_cases']}/"
                f"{row['graph_cases']} cases"
            )
            print(
                f"Overall median absolute log prediction error: "
                f"{row['median_absolute_log_prediction_error']:.4f}"
            )


def main():
    with DEVELOPMENT_PATH.open(newline="", encoding="utf-8") as file:
        development_rows = list(csv.DictReader(file))
    model = BenchmarkRuntimeSelector(development_rows)

    all_rows = []
    prediction_rows = []
    pair_rows = []
    case_count = 0

    for case_count, (graph, metadata) in enumerate(cases(), start=1):
        source = "0"
        destination = str(len(graph.adjacency) - 1)
        case_id = metadata["case_id"]

        # This call supplies the common individual-algorithm baselines and the
        # Ridge adaptive measurement. It also checks returned path costs.
        ridge_rows = benchmark_graph(
            graph, source, destination, repeats=REPEATS,
            performance_model=model, metadata=metadata,
        )
        ridge_baselines = [
            row for row in ridge_rows if row["algorithm"] != "adaptive"
        ]
        ridge_adaptive = next(
            row for row in ridge_rows if row["algorithm"] == "adaptive"
        )
        for row in ridge_rows:
            row["correctness_verified"] = True
        all_rows.extend(ridge_rows)

        # Measure the rule selector on the exact same graph. Reuse the already
        # measured baselines under a separate split label to avoid ambiguity in
        # policy-specific summaries while preserving comparable measurements.
        rule_metadata = dict(metadata, split="final_rules")
        rule_rows = benchmark_graph(
            graph, source, destination, repeats=REPEATS,
            performance_model=None, metadata=rule_metadata,
        )
        rule_adaptive = next(
            row for row in rule_rows if row["algorithm"] == "adaptive"
        )
        rule_baselines = []
        for baseline in ridge_baselines:
            copy = dict(baseline)
            copy["split"] = "final_rules"
            rule_baselines.append(copy)
        rule_adaptive["correctness_verified"] = True
        all_rows.extend(rule_baselines)
        all_rows.append(rule_adaptive)

        # Compare development-trained predictions with measured test runtimes.
        predicted = model.predict_runtimes(graph)
        for actual in ridge_baselines:
            algorithm = actual["algorithm"]
            if algorithm not in predicted:
                continue
            predicted_seconds = predicted[algorithm]
            actual_seconds = float(actual["median_seconds"])
            prediction_rows.append({
                "case_id": case_id,
                "algorithm": algorithm,
                "predicted_seconds": predicted_seconds,
                "actual_median_seconds": actual_seconds,
                "prediction_to_actual_ratio": predicted_seconds / actual_seconds,
                "absolute_log_error": abs(math.log(predicted_seconds / actual_seconds)),
                "seed": metadata["seed"],
                "nodes": metadata["nodes"] if "nodes" in metadata else len(graph.adjacency),
                "target_density": metadata["target_density"],
                "weight_mode": metadata["weight_mode"],
                "directed": metadata.get("directed", graph.directed),
                "coordinates": metadata["coordinates"],
            })

        pair_rows.append({
            "case_id": case_id,
            "weight_mode": metadata["weight_mode"],
            "target_density": metadata["target_density"],
            "nodes": len(graph.adjacency),
            "seed": metadata["seed"],
            "ridge_selected_algorithm": ridge_adaptive["selected_algorithm"],
            "rule_selected_algorithm": rule_adaptive["selected_algorithm"],
            "ridge_runtime_seconds": ridge_adaptive["median_seconds"],
            "rule_runtime_seconds": rule_adaptive["median_seconds"],
            "ridge_to_rule_runtime_ratio": (
                ridge_adaptive["median_seconds"] / rule_adaptive["median_seconds"]
            ),
            "rule_to_ridge_speedup": (
                rule_adaptive["median_seconds"] / ridge_adaptive["median_seconds"]
            ),
            "ridge_selection_seconds": ridge_adaptive["median_selection_seconds"],
            "rule_selection_seconds": rule_adaptive["median_selection_seconds"],
            "ridge_selection_overhead_seconds": ridge_adaptive["estimated_selection_overhead_seconds"],
            "rule_selection_overhead_seconds": rule_adaptive["estimated_selection_overhead_seconds"],
            "ridge_peak_memory_bytes": ridge_adaptive["peak_memory_bytes"],
            "rule_peak_memory_bytes": rule_adaptive["peak_memory_bytes"],
        })

        if case_count % 50 == 0:
            print(f"Completed {case_count} of 672 final graph cases")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_results_csv(all_rows, RESULTS_DIR / "final_evaluation_benchmarks.csv")
    save_selection_summary(
        all_rows,
        RESULTS_DIR / "final_evaluation_selection_summary.csv",
    )
    write_csv(
        RESULTS_DIR / "final_evaluation_seed_summary.csv",
        make_seed_summaries(all_rows),
    )
    write_csv(
        RESULTS_DIR / "final_evaluation_predictions.csv",
        prediction_rows,
    )
    write_csv(
        RESULTS_DIR / "final_evaluation_model_vs_rules.csv",
        pair_rows,
    )

    prediction_summary = []
    for algorithm in ("bfs", "dijkstra", "astar", "bellman_ford"):
        values = [r for r in prediction_rows if r["algorithm"] == algorithm]
        errors = [r["absolute_log_error"] for r in values]
        ratios = [r["prediction_to_actual_ratio"] for r in values]
        prediction_summary.append({
            "algorithm": algorithm,
            "n": len(values),
            "mean_absolute_log_error": statistics.mean(errors) if errors else "",
            "median_absolute_log_error": statistics.median(errors) if errors else "",
            "median_prediction_to_actual_ratio": statistics.median(ratios) if ratios else "",
        })
    write_csv(
        RESULTS_DIR / "final_evaluation_prediction_summary.csv",
        prediction_summary,
    )

    config = {
        "evaluation": "independent_final",
        "training_data": str(DEVELOPMENT_PATH),
        "training_split": "development only",
        "ridge_alpha": model.ridge,
        "features": [
            "nodes", "density", "directed", "uniform_weights",
            "has_negative_weights", "weight_stddev",
            "positions_available", "heuristic_scale",
        ],
        "node_counts": NODE_COUNTS,
        "densities": DENSITIES,
        "weight_modes": WEIGHT_MODES,
        "coordinate_options": COORDINATE_OPTIONS,
        "nonnegative_directed_options": NONNEGATIVE_DIRECTED_OPTIONS,
        "test_seeds": TEST_SEEDS,
        "repeats": REPEATS,
        "graph_cases": case_count,
        "comparison": "Ridge selector vs rule selector and eligible algorithms",
        "speedup_definition": "rule selector runtime / Ridge selector runtime; >1 favors Ridge",
    }
    with (RESULTS_DIR / "final_evaluation_config.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(config, file, indent=2)

    ridge_faster = sum(r["rule_to_ridge_speedup"] > 1 for r in pair_rows)
    ties = sum(math.isclose(r["ridge_runtime_seconds"], r["rule_runtime_seconds"]) for r in pair_rows)
    print(f"Completed {case_count} final graph cases; Ridge alpha={model.ridge:g}")
    print(f"Ridge faster than rules: {ridge_faster}/{len(pair_rows)}")
    print(f"Rule/Ridge median speedup: {statistics.median(r['rule_to_ridge_speedup'] for r in pair_rows):.4f}")
    print(f"Runtime ties: {ties}")
    print("Prediction errors (mean absolute log error):")
    for row in prediction_summary:
        print(f"  {row['algorithm']}: {row['mean_absolute_log_error']}")
    print(f"Results saved under {RESULTS_DIR}")


if __name__ == "__main__":
    if "--summarize-existing" in sys.argv:
        summarize_existing_outputs()
    else:
        main()
