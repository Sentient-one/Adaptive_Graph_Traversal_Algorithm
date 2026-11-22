"""Statistical summaries for the frozen synthetic and real-world evaluations."""

import csv
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BACKEND_DIR / "results"
BOOTSTRAP_REPEATS = 5000
BOOTSTRAP_SEED = 73129


def _read(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _number(value, default=None):
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def _wilson(successes, count, z=1.959963984540054):
    if not count:
        return "", ""
    p = successes / count
    denominator = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    margin = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def _percentile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return ""
    return ordered[min(len(ordered) - 1, int(fraction * (len(ordered) - 1)))]


def _exact_sign_p(positive, negative):
    count = positive + negative
    if count == 0:
        return 1.0
    tail = min(positive, negative)
    probability = sum(math.comb(count, i) for i in range(tail + 1)) / (2 ** count)
    return min(1.0, 2 * probability)


def _exact_mcnemar_p(ridge_only, rules_only):
    discordant = ridge_only + rules_only
    if not discordant:
        return 1.0
    tail = min(ridge_only, rules_only)
    probability = sum(math.comb(discordant, i) for i in range(tail + 1)) / (2 ** discordant)
    return min(1.0, 2 * probability)


def _bootstrap_mean_difference(pairs, *, clustered=False, repeats=BOOTSTRAP_REPEATS, seed=BOOTSTRAP_SEED):
    rng = random.Random(seed)
    if clustered:
        groups = defaultdict(list)
        for row in pairs:
            groups[row["cluster"]].append(row["difference"])
        clusters = list(groups.values())
        if len(clusters) < 2:
            return "", ""
        samples = []
        for _ in range(repeats):
            selected = [rng.choice(clusters) for _ in clusters]
            values = [value for cluster in selected for value in cluster]
            samples.append(statistics.mean(values))
    else:
        values = [row["difference"] for row in pairs]
        samples = [statistics.mean(rng.choices(values, k=len(values))) for _ in range(repeats)]
    return _percentile(samples, 0.025), _percentile(samples, 0.975)


def _bootstrap_median(values, *, repeats=BOOTSTRAP_REPEATS, seed=BOOTSTRAP_SEED + 1):
    rng = random.Random(seed)
    groups = defaultdict(list)
    for cluster, value in values:
        groups[cluster].append(value)
    clusters = list(groups.values())
    if len(clusters) < 2:
        return "", ""
    estimates = []
    for _ in range(repeats):
        sample_clusters = [rng.choice(clusters) for _ in clusters]
        sample = [value for cluster in sample_clusters for value in cluster]
        estimates.append(statistics.median(sample))
    return _percentile(estimates, 0.025), _percentile(estimates, 0.975)


def _records_from_synthetic(benchmark_rows):
    grouped = defaultdict(list)
    for row in benchmark_rows:
        if row.get("evaluation") != "independent_final":
            continue
        grouped[(row.get("case_id"), row.get("split"))].append(row)

    records = []
    for (case_id, split), rows in grouped.items():
        adaptive = next((row for row in rows if row.get("algorithm") == "adaptive"), None)
        if adaptive is None:
            continue
        algorithms = [row for row in rows if row.get("algorithm") != "adaptive"]
        if not algorithms:
            continue
        fastest = min(float(row["median_seconds"]) for row in algorithms)
        winners = {
            row["algorithm"] for row in algorithms
            if math.isclose(float(row["median_seconds"]), fastest, rel_tol=1e-9, abs_tol=1e-12)
        }
        selected = next(row for row in algorithms if row["algorithm"] == adaptive["selected_algorithm"])
        records.append({
            "evaluation": "synthetic_final",
            "dataset_id": "all_synthetic_test_cases",
            "cluster": str(adaptive.get("seed", case_id)),
            "case_id": case_id,
            "policy": adaptive.get("selector_policy", "unknown"),
            "weight_mode": adaptive.get("weight_mode", "unknown"),
            "algorithm_count": len(algorithms),
            "winner_correct": adaptive["selected_algorithm"] in winners,
            "selected_to_oracle_ratio": float(selected["median_seconds"]) / fastest,
            "adaptive_to_oracle_ratio": float(adaptive["median_seconds"]) / fastest,
            "adaptive_runtime": float(adaptive["median_seconds"]),
            "oracle_runtime": fastest,
            "regret": float(adaptive["median_seconds"]) - fastest,
            "near_selected": float(selected["median_seconds"]) / fastest <= 1.10,
            "near_adaptive": float(adaptive["median_seconds"]) / fastest <= 1.10,
            "correctness": adaptive.get("correctness_verified", "True").lower() == "true",
        })
    return records


def _records_from_real(query_rows):
    records = []
    for row in query_rows:
        records.append({
            "evaluation": "real_world_sampled",
            "dataset_id": row["dataset_id"],
            "cluster": row["dataset_id"].replace("DIMACS9-", "").split("-")[0],
            "case_id": row["case_id"],
            "policy": row["selector_policy"],
            "weight_mode": row["endpoint_class"],
            "algorithm_count": int(row["eligible_algorithm_count"]),
            "winner_correct": row["winner_correct"].lower() == "true",
            "selected_to_oracle_ratio": float(row["selected_to_oracle_ratio"]),
            "adaptive_to_oracle_ratio": float(row["adaptive_to_oracle_ratio"]),
            "adaptive_runtime": float(row["adaptive_runtime_seconds"]),
            "oracle_runtime": float(row["oracle_runtime_seconds"]),
            "regret": float(row["adaptive_runtime_regret_seconds"]),
            "near_selected": float(row["selected_to_oracle_ratio"]) <= 1.10,
            "near_adaptive": float(row["adaptive_to_oracle_ratio"]) <= 1.10,
            "correctness": row["correctness_verified"].lower() == "true",
        })
    return records


def _summary_row(evaluation, dataset, policy, records):
    regrets = [row["regret"] for row in records]
    ratios = [row["adaptive_to_oracle_ratio"] for row in records]
    nontrivial = [row for row in records if row["algorithm_count"] > 1]
    return {
        "evaluation": evaluation,
        "dataset_or_stratum": dataset,
        "selector_policy": policy,
        "query_count": len(records),
        "winner_accuracy": statistics.mean(row["winner_correct"] for row in records),
        "nontrivial_case_count": len(nontrivial),
        "nontrivial_winner_accuracy": statistics.mean(row["winner_correct"] for row in nontrivial) if nontrivial else "",
        "median_selected_to_oracle_ratio": statistics.median(row["selected_to_oracle_ratio"] for row in records),
        "median_adaptive_to_oracle_ratio": statistics.median(ratios),
        "mean_runtime_regret_seconds": statistics.mean(regrets),
        "median_runtime_regret_seconds": statistics.median(regrets),
        "p90_runtime_regret_seconds": _percentile(regrets, 0.90),
        "maximum_runtime_regret_seconds": max(regrets),
        "near_optimal_without_overhead_rate": statistics.mean(row["near_selected"] for row in records),
        "near_optimal_with_overhead_rate": statistics.mean(row["near_adaptive"] for row in records),
        "all_path_costs_verified": all(row["correctness"] for row in records),
    }


def _paired_comparison(evaluation, ridge_rows, rule_rows, *, clustered=False):
    rule_by_case = {row["case_id"]: row for row in rule_rows}
    pairs = [
        {"ridge": row, "rules": rule_by_case[row["case_id"]],
         "difference": float(row["winner_correct"]) - float(rule_by_case[row["case_id"]]["winner_correct"]),
         "cluster": row["cluster"]}
        for row in ridge_rows if row["case_id"] in rule_by_case
    ]
    ridge_correct = sum(pair["ridge"]["winner_correct"] for pair in pairs)
    rules_correct = sum(pair["rules"]["winner_correct"] for pair in pairs)
    nontrivial_pairs = [pair for pair in pairs if pair["ridge"]["algorithm_count"] > 1]
    ridge_only = sum(pair["difference"] > 0 for pair in pairs)
    rules_only = sum(pair["difference"] < 0 for pair in pairs)
    acc_diffs = [{"difference": pair["difference"], "cluster": pair["cluster"]} for pair in pairs]
    ci_low, ci_high = _bootstrap_mean_difference(acc_diffs, clustered=clustered)
    runtime_ratios = [pair["ridge"]["adaptive_runtime"] / pair["rules"]["adaptive_runtime"] for pair in pairs]
    faster = sum(ratio < 1 for ratio in runtime_ratios)
    slower = sum(ratio > 1 for ratio in runtime_ratios)
    runtime_ci_low, runtime_ci_high = _bootstrap_median(
        [(pair["cluster"], pair["ridge"]["adaptive_runtime"] / pair["rules"]["adaptive_runtime"])
         for pair in pairs]
    ) if clustered else ("", "")
    return {
        "evaluation": evaluation,
        "paired_case_count": len(pairs),
        "ridge_winner_accuracy": ridge_correct / len(pairs) if pairs else "",
        "rule_winner_accuracy": rules_correct / len(pairs) if pairs else "",
        "nontrivial_case_count": len(nontrivial_pairs),
        "ridge_nontrivial_winner_accuracy": statistics.mean(pair["ridge"]["winner_correct"] for pair in nontrivial_pairs) if nontrivial_pairs else "",
        "rule_nontrivial_winner_accuracy": statistics.mean(pair["rules"]["winner_correct"] for pair in nontrivial_pairs) if nontrivial_pairs else "",
        "accuracy_difference_ridge_minus_rules": (ridge_correct - rules_correct) / len(pairs) if pairs else "",
        "accuracy_difference_ci95_low": ci_low,
        "accuracy_difference_ci95_high": ci_high,
        "accuracy_ci_method": "cluster bootstrap by graph/network" if clustered else "paired case bootstrap",
        "ridge_only_correct": ridge_only,
        "rules_only_correct": rules_only,
        "mcnemar_exact_two_sided_p": "" if clustered else _exact_mcnemar_p(ridge_only, rules_only),
        "mcnemar_note": "Not reported for clustered real-world queries" if clustered else "Exact two-sided test on paired case correctness",
        "median_paired_ridge_over_rule_runtime_ratio": statistics.median(runtime_ratios) if runtime_ratios else "",
        "median_paired_runtime_ratio_cluster_ci95_low": runtime_ci_low,
        "median_paired_runtime_ratio_cluster_ci95_high": runtime_ci_high,
        "ridge_faster_cases": faster,
        "rules_faster_cases": slower,
        "paired_runtime_sign_test_p": "" if clustered else _exact_sign_p(faster, slower),
        "runtime_sign_test_note": "Not reported for clustered real-world queries" if clustered else "Exact sign test on paired end-to-end times",
        "runtime_tie_cases": len(runtime_ratios) - faster - slower,
        "runtime_ratio_note": "Ridge adaptive end-to-end runtime divided by rules adaptive end-to-end runtime; below 1 favors Ridge",
    }


def _write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows for {path}")
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def run():
    final_benchmarks = _read(RESULTS_DIR / "final_evaluation_benchmarks.csv")
    real_queries = _read(RESULTS_DIR / "real_world_query_results.csv")
    synthetic = _records_from_synthetic(final_benchmarks)
    real = _records_from_real(real_queries)

    summaries = []
    confidence = []
    for evaluation, records in (("synthetic_final", synthetic), ("real_world_sampled", real)):
        groups = defaultdict(list)
        for row in records:
            groups[(row["dataset_id"], row["policy"])].append(row)
            if evaluation == "synthetic_final":
                groups[(f"weight_mode:{row['weight_mode']}", row["policy"])].append(row)
            else:
                groups[("ALL_FIVE_GRAPH_INPUTS", row["policy"])].append(row)
        for (dataset, policy), values in sorted(groups.items()):
            summaries.append(_summary_row(evaluation, dataset, policy, values))
            successes = sum(row["winner_correct"] for row in values)
            low, high = _wilson(successes, len(values))
            confidence.append({
                "evaluation": evaluation,
                "dataset_or_stratum": dataset,
                "selector_policy": policy,
                "metric": "winner_accuracy",
                "estimate": successes / len(values),
                "ci95_low": low,
                "ci95_high": high,
                "sample_count": len(values),
                "method": "Wilson score; query-level descriptive interval",
            })
            if evaluation == "real_world_sampled" and dataset == "ALL_FIVE_GRAPH_INPUTS":
                cluster_low, cluster_high = _bootstrap_mean_difference(
                    [{"difference": float(row["winner_correct"]), "cluster": row["cluster"]}
                     for row in values],
                    clustered=True,
                )
                confidence.append({
                    "evaluation": evaluation,
                    "dataset_or_stratum": dataset,
                    "selector_policy": policy,
                    "metric": "winner_accuracy",
                    "estimate": successes / len(values),
                    "ci95_low": cluster_low,
                    "ci95_high": cluster_high,
                    "sample_count": len(values),
                    "method": "geographic-network cluster bootstrap; four networks, exploratory",
                })

    synthetic_ridge = [row for row in synthetic if row["policy"] == "ridge_regression"]
    synthetic_rules = [row for row in synthetic if row["policy"] == "rules"]
    real_ridge = [row for row in real if row["policy"] == "ridge_regression"]
    real_rules = [row for row in real if row["policy"] == "rules"]
    paired = [
        _paired_comparison("synthetic_final", synthetic_ridge, synthetic_rules),
        _paired_comparison("real_world_sampled", real_ridge, real_rules, clustered=True),
    ]

    prediction_rows = _read(RESULTS_DIR / "final_evaluation_predictions.csv")
    errors = defaultdict(list)
    for row in prediction_rows:
        error = _number(row.get("absolute_log_error"))
        if error is None:
            continue
        algo = row["algorithm"]
        errors[("prediction_error", f"algorithm:{algo}")].append(error)
        errors[("prediction_error", f"algorithm:{algo},size:{row['nodes']}")].append(error)
        errors[("prediction_error", f"algorithm:{algo},weight:{row['weight_mode']}")].append(error)
        errors[("prediction_error", f"algorithm:{algo},coordinates:{row['coordinates']}")].append(error)
    for (evaluation, stratum), values in errors.items():
        summaries.append({
            "evaluation": evaluation,
            "dataset_or_stratum": stratum,
            "selector_policy": "ridge_regression",
            "query_count": len(values),
            "mean_absolute_log_error": statistics.mean(values),
            "median_absolute_log_error": statistics.median(values),
            "p90_absolute_log_error": _percentile(values, 0.90),
        })

    relationships = []
    algorithm_rows = [row for row in final_benchmarks if row.get("algorithm") != "adaptive"]
    def correlation(label, first_key, second_key, rows):
        pairs = [(_number(row.get(first_key)), _number(row.get(second_key))) for row in rows]
        pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
        if len(pairs) < 2:
            return
        xs, ys = zip(*pairs)
        mx, my = statistics.mean(xs), statistics.mean(ys)
        den = math.sqrt(sum((x-mx)**2 for x in xs) * sum((y-my)**2 for y in ys))
        if den:
            relationships.append({"relationship": label, "sample_count": len(pairs), "pearson_r": sum((x-mx)*(y-my) for x,y in pairs) / den})

    correlation("density_vs_average_degree", "density", "average_degree", algorithm_rows)
    correlation("uniform_weights_vs_weight_stddev", "uniform_weights", "weight_stddev", algorithm_rows)
    for algorithm in ("bfs", "dijkstra", "astar", "bellman_ford"):
        rows = [row for row in algorithm_rows if row.get("algorithm") == algorithm]
        correlation(f"nodes_vs_{algorithm}_runtime", "nodes", "median_seconds", rows)
        correlation(f"density_vs_{algorithm}_runtime", "density", "median_seconds", rows)
    correlation("heuristic_scale_vs_astar_runtime", "heuristic_scale", "median_seconds", [
        row for row in algorithm_rows if row.get("algorithm") == "astar"
    ])

    _write_csv(RESULTS_DIR / "statistical_summary.csv", summaries)
    _write_csv(RESULTS_DIR / "confidence_intervals.csv", confidence)
    _write_csv(RESULTS_DIR / "paired_comparisons.csv", paired)
    _write_csv(RESULTS_DIR / "feature_relationships.csv", relationships)
    print("Saved statistical analysis CSV files")


if __name__ == "__main__":
    run()
