"""Tune Ridge alpha for Model B using repeated grouped cross-validation.

Uses only the existing development benchmark measurements. Each alpha uses
identical features, cases, seeds, and grouped folds; only regularization varies.
The production performance model is not modified.
"""

import csv
import math
import statistics
from pathlib import Path

import experiments.feature_ablation_cv as cv

ALPHAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
FEATURES = cv.FEATURES["B"]
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "results" / "ridge_alpha_cv.csv"
STABILITY_METRICS = (
    "winner_accuracy",
    "nontrivial_winner_accuracy",
    "median_abs_log_error",
    "bfs_median_abs_log_error",
    "dijkstra_median_abs_log_error",
    "astar_median_abs_log_error",
    "bellman_ford_median_abs_log_error",
    "astar_median_prediction_ratio",
)


def run():
    cases = cv.load_cases()
    output_rows = []
    alpha_seed_summaries = {}

    for alpha in ALPHAS:
        cv.RIDGE_ALPHA = alpha
        alpha_seed_summaries[alpha] = []

        for seed in cv.SEEDS:
            seed_predictions = []
            folds = cv.grouped_folds(cases.keys(), seed)

            for fold_number, validation_ids in enumerate(folds, start=1):
                validation_ids = list(validation_ids)
                validation_set = set(validation_ids)
                training_cases = {
                    key: value for key, value in cases.items()
                    if key not in validation_set
                }
                validation_cases = {
                    key: cases[key] for key in validation_ids
                }
                models = cv.train_model(training_cases, FEATURES)
                predictions = cv.evaluate_cases(validation_cases, models)
                fold_metrics = cv.summarize(predictions)
                output_rows.append({
                    "record_type": "fold",
                    "alpha": alpha,
                    "seed": seed,
                    "fold": fold_number,
                    **fold_metrics,
                })
                seed_predictions.extend(predictions)

            seed_metrics = cv.summarize(seed_predictions)
            alpha_seed_summaries[alpha].append(seed_metrics)
            output_rows.append({
                "record_type": "seed_summary",
                "alpha": alpha,
                "seed": seed,
                "fold": "ALL",
                **seed_metrics,
            })

    summary_by_alpha = {}
    for alpha, seed_summaries in alpha_seed_summaries.items():
        overall = {"record_type": "overall_summary", "alpha": alpha,
                   "seed": "ALL", "fold": "ALL"}
        for metric in STABILITY_METRICS:
            values = [item[metric] for item in seed_summaries
                      if math.isfinite(item[metric])]
            overall[metric + "_seed_mean"] = (
                statistics.mean(values) if values else float("nan")
            )
            overall[metric + "_seed_std"] = (
                statistics.stdev(values) if len(values) > 1 else 0.0
            )
        summary_by_alpha[alpha] = overall

    # Report the strongest accuracy values while retaining all metrics for
    # interpretation; ties in non-trivial accuracy break on median log error.
    best = sorted(
        summary_by_alpha.values(),
        key=lambda row: (
            -row["nontrivial_winner_accuracy_seed_mean"],
            row["median_abs_log_error_seed_mean"],
            row["alpha"],
        ),
    )[0]
    for row in output_rows:
        row["selected_best_alpha_by_accuracy_then_error"] = best["alpha"]
        row["selection_rule"] = (
            "highest mean non-trivial winner accuracy; tie-break by lowest "
            "mean median absolute log error"
        )
    for row in summary_by_alpha.values():
        row["selected_best_alpha_by_accuracy_then_error"] = best["alpha"]
        row["selection_rule"] = (
            "highest mean non-trivial winner accuracy; tie-break by lowest "
            "mean median absolute log error"
        )
        output_rows.append(row)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for row in output_rows for key in row))
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Development graph cases: {len(cases)}")
    print(f"Model B features ({len(FEATURES)}): {', '.join(FEATURES)}")
    print(f"Seeds: {cv.SEEDS}; grouped folds per seed: {cv.FOLDS}")
    print("alpha,winner_accuracy,nontrivial_winner_accuracy,"
          "median_abs_log_error,BFS_error,Dijkstra_error,A*_error,"
          "Bellman-Ford_error,A*_ratio")
    for alpha in ALPHAS:
        row = summary_by_alpha[alpha]
        print(",".join([
            f"{alpha:g}",
            f"{row['winner_accuracy_seed_mean']:.6f}"
            f" +/- {row['winner_accuracy_seed_std']:.6f}",
            f"{row['nontrivial_winner_accuracy_seed_mean']:.6f}"
            f" +/- {row['nontrivial_winner_accuracy_seed_std']:.6f}",
            f"{row['median_abs_log_error_seed_mean']:.6f}"
            f" +/- {row['median_abs_log_error_seed_std']:.6f}",
            f"{row['bfs_median_abs_log_error_seed_mean']:.6f}"
            f" +/- {row['bfs_median_abs_log_error_seed_std']:.6f}",
            f"{row['dijkstra_median_abs_log_error_seed_mean']:.6f}"
            f" +/- {row['dijkstra_median_abs_log_error_seed_std']:.6f}",
            f"{row['astar_median_abs_log_error_seed_mean']:.6f}"
            f" +/- {row['astar_median_abs_log_error_seed_std']:.6f}",
            f"{row['bellman_ford_median_abs_log_error_seed_mean']:.6f}"
            f" +/- {row['bellman_ford_median_abs_log_error_seed_std']:.6f}",
            f"{row['astar_median_prediction_ratio_seed_mean']:.6f}"
            f" +/- {row['astar_median_prediction_ratio_seed_std']:.6f}",
        ]))
    print(f"Selected by stated rule: alpha={best['alpha']:g}")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    run()
