
"""Repeated grouped five-fold CV for feature ablation experiments.

Uses existing development benchmark measurements only.
Does not modify the production performance model.
"""

import csv
import math
import random
import statistics
from pathlib import Path

from collections import defaultdict


BACKEND_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BACKEND_DIR / "results"

INPUT_PATH = RESULTS_DIR / "development_benchmarks.csv"
OUTPUT_PATH = RESULTS_DIR / "feature_ablation_repeated_cv.csv"

SEEDS = [11, 22, 42, 101, 202]
FOLDS = 5
RIDGE_ALPHA = 1e-3

ALGORITHMS = [
    "bfs",
    "dijkstra",
    "astar",
    "bellman_ford",
]

FEATURES = {
    "A": [
        "nodes",
        "density",
        "directed",
        "uniform_weights",
        "has_negative_weights",
        "average_degree",
        "weight_stddev",
        "positions_available",
        "heuristic_scale",
    ],
    "B": [
        "nodes",
        "density",
        "directed",
        "uniform_weights",
        "has_negative_weights",
        "weight_stddev",
        "positions_available",
        "heuristic_scale",
    ],
    "C": [
        "nodes",
        "density",
        "directed",
        "uniform_weights",
        "has_negative_weights",
        "positions_available",
        "heuristic_scale",
    ],
    "D": [
        "nodes",
        "density",
        "directed",
        "has_negative_weights",
        "weight_stddev",
        "positions_available",
        "heuristic_scale",
    ],
}


def number(value, default=0.0):
    if value is None or value == "":
        return default

    if isinstance(value, str):
        value = value.strip()

        if value.lower() == "true":
            return 1.0

        if value.lower() == "false":
            return 0.0

    try:
        result = float(value)
    except (TypeError, ValueError):
        return default

    return result if math.isfinite(result) else default


def feature_value(row, name):
    value = number(row.get(name))

    if name == "nodes":
        if value < 0:
            raise ValueError("Node count cannot be negative")
        return math.log1p(value)

    return value


def load_cases():
    cases = {}

    with INPUT_PATH.open(
        newline="",
        encoding="utf-8",
    ) as file:

        for row in csv.DictReader(file):

            algorithm = row["algorithm"]

            # Adaptive rows are not algorithm training targets.
            if algorithm == "adaptive":
                continue

            if algorithm not in ALGORITHMS:
                continue

            case_id = row["case_id"]

            runtime = number(
                row.get("median_seconds"),
                -1,
            )

            if runtime <= 0:
                raise ValueError(
                    f"Invalid runtime in case {case_id}"
                )

            if case_id not in cases:
                cases[case_id] = {
                    "features": {
                        name: feature_value(row, name)
                        for name in FEATURES["A"]
                    },
                    "runtimes": {},
                }

            if algorithm in cases[case_id]["runtimes"]:
                raise ValueError(
                    f"Duplicate {algorithm} measurement "
                    f"for case {case_id}"
                )

            cases[case_id]["runtimes"][algorithm] = runtime

    if not cases:
        raise ValueError("No development cases found")

    for case_id, case in cases.items():
        if not case["runtimes"]:
            raise ValueError(
                f"No algorithm measurements for {case_id}"
            )

    return cases


def solve(matrix, vector):
    """Gaussian elimination with partial pivoting."""

    size = len(vector)

    augmented = [
        row[:] + [value]
        for row, value in zip(matrix, vector)
    ]

    for column in range(size):

        pivot = max(
            range(column, size),
            key=lambda row: abs(
                augmented[row][column]
            ),
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


def train_model(training_cases, selected_features):
    """Fit independent standardized Ridge models."""

    vectors = [
        [
            case["features"][feature]
            for feature in selected_features
        ]
        for case in training_cases.values()
    ]

    width = len(selected_features)

    centers = [
        statistics.mean(
            vector[index] for vector in vectors
        )
        for index in range(width)
    ]

    scales = [
        math.sqrt(
            statistics.mean(
                (vector[index] - centers[index]) ** 2
                for vector in vectors
            )
        ) or 1.0
        for index in range(width)
    ]

    def standardize(features):
        return [
            (
                features[name] - centers[index]
            ) / scales[index]
            for index, name in enumerate(
                selected_features
            )
        ]

    models = {}

    for algorithm in ALGORITHMS:

        observations = []

        for case in training_cases.values():

            runtime = case["runtimes"].get(
                algorithm
            )

            if runtime is None:
                continue

            vector = standardize(
                case["features"]
            )

            observations.append(
                ([1.0, *vector], math.log(runtime))
            )

        if not observations:
            continue

        model_width = width + 1

        normal = [
            [0.0] * model_width
            for _ in range(model_width)
        ]

        target = [0.0] * model_width

        for design, value in observations:

            for i in range(model_width):

                target[i] += design[i] * value

                for j in range(model_width):
                    normal[i][j] += (
                        design[i] * design[j]
                    )

        # Do not regularize the intercept.
        for i in range(1, model_width):
            normal[i][i] += RIDGE_ALPHA

        models[algorithm] = {
            "coefficients": solve(
                normal,
                target,
            ),
            "centers": centers,
            "scales": scales,
            "features": selected_features,
        }

    return models


def predict_runtime(model, features):

    selected = model["features"]

    vector = [
        1.0,
        *[
            (
                features[name] - model["centers"][index]
            ) / model["scales"][index]
            for index, name in enumerate(selected)
        ],
    ]

    log_prediction = sum(
        a * b
        for a, b in zip(
            model["coefficients"],
            vector,
        )
    )

    return math.exp(
        max(-700, min(700, log_prediction))
    )


def evaluate_cases(validation_cases, models):

    predictions = []

    for case_id, case in validation_cases.items():

        actual = case["runtimes"]

        eligible = sorted(
            algorithm
            for algorithm in actual
            if algorithm in models
        )

        if not eligible:
            raise ValueError(
                f"No trained eligible models for {case_id}"
            )

        predicted = {
            algorithm: predict_runtime(
                models[algorithm],
                case["features"],
            )
            for algorithm in eligible
        }

        selected = min(
            predicted,
            key=predicted.get,
        )

        actual_minimum = min(
            actual[algorithm]
            for algorithm in eligible
        )

        actual_winners = {
            algorithm
            for algorithm in eligible
            if math.isclose(
                actual[algorithm],
                actual_minimum,
                rel_tol=1e-9,
                abs_tol=1e-12,
            )
        }

        errors = {
            algorithm: abs(
                math.log(
                    predicted[algorithm]
                    / actual[algorithm]
                )
            )
            for algorithm in eligible
        }

        ratios = {
            algorithm: (
                predicted[algorithm]
                / actual[algorithm]
            )
            for algorithm in eligible
        }

        predictions.append({
            "case_id": case_id,
            "eligible_count": len(eligible),
            "winner_correct": (
                selected in actual_winners
            ),
            "nontrivial_correct": (
                selected in actual_winners
                if len(eligible) > 1
                else None
            ),
            "errors": errors,
            "ratios": ratios,
        })

    return predictions


def summarize(predictions):

    winner_values = [
        float(item["winner_correct"])
        for item in predictions
    ]

    nontrivial_values = [
        float(item["nontrivial_correct"])
        for item in predictions
        if item["nontrivial_correct"] is not None
    ]

    all_errors = [
        error
        for item in predictions
        for error in item["errors"].values()
    ]

    result = {
        "n_cases": len(predictions),
        "n_nontrivial_cases": len(nontrivial_values),

        "winner_accuracy": (
            statistics.mean(winner_values)
            if winner_values else float("nan")
        ),

        "nontrivial_winner_accuracy": (
            statistics.mean(nontrivial_values)
            if nontrivial_values else float("nan")
        ),

        "median_abs_log_error": (
            statistics.median(all_errors)
            if all_errors else float("nan")
        ),
    }

    for algorithm in ALGORITHMS:

        errors = [
            item["errors"][algorithm]
            for item in predictions
            if algorithm in item["errors"]
        ]

        result[f"{algorithm}_n"] = len(errors)

        result[
            f"{algorithm}_median_abs_log_error"
        ] = (
            statistics.median(errors)
            if errors else float("nan")
        )

    astar_ratios = [
        item["ratios"]["astar"]
        for item in predictions
        if "astar" in item["ratios"]
    ]

    result["astar_median_prediction_ratio"] = (
        statistics.median(astar_ratios)
        if astar_ratios else float("nan")
    )

    result["astar_n"] = len(astar_ratios)

    return result


def grouped_folds(case_ids, seed):

    groups = list(case_ids)

    random.Random(seed).shuffle(groups)

    # Each graph case occurs in exactly one fold.
    return [
        groups[index::FOLDS]
        for index in range(FOLDS)
    ]


def main():

    print("Loading development benchmarks...")

    cases = load_cases()

    print(f"Loaded {len(cases)} graph cases.")

    trivial = sum(
        len(case["runtimes"]) == 1
        for case in cases.values()
    )

    print(f"Single-algorithm cases: {trivial}")
    print(
        f"Non-trivial cases: {len(cases) - trivial}"
    )

    output_rows = []

    seed_results = {
        model: []
        for model in FEATURES
    }

    for seed in SEEDS:

        print(f"\nRepeated CV seed: {seed}")

        folds = grouped_folds(
            cases.keys(),
            seed,
        )

        seed_predictions = {
            model: []
            for model in FEATURES
        }

        for fold_number, validation_ids in enumerate(
            folds,
            start=1,
        ):

            validation_set = set(validation_ids)

            training_cases = {
                key: value
                for key, value in cases.items()
                if key not in validation_set
            }

            validation_cases = {
                key: cases[key]
                for key in validation_ids
            }

            print(
                f"  Fold {fold_number}: "
                f"{len(training_cases)} train, "
                f"{len(validation_cases)} validation"
            )

            # All four models use the exact same fold.
            for model_name, selected_features in FEATURES.items():

                models = train_model(
                    training_cases,
                    selected_features,
                )

                fold_predictions = evaluate_cases(
                    validation_cases,
                    models,
                )

                seed_predictions[model_name].extend(
                    fold_predictions
                )

                metrics = summarize(
                    fold_predictions
                )

                output_rows.append({
                    "record_type": "fold",
                    "model": model_name,
                    "seed": seed,
                    "fold": fold_number,
                    **metrics,
                })

        for model_name in FEATURES:

            metrics = summarize(
                seed_predictions[model_name]
            )

            seed_results[model_name].append(
                metrics
            )

            output_rows.append({
                "record_type": "seed_summary",
                "model": model_name,
                "seed": seed,
                "fold": "",
                **metrics,
            })

    # Calculate mean and sample standard deviation
    # across the five repeated CV seeds.
    metric_names = [
        "winner_accuracy",
        "nontrivial_winner_accuracy",
        "median_abs_log_error",
        "bfs_median_abs_log_error",
        "dijkstra_median_abs_log_error",
        "astar_median_abs_log_error",
        "bellman_ford_median_abs_log_error",
        "astar_median_prediction_ratio",
    ]

    for model_name in FEATURES:

        summaries = seed_results[model_name]

        overall = {}

        for metric in metric_names:

            values = [
                item[metric]
                for item in summaries
                if math.isfinite(item[metric])
            ]

            overall[metric] = (
                statistics.mean(values)
                if values else float("nan")
            )

            overall[f"{metric}_std"] = (
                statistics.stdev(values)
                if len(values) > 1
                else 0.0
            )

        output_rows.append({
            "record_type": "overall_summary",
            "model": model_name,
            "seed": "ALL",
            "fold": "",
            **overall,
        })

    columns = list(dict.fromkeys(
        key
        for row in output_rows
        for key in row
    ))

    with OUTPUT_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=columns,
        )

        writer.writeheader()
        writer.writerows(output_rows)

    print("\n" + "=" * 65)
    print("REPEATED FEATURE ABLATION RESULTS")
    print("=" * 65)

    for model_name in FEATURES:

        summary = next(
            row
            for row in output_rows
            if row["record_type"] == "overall_summary"
            and row["model"] == model_name
        )

        accuracy = summary["winner_accuracy"]
        accuracy_std = summary["winner_accuracy_std"]

        nontrivial = summary[
            "nontrivial_winner_accuracy"
        ]

        nontrivial_std = summary[
            "nontrivial_winner_accuracy_std"
        ]

        log_error = summary["median_abs_log_error"]
        log_error_std = summary[
            "median_abs_log_error_std"
        ]

        astar_ratio = summary[
            "astar_median_prediction_ratio"
        ]

        astar_ratio_std = summary[
            "astar_median_prediction_ratio_std"
        ]

        print(f"\nModel {model_name}")
        print(
            f"  Winner accuracy: "
            f"{accuracy:.4f} ± {accuracy_std:.4f}"
        )
        print(
            f"  Non-trivial accuracy: "
            f"{nontrivial:.4f} ± {nontrivial_std:.4f}"
        )
        print(
            f"  Median absolute log error: "
            f"{log_error:.4f} ± {log_error_std:.4f}"
        )
        print(
            f"  A* median prediction ratio: "
            f"{astar_ratio:.4f} ± {astar_ratio_std:.4f}"
        )

        for algorithm in ALGORITHMS:
            print(
                f"  {algorithm} median absolute log error: "
                f"{summary[algorithm + '_median_abs_log_error']:.4f}"
            )

    print("\nResults saved to:")
    print(OUTPUT_PATH)

    print(
        f"\nTotal result rows: {len(output_rows)}"
    )


if __name__ == "__main__":
    main()
