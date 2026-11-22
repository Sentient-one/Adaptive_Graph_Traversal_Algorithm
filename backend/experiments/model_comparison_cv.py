"""Compare regressors/features using development-only repeated grouped CV.

This experiment never reads the independent final or real-world test outcomes
for model fitting or selection and does not modify the production selector.
Requires the optional scikit-learn dependency listed in
requirements-experiments.txt.
"""

import csv
import json
import math
import pickle
import random
import statistics
from collections import defaultdict
from pathlib import Path
from time import perf_counter_ns

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from adaptive.features import extract_selection_features
from experiments.run_benchmarks import _cases


BACKEND_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BACKEND_DIR / "results"
INPUT_PATH = RESULTS_DIR / "development_benchmarks.csv"
OUTPUT_PATH = RESULTS_DIR / "model_comparison_repeated_cv.csv"
CONFIG_PATH = RESULTS_DIR / "model_comparison_cv_config.json"
SEEDS = (11, 22, 42, 101, 202)
FOLDS = 5
ALGORITHMS = ("bfs", "dijkstra", "astar", "bellman_ford")
BASE_FEATURES = (
    "nodes", "density", "directed", "uniform_weights",
    "has_negative_weights", "weight_stddev", "positions_available",
    "heuristic_scale",
)
EXTENDED_FEATURES = BASE_FEATURES + (
    "source_degree", "destination_degree", "endpoint_heuristic_distance",
    "mean_weight", "unique_weight_ratio",
)
MODEL_SPECS = ("ridge", "random_forest", "gradient_boosting")


def _num(value, default=0.0):
    if value is None or value == "":
        return default
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return float(value.strip().lower() == "true")
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def _extract_features(graph):
    raw = extract_selection_features(graph)
    source, destination = "0", str(len(graph.adjacency) - 1)
    raw["nodes"] = math.log1p(raw["node_count"])
    raw["source_degree"] = len(graph.neighbors(source))
    raw["destination_degree"] = len(graph.neighbors(destination))
    raw["mean_weight"] = graph.mean_weight or 0.0
    raw["unique_weight_ratio"] = graph.unique_weight_ratio
    raw["endpoint_heuristic_distance"] = 0.0
    if (
        raw["heuristic_scale"] is not None
        and source in graph.positions
        and destination in graph.positions
    ):
        x1, y1 = graph.positions[source]
        x2, y2 = graph.positions[destination]
        raw["endpoint_heuristic_distance"] = (
            raw["heuristic_scale"] * math.hypot(x2 - x1, y2 - y1)
        )
    for name in BASE_FEATURES + EXTENDED_FEATURES[len(BASE_FEATURES):]:
        if raw.get(name) is None:
            raw[name] = 0.0
    return raw


def _load_cases():
    runtime_by_case = defaultdict(dict)
    with INPUT_PATH.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if row.get("algorithm") == "adaptive":
                continue
            runtime = _num(row.get("median_seconds"), -1)
            if runtime > 0:
                runtime_by_case[row["case_id"]][row["algorithm"]] = runtime

    features_by_case = {}
    for graph, metadata in _cases("development"):
        features_by_case[metadata["case_id"]] = _extract_features(graph)
    if runtime_by_case.keys() != features_by_case.keys():
        raise ValueError("Reconstructed development cases do not match saved benchmark case IDs")
    return {
        case_id: {"features": features_by_case[case_id], "runtimes": runtimes}
        for case_id, runtimes in runtime_by_case.items()
    }


def _make_regressor(model_name, seed):
    if model_name == "ridge":
        # Standardized features reproduce the production preprocessing.
        return make_pipeline(StandardScaler(), Ridge(alpha=1e-4))
    if model_name == "random_forest":
        return RandomForestRegressor(
            n_estimators=100,
            min_samples_leaf=2,
            max_features=1.0,
            random_state=seed,
            n_jobs=1,
        )
    return GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=2,
        loss="huber",
        random_state=seed,
    )


def _feature_sets():
    return {"model_b": BASE_FEATURES, "model_b_plus_endpoints": EXTENDED_FEATURES}


def _train(training, feature_names, model_name, seed):
    models = {}
    for algorithm in ALGORITHMS:
        rows = [case for case in training.values() if algorithm in case["runtimes"]]
        if not rows:
            continue
        estimator = _make_regressor(model_name, seed + ALGORITHMS.index(algorithm))
        x = [[case["features"][name] for name in feature_names] for case in rows]
        y = [math.log(case["runtimes"][algorithm]) for case in rows]
        estimator.fit(x, y)
        models[algorithm] = estimator
    return models


def _evaluate(validation, models, feature_names):
    results = []
    for case_id, case in validation.items():
        eligible = sorted(name for name in case["runtimes"] if name in models)
        if not eligible:
            continue
        x = [[case["features"][name] for name in feature_names]]
        predicted, errors = {}, []
        inference_ns = 0
        for algorithm in eligible:
            started = perf_counter_ns()
            log_prediction = float(models[algorithm].predict(x)[0])
            inference_ns += perf_counter_ns() - started
            prediction = math.exp(max(-700, min(700, log_prediction)))
            predicted[algorithm] = prediction
            errors.append(abs(math.log(prediction / case["runtimes"][algorithm])))
        selected = min(predicted, key=predicted.get)
        oracle_runtime = min(case["runtimes"][algorithm] for algorithm in eligible)
        winners = {
            algorithm for algorithm in eligible
            if math.isclose(case["runtimes"][algorithm], oracle_runtime, rel_tol=1e-9, abs_tol=1e-12)
        }
        selected_runtime = case["runtimes"][selected]
        results.append({
            "case_id": case_id,
            "eligible_count": len(eligible),
            "winner_correct": selected in winners,
            "nontrivial_correct": selected in winners if len(eligible) > 1 else None,
            "runtime_ratio": selected_runtime / oracle_runtime,
            "median_abs_log_error": statistics.median(errors),
            "prediction_inference_ns": inference_ns,
        })
    return results


def _metrics(results):
    nontrivial = [row for row in results if row["nontrivial_correct"] is not None]
    return {
        "n_cases": len(results),
        "n_nontrivial_cases": len(nontrivial),
        "winner_accuracy": statistics.mean(row["winner_correct"] for row in results),
        "nontrivial_winner_accuracy": statistics.mean(row["nontrivial_correct"] for row in nontrivial) if nontrivial else "",
        "median_abs_log_error": statistics.median(row["median_abs_log_error"] for row in results),
        "median_selected_to_oracle_ratio": statistics.median(row["runtime_ratio"] for row in results),
        "near_optimal_selected_rate_1_10": statistics.mean(row["runtime_ratio"] <= 1.10 for row in results),
        "median_prediction_inference_ns": statistics.median(row["prediction_inference_ns"] for row in results),
    }


def _grouped_folds(case_ids, seed):
    groups = list(case_ids)
    random.Random(seed).shuffle(groups)
    return [groups[index::FOLDS] for index in range(FOLDS)]


def _write_csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def run():
    cases = _load_cases()
    output = []
    seed_metrics = defaultdict(list)
    for seed in SEEDS:
        folds = _grouped_folds(cases.keys(), seed)
        for fold_number, validation_ids in enumerate(folds, start=1):
            validation_ids = set(validation_ids)
            train = {key: value for key, value in cases.items() if key not in validation_ids}
            validation = {key: cases[key] for key in validation_ids}
            for feature_name, feature_names in _feature_sets().items():
                for model_name in MODEL_SPECS:
                    models = _train(train, feature_names, model_name, seed)
                    fold_results = _evaluate(validation, models, feature_names)
                    metrics = _metrics(fold_results)
                    model_size = sum(len(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)) for model in models.values())
                    row = {
                        "record_type": "fold", "seed": seed, "fold": fold_number,
                        "feature_set": feature_name, "regressor": model_name,
                        "model_pickle_bytes": model_size,
                        "feature_count": len(feature_names), **metrics,
                    }
                    output.append(row)
                    seed_metrics[(feature_name, model_name, seed)].extend(fold_results)

    for (feature_name, model_name, seed), results in seed_metrics.items():
        output.append({
            "record_type": "seed_summary", "seed": seed, "fold": "",
            "feature_set": feature_name, "regressor": model_name,
            "feature_count": len(_feature_sets()[feature_name]), **_metrics(results),
        })
    for feature_name, feature_names in _feature_sets().items():
        for model_name in MODEL_SPECS:
            summaries = [
                _metrics(seed_metrics[(feature_name, model_name, seed)])
                for seed in SEEDS
            ]
            output.append({
                "record_type": "overall_summary", "seed": "ALL", "fold": "",
                "feature_set": feature_name, "regressor": model_name,
                "feature_count": len(feature_names),
                **{
                    key: statistics.mean(float(summary[key]) for summary in summaries)
                    for key in (
                        "winner_accuracy", "nontrivial_winner_accuracy",
                        "median_abs_log_error", "median_selected_to_oracle_ratio",
                        "near_optimal_selected_rate_1_10", "median_prediction_inference_ns",
                    )
                },
            })

    _write_csv(OUTPUT_PATH, output)
    config = {
        "data": str(INPUT_PATH), "cases": len(cases), "seeds": SEEDS,
        "folds": FOLDS, "target": "log median runtime",
        "feature_sets": {name: names for name, names in _feature_sets().items()},
        "models": {
            "ridge": {"alpha": 0.0001, "standardize": True},
            "random_forest": {"n_estimators": 100, "min_samples_leaf": 2, "max_features": 1.0},
            "gradient_boosting": {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 2, "loss": "huber"},
        },
        "note": "Development-only repeated grouped CV; does not change production model or use final-test outcomes.",
    }
    CONFIG_PATH.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"Saved model comparison CV to {OUTPUT_PATH}")


if __name__ == "__main__":
    run()
