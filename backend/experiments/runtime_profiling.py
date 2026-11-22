"""Profile selector stages and runtime regret without touching historical data.

Writes new runtime_profiling_*.csv files. Cold measurements use fresh graph
instances with graph caches uninitialized; warm measurements reuse an
unchanged graph after its selection features have been extracted once.
"""

import csv
import math
import statistics
from pathlib import Path
from time import perf_counter_ns

from adaptive.features import extract_selection_features
from adaptive.performance_model import BenchmarkRuntimeSelector, _vector
from adaptive.selector import eligible_algorithms
from analysis.benchmark import benchmark_graph
from graph.generator import generate_random_graph


BACKEND_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BACKEND_DIR / "results"
DEVELOPMENT_PATH = RESULTS_DIR / "development_benchmarks.csv"
STAGE_REPEATS = 31
BENCHMARK_REPEATS = 5
CASES = (
    (50, 0.05, "unit", False, True),
    (50, 0.30, "positive", False, False),
    (50, 0.30, "negative", True, True),
    (100, 0.05, "unit", True, True),
    (100, 0.30, "positive", True, False),
    (100, 0.30, "negative", True, False),
    (200, 0.05, "unit", False, False),
    (200, 0.30, "positive", True, True),
    (200, 0.30, "negative", True, True),
)


def _new_graph(case, seed):
    nodes, density, weight_mode, directed, coordinates = case
    return generate_random_graph(
        node_count=nodes,
        edge_probability=density,
        weight_mode=weight_mode,
        directed=directed,
        coordinates=coordinates,
        seed=seed,
    )


def _median_ns(samples):
    return statistics.median(samples)


def _measure(callable_, repeats=STAGE_REPEATS):
    samples = []
    for _ in range(repeats):
        started = perf_counter_ns()
        callable_()
        samples.append(perf_counter_ns() - started)
    return samples


def _baseline_prediction(model, features):
    """Original generator/sum prediction implementation for A/B timing."""
    vector = [1.0, *model._standardize(_vector(features))]
    return {
        algorithm: math.exp(max(-700, min(700, sum(
            coefficient * value
            for coefficient, value in zip(coefficients, vector)
        ))))
        for algorithm, coefficients in model.models.items()
    }


def _paired_prediction_measure(baseline, candidate):
    before, after = [], []
    for repeat in range(STAGE_REPEATS):
        order = ((baseline, before), (candidate, after))
        if repeat % 2:
            order = tuple(reversed(order))
        for callable_, output in order:
            started = perf_counter_ns()
            callable_()
            output.append(perf_counter_ns() - started)
    return _median_ns(before), _median_ns(after)


def _timed_stage_samples(case, case_index, graph, model):
    source, destination = "0", str(case[0] - 1)
    features = extract_selection_features(graph)
    eligible = eligible_algorithms(graph, source, destination, features)
    predictions = model.predict_runtimes_from_features(features)
    baseline_predictions = _baseline_prediction(model, features)
    if predictions.keys() != baseline_predictions.keys() or any(
        not math.isclose(
            predictions[name], baseline_predictions[name],
            rel_tol=1e-12, abs_tol=1e-15,
        )
        for name in predictions
    ):
        raise AssertionError("Prediction optimization changed model outputs")
    candidates = [name for name in eligible if name in predictions]
    baseline_prediction_ns, optimized_prediction_ns = (
        _paired_prediction_measure(
            lambda: _baseline_prediction(model, features),
            lambda: model.predict_runtimes_from_features(features),
        )
    )

    warm = {
        "feature_extraction": _measure(
            lambda: extract_selection_features(graph)
        ),
        "eligibility": _measure(
            lambda: eligible_algorithms(graph, source, destination, features)
        ),
        "prediction": _measure(
            lambda: model.predict_runtimes_from_features(features)
        ),
        "winner_comparison": _measure(
            lambda: min(candidates, key=predictions.__getitem__)
        ),
        "complete_selection": _measure(
            lambda: model.choose_algorithm(graph, source, destination)
        ),
    }

    cold_features = []
    cold_complete = []
    for repeat in range(STAGE_REPEATS):
        fresh = _new_graph(case, 900000 + case_index * 100 + repeat)
        started = perf_counter_ns()
        extract_selection_features(fresh)
        cold_features.append(perf_counter_ns() - started)

        fresh = _new_graph(case, 910000 + case_index * 100 + repeat)
        started = perf_counter_ns()
        model.choose_algorithm(fresh, source, destination)
        cold_complete.append(perf_counter_ns() - started)

    return {
        **{f"warm_{key}_median_ns": _median_ns(value)
           for key, value in warm.items()},
        "cold_feature_extraction_median_ns": _median_ns(cold_features),
        "cold_complete_selection_median_ns": _median_ns(cold_complete),
        "baseline_prediction_median_ns": baseline_prediction_ns,
        "optimized_prediction_median_ns": optimized_prediction_ns,
        "prediction_optimization_change_percent": (
            (optimized_prediction_ns / baseline_prediction_ns - 1) * 100
        ),
        "stage_repeats": STAGE_REPEATS,
    }


def _write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def run():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model = BenchmarkRuntimeSelector.from_csv(DEVELOPMENT_PATH)
    profile_rows = []
    benchmark_rows = []

    for index, case in enumerate(CASES):
        graph = _new_graph(case, 7000 + index)
        source, destination = "0", str(case[0] - 1)
        stage_metrics = _timed_stage_samples(case, index, graph, model)

        measured = benchmark_graph(
            graph,
            source,
            destination,
            repeats=BENCHMARK_REPEATS,
            performance_model=model,
            metadata={"profile_case": f"profile-{index + 1:02d}"},
        )
        benchmark_rows.extend(measured)

        adaptive = next(row for row in measured if row["algorithm"] == "adaptive")
        algorithms = [row for row in measured if row["algorithm"] != "adaptive"]
        fastest = min(algorithms, key=lambda row: row["median_seconds"])
        chosen = next(
            row for row in algorithms
            if row["algorithm"] == adaptive["selected_algorithm"]
        )
        adaptive_seconds = adaptive["median_seconds"]
        oracle_seconds = fastest["median_seconds"]
        selected_seconds = chosen["median_seconds"]

        profile_rows.append({
            "profile_case": f"profile-{index + 1:02d}",
            "seed": 7000 + index,
            "nodes": case[0],
            "target_density": case[1],
            "weight_mode": case[2],
            "directed": case[3],
            "coordinates": case[4],
            "edges": graph.edge_count,
            "selected_algorithm": adaptive["selected_algorithm"],
            "oracle_algorithm": fastest["algorithm"],
            **stage_metrics,
            "adaptive_end_to_end_median_ns": round(adaptive_seconds * 1e9),
            "selected_algorithm_median_ns": round(selected_seconds * 1e9),
            "oracle_algorithm_median_ns": round(oracle_seconds * 1e9),
            "estimated_selection_overhead_ns": round(
                float(adaptive["estimated_selection_overhead_seconds"]) * 1e9
            ),
            "total_runtime_regret_ns": round((adaptive_seconds - oracle_seconds) * 1e9),
            "suboptimal_algorithm_regret_ns": round((selected_seconds - oracle_seconds) * 1e9),
            "path_cost": adaptive["path_cost"],
            "correctness_verified": True,
            "benchmark_repeats": BENCHMARK_REPEATS,
        })

        print(f"Profiled {index + 1}/{len(CASES)} cases")

    _write_csv(RESULTS_DIR / "runtime_profiling.csv", profile_rows)
    _write_csv(RESULTS_DIR / "runtime_profiling_benchmarks.csv", benchmark_rows)

    summary = []
    for field in (
        "cold_feature_extraction_median_ns",
        "warm_feature_extraction_median_ns",
        "warm_eligibility_median_ns",
        "warm_prediction_median_ns",
        "baseline_prediction_median_ns",
        "optimized_prediction_median_ns",
        "warm_winner_comparison_median_ns",
        "cold_complete_selection_median_ns",
        "warm_complete_selection_median_ns",
        "adaptive_end_to_end_median_ns",
        "selected_algorithm_median_ns",
        "oracle_algorithm_median_ns",
        "estimated_selection_overhead_ns",
        "total_runtime_regret_ns",
        "suboptimal_algorithm_regret_ns",
    ):
        values = [int(row[field]) for row in profile_rows]
        summary.append({
            "metric": field,
            "case_count": len(values),
            "median_ns": statistics.median(values),
            "p90_ns": sorted(values)[int(0.9 * (len(values) - 1))],
            "minimum_ns": min(values),
            "maximum_ns": max(values),
        })
    _write_csv(RESULTS_DIR / "runtime_profiling_summary.csv", summary)
    print(f"Saved runtime profiling results to {RESULTS_DIR}")


if __name__ == "__main__":
    run()
