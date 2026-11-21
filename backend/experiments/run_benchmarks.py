"""Run reproducible development and held-out selector experiments."""

import csv
import json
from pathlib import Path

from adaptive.performance_model import BenchmarkRuntimeSelector
from analysis.benchmark import benchmark_graph, save_results_csv
from analysis.reporting import save_selection_summary
from graph.generator import generate_random_graph


BACKEND_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BACKEND_DIR / "results"
NODE_COUNTS = (50, 100, 200)
DENSITIES = (0.05, 0.20, 0.40)
WEIGHT_MODES = ("unit", "uniform", "positive", "negative")
COORDINATE_OPTIONS = (False, True)
DIRECTED_OPTIONS = (False, True)
SPLIT_SEEDS = {
    "development": (11, 22),
    "heldout": (101, 202),
}
REPEATS = 5
RIDGE_ALPHA = 1e-4


def _cases(split: str):
    for node_count in NODE_COUNTS:
        for density in DENSITIES:
            for weight_mode in WEIGHT_MODES:
                directed_options = (True,) if weight_mode == "negative" else DIRECTED_OPTIONS
                for directed in directed_options:
                    for coordinates in COORDINATE_OPTIONS:
                        for seed in SPLIT_SEEDS[split]:
                            case_id = (
                                f"{split}-n{node_count}-d{density:.2f}-"
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
                                "split": split,
                                "seed": seed,
                                "weight_mode": weight_mode,
                                "target_density": density,
                                "coordinates": coordinates,
                            }
                            yield graph, metadata


def _run_split(split: str, performance_model=None) -> list[dict]:
    all_rows = []
    for case_number, (graph, metadata) in enumerate(_cases(split), start=1):
        rows = benchmark_graph(
            graph,
            source="0",
            destination=str(len(graph.adjacency) - 1),
            repeats=REPEATS,
            performance_model=performance_model,
            metadata=metadata,
        )
        all_rows.extend(rows)
        if case_number % 25 == 0:
            print(f"{split}: completed {case_number} graph cases")
    print(f"{split}: completed {case_number} graph cases")
    output_path = RESULTS_DIR / f"{split}_benchmarks.csv"
    save_results_csv(all_rows, output_path)
    print(f"Saved {len(all_rows)} rows to {output_path}")
    return all_rows


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config = {
        "node_counts": NODE_COUNTS,
        "densities": DENSITIES,
        "weight_modes": WEIGHT_MODES,
        "coordinate_options": COORDINATE_OPTIONS,
        "directed_options": DIRECTED_OPTIONS,
        "split_seeds": SPLIT_SEEDS,
        "repeats": REPEATS,
        "selector_model": "ridge_regression",
        "ridge_alpha": RIDGE_ALPHA,
    }
    with (RESULTS_DIR / "experiment_config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)

    development_rows = _run_split("development")
    performance_model = BenchmarkRuntimeSelector(
        development_rows,
        ridge=RIDGE_ALPHA,
    )

    heldout_rows = _run_split("heldout", performance_model=performance_model)
    combined_rows = development_rows + heldout_rows
    save_results_csv(combined_rows, RESULTS_DIR / "benchmark_results.csv")
    save_selection_summary(combined_rows, RESULTS_DIR / "selection_summary.csv")

    print("\nHeld-out summary by graph category:")
    with (RESULTS_DIR / "selection_summary.csv").open(
        newline="", encoding="utf-8"
    ) as file:
        for row in csv.DictReader(file):
            if row["split"] == "heldout":
                print(
                    f"{row['weight_mode']}, directed={row['directed']}, "
                    f"coordinates={row['coordinates']}: "
                    f"median adaptive/best runtime="
                    f"{float(row['median_adaptive_to_best_runtime_ratio']):.3f}, "
                    f"best-choice rate="
                    f"{float(row['selected_fastest_algorithm_rate']):.1%}"
                )
    print(f"\nSummary saved to {RESULTS_DIR / 'selection_summary.csv'}")


if __name__ == "__main__":
    main()
