"""Summarize selector quality and end-to-end runtime from benchmark records."""

import csv
import statistics
from collections import defaultdict
from pathlib import Path


def summarize_selection(rows: list[dict]) -> list[dict]:
    cases = defaultdict(list)
    for row in rows:
        cases[row["case_id"]].append(row)

    groups = defaultdict(list)
    for case_id, case_rows in cases.items():
        baseline = [row for row in case_rows if row["algorithm"] != "adaptive"]
        adaptive = next((row for row in case_rows if row["algorithm"] == "adaptive"), None)
        if not baseline or adaptive is None:
            continue
        fastest = min(baseline, key=lambda row: float(row["median_seconds"]))
        best_time = float(fastest["median_seconds"])
        adaptive_time = float(adaptive["median_seconds"])
        category = (
            adaptive.get("split", "unspecified"),
            adaptive.get("selector_policy", "rules"),
            adaptive.get("weight_mode", "unspecified"),
            adaptive.get("coordinates", "unspecified"),
            adaptive.get("directed", "unspecified"),
            adaptive.get("nodes", "unspecified"),
            adaptive.get("target_density", adaptive.get("density", "unspecified")),
        )
        groups[category].append({
            "selected_best": adaptive.get("selected_algorithm") == fastest["algorithm"],
            "runtime_ratio": adaptive_time / best_time if best_time else 1.0,
            "observed_density": float(adaptive.get("density", 0.0)),
        })

    summaries = []
    for category, values in sorted(groups.items()):
        split, policy, weight_mode, coordinates, directed, nodes, target_density = category
        ratios = [item["runtime_ratio"] for item in values]
        summaries.append({
            "split": split,
            "selector_policy": policy,
            "weight_mode": weight_mode,
            "coordinates": coordinates,
            "directed": directed,
            "nodes": nodes,
            "target_density": target_density,
            "mean_observed_density": statistics.mean(
                item["observed_density"] for item in values
            ),
            "graph_cases": len(values),
            "selected_fastest_algorithm_rate": sum(item["selected_best"] for item in values) / len(values),
            "median_adaptive_to_best_runtime_ratio": statistics.median(ratios),
            "adaptive_faster_than_best_rate": sum(ratio < 1 for ratio in ratios) / len(ratios),
        })
    return summaries


def save_selection_summary(rows: list[dict], output_path: str | Path) -> None:
    summaries = summarize_selection(rows)
    if not summaries:
        raise ValueError("No adaptive benchmark records to summarize")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
