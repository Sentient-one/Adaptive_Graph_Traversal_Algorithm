from analysis.benchmark import benchmark_graph, save_results_csv
from graph.generator import generate_random_graph
from pathlib import Path


def main() -> None:
    node_counts = [50, 100, 200]
    densities = [0.05, 0.10, 0.20, 0.40]
    weight_types = [False, True]  # False = unit weights; True = random positive weights
    graph_seeds = [10, 20, 30]
    repeats = 7

    all_rows = []

    for node_count in node_counts:
        for density in densities:
            for weighted in weight_types:
                for seed in graph_seeds:
                    graph = generate_random_graph(
                        node_count=node_count,
                        edge_probability=density,
                        weighted=weighted,
                        directed=False,
                        seed=seed,
                    )

                    rows = benchmark_graph(
                        graph=graph,
                        source="0",
                        destination=str(node_count - 1),
                        repeats=repeats,
                    )
                    all_rows.extend(rows)

                    print(
                        f"Finished: nodes={node_count}, density={density}, "
                        f"weighted={weighted}, seed={seed}"
                    )

    output_file = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "benchmark_results.csv"
    )
    save_results_csv(all_rows, output_file)
  
    output_file.parent.mkdir(parents=True, exist_ok=True)
    save_results_csv(all_rows, str(output_file))
    print(f"\nSaved {len(all_rows)} rows to {output_file}")


if __name__ == "__main__":
    main()