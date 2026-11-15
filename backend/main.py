from graph.graph import Graph
from adaptive.features import extract_features
from adaptive.selector import find_shortest_path
from analysis.benchmark import benchmark_graph, save_results_csv


def main() -> None:
    graph = Graph(directed=False)
    graph.add_edge("A", "B", 4)
    graph.add_edge("A", "C", 2)
    graph.add_edge("C", "B", 1)
    graph.add_edge("B", "D", 3)
    graph.add_edge("C", "D", 7)

    print("Graph features:")
    print(extract_features(graph))

    print("\nAdaptive shortest path:")
    print(find_shortest_path(graph, "A", "D"))

    rows = benchmark_graph(graph, "A", "D")
    save_results_csv(rows, "results/benchmark.csv")
    print("\nSaved benchmark results to results/benchmark.csv")


if __name__ == "__main__":
    main()