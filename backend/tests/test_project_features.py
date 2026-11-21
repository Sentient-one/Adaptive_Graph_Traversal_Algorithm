
import math
import csv
import tempfile
import unittest

from adaptive.features import extract_features
from adaptive.performance_model import (
    BenchmarkRuntimeSelector,
    MODEL_FEATURES,
    RIDGE_ALPHA,
    _vector,
)
from adaptive.selector import find_shortest_path
from analysis.benchmark import benchmark_graph, save_results_csv
from analysis.reporting import summarize_selection
from adaptive.validity import validate_algorithm_input
from algorithms import bellman_ford
from graph.generator import generate_random_graph
from graph.graph import Graph


class ProjectFeatureTests(unittest.TestCase):

    def test_production_model_uses_model_b_features_and_selected_alpha(self):
        self.assertEqual(
            MODEL_FEATURES,
            (
                "nodes",
                "density",
                "directed",
                "uniform_weights",
                "has_negative_weights",
                "weight_stddev",
                "positions_available",
                "heuristic_scale",
            ),
        )
        self.assertEqual(RIDGE_ALPHA, 1e-4)

        features = {
            "node_count": 10,
            "density": 0.2,
            "directed": False,
            "uniform_weights": True,
            "has_negative_weights": False,
            "average_degree": 123,
            "weight_stddev": 0.5,
            "positions_available": False,
            "heuristic_scale": None,
        }
        changed_unused_feature = dict(features, average_degree=999)
        self.assertEqual(len(_vector(features)), 8)
        self.assertEqual(_vector(features), _vector(changed_unused_feature))

    def test_model_trains_all_algorithms_and_predicts_positive_finite_values(self):
        rows = [
            {
                "case_id": "case-1",
                "algorithm": algorithm,
                "median_seconds": runtime,
                "nodes": 10,
                "density": 0.2,
                "directed": False,
                "uniform_weights": False,
                "has_negative_weights": False,
                "weight_stddev": 0.5,
                "positions_available": False,
                "heuristic_scale": "",
            }
            for algorithm, runtime in (
                ("bfs", 0.001),
                ("dijkstra", 0.002),
                ("astar", 0.003),
                ("bellman_ford", 0.01),
            )
        ]

        model = BenchmarkRuntimeSelector(rows)
        predictions = model.predict_runtimes_from_features(rows[0])

        self.assertEqual(model.ridge, RIDGE_ALPHA)
        self.assertEqual(
            set(model.models),
            {"bfs", "dijkstra", "astar", "bellman_ford"},
        )
        self.assertEqual(set(predictions), set(model.models))
        self.assertTrue(
            all(math.isfinite(value) and value > 0
                for value in predictions.values())
        )

    def test_selector_uses_lowest_predicted_eligible_algorithm(self):
        rows = [
            {
                "case_id": "case-1",
                "algorithm": algorithm,
                "median_seconds": runtime,
                "nodes": 3,
                "density": 1,
                "directed": False,
                "uniform_weights": False,
                "has_negative_weights": False,
                "weight_stddev": 1,
                "positions_available": False,
                "heuristic_scale": "",
            }
            for algorithm, runtime in (
                ("bfs", 0.001),
                ("dijkstra", 0.002),
                ("astar", 0.003),
                ("bellman_ford", 0.01),
            )
        ]
        model = BenchmarkRuntimeSelector(rows)
        graph = Graph()
        graph.set_position("A", 0, 0)
        graph.set_position("B", 1, 0)
        graph.set_position("C", 2, 0)
        graph.add_edge("A", "B", 2)
        graph.add_edge("B", "C", 3)
        graph.add_edge("A", "C", 8)

        model.predict_runtimes = lambda graph, features=None: {
            "bfs": 0.01,
            "dijkstra": 0.1,
            "astar": 0.001,
            "bellman_ford": 0.2,
        }

        # BFS is ineligible for non-uniform weights; A* has the lowest
        # prediction among the remaining eligible algorithms.
        self.assertEqual(model.choose_algorithm(graph, "A", "C"), "astar")

    def test_unavailable_training_models_fall_back_safely(self):
        model = BenchmarkRuntimeSelector([
            {
                "case_id": "case-1",
                "algorithm": "bfs",
                "median_seconds": 0.001,
                "nodes": 2,
                "density": 1,
                "directed": True,
                "uniform_weights": False,
                "has_negative_weights": False,
                "weight_stddev": 0,
                "positions_available": False,
                "heuristic_scale": "",
            }
        ])
        graph = Graph(directed=True)
        graph.add_edge("A", "B", -1)

        # BFS has training data but is not valid for negative edges. With no
        # trained eligible model, selection falls back to Bellman-Ford.
        self.assertEqual(
            model.choose_algorithm(graph, "A", "B"),
            "bellman_ford",
        )

    def test_model_rejects_empty_or_unusable_training_data(self):
        with self.assertRaisesRegex(ValueError, "No algorithm benchmark rows"):
            BenchmarkRuntimeSelector([])
        with self.assertRaisesRegex(ValueError, "No algorithm benchmark rows"):
            BenchmarkRuntimeSelector([
                {"algorithm": "bfs", "median_seconds": "0"}
            ])

    def test_graph_rejects_nonfinite_weights_and_coordinates(self):
        graph = Graph()
        for value in (math.nan, math.inf, -math.inf, "not-a-number"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                graph.add_edge("A", "B", value)

        for value in (math.nan, math.inf, "bad"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                graph.set_position("A", value, 0)

    def test_graph_rejects_self_loops_and_duplicate_edges(self):
        graph = Graph()
        with self.assertRaisesRegex(ValueError, "Self-loops"):
            graph.add_edge("A", "A")

        graph.add_edge("A", "B")
        with self.assertRaisesRegex(ValueError, "Duplicate edge"):
            graph.add_edge("B", "A")

    def test_cached_weight_summary_updates_as_graph_changes(self):
        graph = Graph()
        graph.add_edge("A", "B", 2)

        self.assertTrue(graph.has_uniform_nonnegative_weights())

        graph.add_edge("B", "C", 3)
        self.assertFalse(graph.has_uniform_nonnegative_weights())

        graph.add_edge("C", "D", -1)

        self.assertTrue(graph.has_negative_weights())
        self.assertEqual(graph.minimum_weight, -1)
        self.assertEqual(graph.edge_count, 3)

    def test_features_include_distribution_degree_and_heuristic_fields(self):
        graph = Graph()
        graph.set_position("A", 0, 0)
        graph.set_position("B", 3, 0)
        graph.add_edge("A", "B", 4)

        features = extract_features(graph)

        self.assertEqual(features["density"], 1.0)
        self.assertEqual(features["average_degree"], 1.0)
        self.assertEqual(features["weight_stddev"], 0.0)
        self.assertTrue(features["positions_available"])
        self.assertGreater(features["heuristic_scale"], 0)

    # --------------------------------------------------
    # NEW TESTS: GRAPH FEATURE EXTRACTION
    # --------------------------------------------------

    def test_features_for_disconnected_graph(self):
        graph = Graph()
        graph.add_edge("A", "B", 2)
        graph.add_edge("B", "C", 4)
        graph.add_node("D")

        features = extract_features(graph, include_heuristic=False)

        self.assertEqual(features["node_count"], 4)
        self.assertEqual(features["edge_count"], 2)
        self.assertAlmostEqual(features["density"], 1 / 3)
        self.assertEqual(features["component_count"], 2)
        self.assertFalse(features["is_connected"])
        self.assertEqual(features["isolated_node_count"], 1)

    def test_features_for_directed_graph(self):
        graph = Graph(directed=True)
        graph.add_edge("A", "B", 3)
        graph.add_edge("A", "C", 5)
        graph.add_edge("B", "C", 2)

        features = extract_features(graph, include_heuristic=False)

        self.assertTrue(features["directed"])
        self.assertEqual(features["edge_count"], 3)
        self.assertAlmostEqual(features["density"], 0.5)
        self.assertEqual(features["average_out_degree"], 1)
        self.assertEqual(features["average_in_degree"], 1)
        self.assertEqual(features["component_count"], 1)
        self.assertIsNone(features["is_connected"])
        self.assertTrue(features["is_weakly_connected"])

    def test_features_for_unweighted_graph(self):
        graph = Graph()
        graph.add_edge("A", "B")
        graph.add_edge("B", "C")

        features = extract_features(graph, include_heuristic=False)

        self.assertFalse(features["is_weighted"])
        self.assertTrue(features["all_unit_weights"])

    def test_features_for_empty_graph(self):
        graph = Graph()

        features = extract_features(graph, include_heuristic=False)

        self.assertEqual(features["node_count"], 0)
        self.assertEqual(features["edge_count"], 0)
        self.assertEqual(features["density"], 0)
        self.assertEqual(features["component_count"], 0)
        self.assertFalse(features["is_connected"])
        self.assertEqual(features["isolated_node_count"], 0)

    def test_features_for_single_node_graph(self):
        graph = Graph()
        graph.add_node("A")

        features = extract_features(graph, include_heuristic=False)

        self.assertEqual(features["node_count"], 1)
        self.assertEqual(features["density"], 0)
        self.assertEqual(features["component_count"], 1)
        self.assertEqual(features["isolated_node_count"], 1)

    # --------------------------------------------------
    # EXISTING TESTS
    # --------------------------------------------------

    def test_heuristic_scale_cache_invalidates_after_graph_changes(self):
        graph = Graph()
        graph.set_position("A", 0, 0)
        graph.set_position("B", 2, 0)
        graph.set_position("C", 3, 0)

        graph.add_edge("A", "B", 4)
        self.assertEqual(graph.heuristic_scale(), 2.0)

        graph.add_edge("B", "C", 0.5)
        self.assertEqual(graph.heuristic_scale(), 0.5)

        graph.set_position("C", 4, 0)
        self.assertEqual(graph.heuristic_scale(), 0.25)

    def test_validation_rejects_unknown_algorithms(self):
        graph = Graph()
        graph.add_edge("A", "B")

        with self.assertRaisesRegex(ValueError, "Unknown algorithm"):
            validate_algorithm_input(graph, "A", "B", "floyd-warshall")

    def test_bellman_ford_reports_work_counters(self):
        graph = Graph(directed=True)
        graph.add_edge("A", "B", -1)
        graph.add_edge("B", "C", 3)

        result = bellman_ford.shortest_path(graph, "A", "C")

        self.assertIsNone(result["explored_nodes"])
        self.assertGreater(result["passes_executed"], 0)
        self.assertGreater(result["relaxation_attempts"], 0)
        self.assertGreater(result["successful_relaxations"], 0)

    def test_negative_generator_produces_finite_paths_without_negative_cycles(self):
        graph = generate_random_graph(
            5,
            edge_probability=1,
            directed=True,
            seed=5,
            weight_mode="negative",
            coordinates=False,
        )

        self.assertTrue(graph.has_negative_weights())

        result = bellman_ford.shortest_path(graph, "0", "4")
        self.assertIsNotNone(result["cost"])

    def test_runtime_model_selects_fastest_valid_candidate_from_development_rows(self):
        rows = [
            {
                "case_id": "dev-1",
                "algorithm": "bfs",
                "median_seconds": "0.001",
                "nodes": "10",
                "density": "0.2",
                "directed": "False",
                "uniform_weights": "True",
                "has_negative_weights": "False",
                "average_degree": "2",
                "weight_stddev": "0",
                "positions_available": "False",
                "heuristic_scale": "",
            },
            {
                "case_id": "dev-1",
                "algorithm": "dijkstra",
                "median_seconds": "0.002",
                "nodes": "10",
                "density": "0.2",
                "directed": "False",
                "uniform_weights": "True",
                "has_negative_weights": "False",
                "average_degree": "2",
                "weight_stddev": "0",
                "positions_available": "False",
                "heuristic_scale": "",
            },
            {
                "case_id": "dev-1",
                "algorithm": "bellman_ford",
                "median_seconds": "0.01",
                "nodes": "10",
                "density": "0.2",
                "directed": "False",
                "uniform_weights": "True",
                "has_negative_weights": "False",
                "average_degree": "2",
                "weight_stddev": "0",
                "positions_available": "False",
                "heuristic_scale": "",
            },
        ]

        model = BenchmarkRuntimeSelector(rows, ridge=1e-3)
        graph = Graph()
        graph.add_edge("A", "B")

        result = find_shortest_path(
            graph, "A", "B", performance_model=model
        )

        self.assertEqual(result["algorithm"], "bfs")
        self.assertEqual(result["cost"], 1)

    def test_benchmark_model_runs_on_separate_heldout_case(self):
        development_graph = generate_random_graph(
            8,
            edge_probability=0.6,
            seed=1,
            weight_mode="unit",
            coordinates=False,
        )

        development_rows = benchmark_graph(
            development_graph,
            "0",
            "7",
            repeats=1,
            metadata={"case_id": "dev-1", "split": "development"},
        )

        model = BenchmarkRuntimeSelector(
            development_rows, ridge=1e-3
        )

        heldout_graph = generate_random_graph(
            8,
            edge_probability=0.6,
            seed=2,
            weight_mode="positive",
            coordinates=False,
        )

        heldout_rows = benchmark_graph(
            heldout_graph,
            "0",
            "7",
            repeats=1,
            performance_model=model,
            metadata={"case_id": "heldout-1", "split": "heldout"},
        )

        adaptive_row = next(
            row for row in heldout_rows
            if row["algorithm"] == "adaptive"
        )

        self.assertEqual(
            adaptive_row["selector_policy"], "ridge_regression"
        )
        self.assertTrue(adaptive_row["selected_algorithm"])

        summary = summarize_selection(
            development_rows + heldout_rows
        )

        self.assertEqual(
            sum(row["graph_cases"] for row in summary), 2
        )

        # Verify both feature extraction measurements reach the summary.
        for summary_row in summary:
            self.assertIn(
                "median_full_feature_extraction_seconds",
                summary_row,
            )
            self.assertIn(
                "median_selection_feature_extraction_seconds",
                summary_row,
            )

            self.assertGreater(
                summary_row["median_full_feature_extraction_seconds"],
                0,
            )
            self.assertGreater(
                summary_row["median_selection_feature_extraction_seconds"],
                0,
            )

        with tempfile.TemporaryDirectory() as directory:
            output = f"{directory}/mixed_rows.csv"

            save_results_csv(heldout_rows, output)

            with open(
                output, newline="", encoding="utf-8"
            ) as file:
                written = list(csv.DictReader(file))

            self.assertEqual(
                written[-1]["selector_policy"], "ridge_regression"
            )

    # --------------------------------------------------
    # NEW TESTS: BENCHMARK INSTRUMENTATION
    # --------------------------------------------------

    def test_benchmark_records_adaptive_timing_and_counters(self):
        graph = generate_random_graph(
            node_count=10,
            edge_probability=0.4,
            seed=42,
            weight_mode="positive",
            coordinates=True,
        )

        rows = benchmark_graph(
            graph,
            "0",
            "9",
            repeats=3,
            metadata={"case_id": "measurement-test"},
        )

        adaptive = next(
            row for row in rows if row["algorithm"] == "adaptive"
        )

        required_fields = [
            "median_seconds",
            "minimum_seconds",
            "maximum_seconds",
            "runtime_stddev_seconds",

            # Feature extraction measurements
            "median_full_feature_extraction_seconds",
            "median_selection_feature_extraction_seconds",
            "median_feature_extraction_seconds",

            # Selection measurements
            "median_selection_seconds",
            "median_selected_algorithm_seconds",
            "estimated_selection_overhead_seconds",

            # Algorithm counters
            "median_explored_nodes",
            "median_edge_scans",
            "median_relaxation_attempts",
        ]

        for field in required_fields:
            self.assertIn(field, adaptive)
            self.assertNotEqual(adaptive[field], "")

        self.assertGreater(adaptive["median_seconds"], 0)
        self.assertGreater(adaptive["median_selection_seconds"], 0)

        # Full feature extraction.
        self.assertGreater(
            adaptive["median_full_feature_extraction_seconds"], 0
        )

        # Lightweight feature extraction used by the model.
        self.assertGreater(
            adaptive["median_selection_feature_extraction_seconds"], 0
        )

        # Backward compatibility.
        self.assertGreater(
            adaptive["median_feature_extraction_seconds"], 0
        )

        self.assertGreaterEqual(adaptive["median_edge_scans"], 0)

        self.assertLessEqual(
            adaptive["minimum_seconds"],
            adaptive["median_seconds"],
        )

        self.assertLessEqual(
            adaptive["median_seconds"],
            adaptive["maximum_seconds"],
        )

    def test_benchmark_runtime_statistics_exist_for_all_algorithms(self):
        graph = generate_random_graph(
            node_count=10,
            edge_probability=0.4,
            seed=21,
            weight_mode="unit",
            coordinates=False,
        )

        rows = benchmark_graph(
            graph,
            "0",
            "9",
            repeats=3,
        )

        for row in rows:
            self.assertIn("median_seconds", row)
            self.assertIn("minimum_seconds", row)
            self.assertIn("maximum_seconds", row)
            self.assertIn("runtime_stddev_seconds", row)

            self.assertGreater(row["median_seconds"], 0)


if __name__ == "__main__":
    unittest.main()
