import unittest

from graph.graph import Graph
from adaptive.selector import choose_algorithm, find_shortest_path
from algorithms import astar,bfs, dijkstra, bellman_ford


class AlgorithmTests(unittest.TestCase):
    def test_bfs_finds_minimum_hop_path(self):
        graph = Graph(directed=False)
        graph.add_edge("A", "B")
        graph.add_edge("A", "C")
        graph.add_edge("B", "D")
        graph.add_edge("C", "D")

        result = bfs.shortest_path(graph, "A", "D")

        self.assertEqual(result["cost"], 2)
        self.assertEqual(len(result["path"]), 3)
        self.assertEqual(result["path"][0], "A")
        self.assertEqual(result["path"][-1], "D")

    def test_dijkstra_finds_minimum_cost_path(self):
        graph = Graph(directed=False)
        graph.add_edge("A", "B", 4)
        graph.add_edge("A", "C", 2)
        graph.add_edge("C", "B", 1)
        graph.add_edge("B", "D", 3)
        graph.add_edge("C", "D", 7)

        result = dijkstra.shortest_path(graph, "A", "D")

        self.assertEqual(result["cost"], 6)
        self.assertEqual(result["path"], ["A", "C", "B", "D"])

    def test_selector_chooses_bfs_for_unit_weights(self):
        graph = Graph()
        graph.add_edge("A", "B")

        self.assertEqual(choose_algorithm(graph), "bfs")

    def test_selector_chooses_bfs_for_equal_positive_weights(self):
        graph = Graph()
        graph.add_edge("A", "B", 2)

        self.assertEqual(choose_algorithm(graph), "bfs")

    def test_bfs_supports_equal_positive_weights(self):
        graph = Graph()
        graph.add_edge("A", "B", 2)
        graph.add_edge("A", "C", 2)
        graph.add_edge("B", "D", 2)
        graph.add_edge("C", "D", 2)

        result = bfs.shortest_path(graph, "A", "D")

        self.assertEqual(result["cost"], 4)
        self.assertEqual(len(result["path"]), 3)

    def test_selector_chooses_bellman_ford_for_negative_weights(self):
        graph = Graph(directed=True)
        graph.add_edge("A", "B", -2)

        self.assertEqual(choose_algorithm(graph), "bellman_ford")

    def test_adaptive_result_includes_selected_algorithm(self):
        graph = Graph()
        graph.add_edge("A", "B")

        result = find_shortest_path(graph, "A", "B")

        self.assertEqual(result["algorithm"], "bfs")
        self.assertEqual(result["path"], ["A", "B"])
        self.assertEqual(result["cost"], 1)

    def test_unreachable_destination_returns_no_path(self):
        graph = Graph(directed=True)
        graph.add_edge("A", "B")
        graph.add_node("C")

        result = dijkstra.shortest_path(graph, "A", "C")

        self.assertIsNone(result["path"])
        self.assertIsNone(result["cost"])

    def test_bellman_ford_handles_negative_edge(self):
        graph = Graph(directed=True)
        graph.add_edge("A", "B", 4)
        graph.add_edge("A", "C", 5)
        graph.add_edge("B", "C", -2)

        result = bellman_ford.shortest_path(graph, "A", "C")

        self.assertEqual(result["cost"], 2)
        self.assertEqual(result["path"], ["A", "B", "C"])

    def test_bellman_ford_detects_reachable_negative_cycle(self):
        graph = Graph(directed=True)
        graph.add_edge("A", "B", 1)
        graph.add_edge("B", "C", -2)
        graph.add_edge("C", "B", -2)

        with self.assertRaisesRegex(ValueError, "negative-weight cycle"):
            bellman_ford.shortest_path(graph, "A", "C")
    
    from algorithms import astar


def test_astar_matches_dijkstra():
    graph = Graph(directed=False)
    graph.set_position("A", 0, 0)
    graph.set_position("B", 1, 0)
    graph.set_position("C", 0, 1)
    graph.set_position("D", 1, 1)

    graph.add_edge("A", "B", 1)
    graph.add_edge("A", "C", 1)
    graph.add_edge("B", "D", 1)
    graph.add_edge("C", "D", 1)

    astar_result = astar.shortest_path(graph, "A", "D")
    dijkstra_result = dijkstra.shortest_path(graph, "A", "D")

    assert astar_result["cost"] == dijkstra_result["cost"]
    assert astar_result["path"][0] == "A"
    assert astar_result["path"][-1] == "D"


if __name__ == "__main__":
    unittest.main()
