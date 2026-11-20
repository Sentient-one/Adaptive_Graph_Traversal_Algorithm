# ADSA: Adaptive Shortest-Path Algorithms

This project compares BFS, Dijkstra, A*, and Bellman–Ford, then selects a
shortest-path algorithm using graph eligibility rules or a runtime model trained
from benchmark cases.

## Requirements and setup

Python 3.10 or newer is recommended. The implementation and tests use only the
Python standard library plus pytest for the test suite.

```bash
cd /Users/mac/Desktop/MyProjects/ADSA_project
source .venv/bin/activate
```

## Run the example

Run from `backend` so its package imports resolve:

```bash
cd /Users/mac/Desktop/MyProjects/ADSA_project/backend
../.venv/bin/python main.py
```

The example prints graph features and the selected path, then overwrites
`backend/results/benchmark.csv` with fresh example measurements.

## Run tests

From the project root:

```bash
.venv/bin/python -m pytest -p no:cacheprovider
```

## Run the development and held-out benchmark study

Run from `backend`:

```bash
cd /Users/mac/Desktop/MyProjects/ADSA_project/backend
../.venv/bin/python -m experiments.run_benchmarks
```

The study uses separate fixed seed sets for development and held-out graph
cases. It trains a nearest-neighbor runtime selector from development results,
then evaluates a ridge-regression runtime model (alpha 0.001) on held-out cases.
The exact graph grid, seeds, repeat count, and model setting are saved in
`experiment_config.json`. Outputs are written to `backend/results/`:

- `development_benchmarks.csv`: per-algorithm and rule-selector measurements
- `heldout_benchmarks.csv`: per-algorithm and benchmark-model measurements
- `benchmark_results.csv`: combined development and held-out measurements
- `selection_summary.csv`: selection accuracy and end-to-end runtime ratio by
  graph category
- `experiment_config.json`: settings needed to reproduce the run

The runtime ratio compares adaptive end-to-end time, including selection, with
the fastest eligible individual algorithm in that case. A ratio below 1 means
the adaptive run was faster in that measurement; a ratio above 1 means it was
slower. These timings are machine- and workload-dependent.

## Algorithm and graph assumptions

- The graph is simple: self-loops and duplicate edges are rejected. Build and
  update graphs through `Graph.add_node`, `Graph.add_edge`, and
  `Graph.set_position` so its cached weight summaries stay correct.
- Edge weights and coordinates must be finite numbers.
- BFS supports equal nonnegative edge weights.
- Dijkstra and A* reject negative weights.
- Bellman–Ford detects reachable negative-weight cycles.
- A* uses a scaled Euclidean heuristic. The scale is bounded by every edge's
  weight-to-distance ratio, making the heuristic admissible and consistent for
  the graph's supplied coordinates and nonnegative weights.
- Random negative-weight cases are directed and use positive cycle potentials,
  so the generator creates negative edges without creating negative cycles.

The rule selector is a transparent baseline. The benchmark-trained regression
selector is an experimental model; its held-out results must be reviewed before
claiming it is better than the rule selector or any fixed algorithm.

## Research paper status

The implementation now supports a reproducible comparison workflow, but the
paper still needs a literature review, final held-out results, analysis,
limitations, and conclusions based on those results. Do not report an
improvement until the held-out data supports it.
