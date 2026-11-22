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
cases. It trains a Ridge Regression runtime selector from development results,
then evaluates it using Model B's eight features and alpha 0.0001 on held-out
cases.
Each run records its graph grid, seeds, repeat count, and model setting in
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
selector uses Model B's eight features and alpha 0.0001. A fresh final
evaluation found higher winner accuracy than the rule selector, but median
end-to-end runtime was still slower than running the fastest eligible
individual algorithm directly. See `docs/RESEARCH_REPORT.md` before making
performance claims.

## Research paper status

The repository now includes independent synthetic results, a bounded
real-world road-network evaluation, paired statistical summaries, and a
runtime-profiling experiment. The synthetic-trained Ridge model did not
generalize well on the sampled road-network cases; see
`docs/REAL_WORLD_EVALUATION.md` and `docs/STATISTICAL_ANALYSIS.md`. The real-
world experiment covers four geographic networks and five graph/weight
inputs, with a 200-node sample from each input. It is not a full-network or
five-independent-network evaluation.

The research report remains an internal draft. An initial focused literature
search protocol, review, and comparison table are recorded in
`docs/LITERATURE_SEARCH_PROTOCOL.md`, `docs/LITERATURE_REVIEW.md`, and
`docs/LITERATURE_COMPARISON.csv`; broader database searches, a larger
independent real-world network set, paper figures and final editing are still
needed. The prior synthetic final set remains reserved for reporting and must
not be used for further model tuning.

## Reproduce the new studies

From the project root:

```bash
PYTHONPATH=backend python backend/experiments/runtime_profiling.py
PYTHONPATH=backend python backend/experiments/heuristic_cache_profiling.py
PYTHONPATH=backend python backend/experiments/real_world_evaluation.py
PYTHONPATH=backend python backend/experiments/statistical_analysis.py
python -m pip install -r requirements-experiments.txt
PYTHONPATH=backend python backend/experiments/model_comparison_cv.py
```

## Interactive demonstration

The optional Streamlit interface shows graph features, eligibility, runtime
predictions, selected paths, and same-query comparisons across eligible
algorithms. It keeps the measured fastest eligible algorithm as an after-the-
run comparison; the selector does not know the oracle choice in advance.

```bash
python -m pip install -r requirements-demo.txt
PYTHONPATH=backend python -m streamlit run backend/demo_app.py
```

See `docs/DEMONSTRATION_GUIDE.md` for scenario and timing notes and
`docs/PRESENTATION_OUTLINE.md` for the current talk structure.

A reviewed 15-slide live presentation is available in the ADSA Codex project
session. The local outline and demonstration guide remain the editable
repository references for that presentation.

The real-world study requires network access on its first run and stores
downloaded inputs under ignored `data/real_world/`. Runtime profiling and
heuristic-cache profiling and real-world evaluation overwrite only their own
named outputs. Statistical
analysis reads the saved synthetic and real-world results and writes its own
summary files. Do not rerun `backend/experiments/final_evaluation.py` or
`backend/experiments/run_benchmarks.py` when preserving the existing historical
evaluation files; those scripts regenerate their result paths.

The model-comparison command is optional and uses scikit-learn only for its
development-only Ridge/Random-Forest/Gradient-Boosting cross-validation. It
does not modify the production selector.
