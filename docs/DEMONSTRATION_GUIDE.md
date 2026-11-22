# Interactive demonstration guide

## Purpose

The Streamlit demonstration exposes the existing shortest-path algorithms and
selector through a small local interface. It does not change the production
selector or write benchmark results. Each run creates a deterministic graph,
checks algorithm eligibility, displays Model B runtime predictions, runs every
eligible algorithm on the same graph/query, and verifies the returned path
costs.

## Setup and launch

From the repository root:

```bash
python -m pip install -r requirements-demo.txt
PYTHONPATH=backend python -m streamlit run backend/demo_app.py
```

The app uses `backend/results/development_benchmarks.csv` to train the
dependency-free Ridge model with the same production feature set and alpha.
Keep that development file in place. The app runs locally and requires no
external service or network access after setup.

## Demonstration scenarios

- **Unit-weight mesh** — BFS is eligible; Dijkstra and A* are also valid. This
  shows how equal weights affect the available choices.
- **Positive weighted mesh** — BFS is ineligible. Coordinates make A* eligible
  alongside Dijkstra and Bellman–Ford.
- **Negative-edge directed graph** — the graph is a forward-only DAG with
  negative edges and no cycles, so Bellman–Ford is the only eligible method.

Change the node count, additional-edge density, seed, endpoints, and timing
repeat count in the sidebar. For the mesh scenarios, density controls the
number of extra links; for the negative DAG, it controls the chance of adding
each forward skip edge. The mesh backbone preserves connectivity; the
negative-edge scenario can be unreachable when querying against edge direction.

## Reading the results

The feature and eligibility panels show what the selector can use before it
runs a path query. Predicted times are displayed only for eligible algorithms.
The measured per-algorithm comparison runs every eligible method on the same
graph and query and checks each returned path cost against its edge weights.

The “observed fastest” algorithm is computed only after all eligible
algorithms have been timed; it is an evaluation result, not an input to the
selector. Model and rule end-to-end timings include their selection decisions
and path execution. The selector-decision row is reported separately. Feature
display warms the graph's heuristic-scale cache before timing, so the measured
comparison represents a repeat query rather than first-query cold latency.
Microsecond timings vary with hardware and system load and should be treated as
demonstration measurements, not research results.

## Correctness and data handling

The demonstration uses only local generated graphs and the saved development
benchmark CSV. It does not alter historical CSV results, the final evaluation,
or the model configuration. It reports an error if eligible algorithms return
different path costs or a path's edge sum disagrees with its reported cost.

The standalone command-line example in `backend/main.py` remains available,
but it overwrites `backend/results/benchmark.csv`; the Streamlit demonstration
does not.
