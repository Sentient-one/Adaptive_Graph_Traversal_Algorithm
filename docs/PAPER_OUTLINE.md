# ADSA paper completion outline

Use results from `backend/results/heldout_benchmarks.csv` and
`backend/results/selection_summary.csv`. Do not fill in outcome claims before
the held-out experiment has completed.

1. **Abstract** — research question, method, evaluation scope, and measured
   conclusion.
2. **Introduction** — shortest-path problem, motivation, research question,
   contributions, and scope.
3. **Related work** — shortest-path algorithms and algorithm-selection research;
   add verified citations.
4. **Background and problem definition** — graph model, path-cost definition,
   supported inputs, and objectives.
5. **Algorithms and eligibility** — BFS, Dijkstra, A*, Bellman–Ford, and
   correctness assumptions.
6. **Proposed selection methods** — explain the rule baseline and the
   development-data ridge-regression runtime model.
7. **Correctness and complexity** — shortest-path guarantees, selector
   overhead, and algorithm-specific work counters.
8. **Experimental methodology** — graph families, sizes, densities, seeds,
   development/held-out separation, repeats, metrics, and environment.
9. **Results** — report category-level held-out results and variability.
10. **Discussion** — cases helped or hurt by selection, runtime overhead, and
    behavior on negative-weight graphs.
11. **Limitations and future work** — dataset coverage, hardware dependence,
    model sensitivity, and remaining assumptions.
12. **Conclusion and references** — answer the research question using measured
    evidence and provide complete references.
