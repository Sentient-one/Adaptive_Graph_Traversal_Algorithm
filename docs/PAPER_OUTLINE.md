# ADSA paper completion outline

Use `backend/results/final_evaluation_*.csv` for final outcome claims. The
existing held-out benchmark was used for diagnostic analysis and is not the
untouched final test set. Development-only feature and alpha decisions are
reported from `feature_ablation_repeated_cv.csv` and `ridge_alpha_cv.csv`.

1. **Abstract** — research question, method, evaluation scope, and measured
   conclusion.
2. **Introduction** — shortest-path problem, motivation, research question,
   contributions, and scope.
3. **Related work** — shortest-path algorithms and algorithm-selection research;
   foundational references are listed in `docs/RESEARCH_REPORT.md`; complete
   the literature review and verify claims against those sources.
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
9. **Results** — report category-level independent final results, cross-
   validation variability, and prediction errors.
10. **Discussion** — cases helped or hurt by selection, runtime overhead, and
    behavior on negative-weight graphs.
11. **Limitations and future work** — dataset coverage, hardware dependence,
    model sensitivity, and remaining assumptions.
12. **Conclusion and references** — answer the research question using measured
    evidence and provide complete references.

## Current evidence checkpoint

The independent final evaluation has completed on 672 new synthetic graph
cases. Ridge beat the rule baseline on winner accuracy, but the measured
median end-to-end runtime remained slower than running the fastest eligible
individual algorithm directly. See `docs/RESEARCH_REPORT.md` for the current
analysis and `docs/OPTIMIZATION_LOG.md` for decisions and measurement caveats.
