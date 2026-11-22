# ADSA presentation outline

This outline follows the planned 15-slide project presentation. Keep all
performance claims tied to the saved experiment artifacts and retain the
real-world generalization caveat.

1. **Title** — Adaptive Graph Algorithm Selection for Shortest-Path Problems.
2. **Problem** — one source-to-destination query, multiple algorithm choices.
3. **Motivation** — input characteristics affect eligibility and measured
   performance; choosing well has a cost of its own.
4. **Algorithm portfolio** — BFS, Dijkstra, A*, Bellman–Ford, and their
   correctness/weight constraints.
5. **Selection task** — map graph features to the fastest eligible candidate;
   distinguish decision accuracy from end-to-end speed.
6. **System architecture** — graph → features → eligibility → runtime
   predictions → algorithm choice → verified path.
7. **Features** — Model B's eight features and why lightweight extraction
   matters.
8. **Ridge Regression** — log-runtime targets, per-algorithm models,
   standardization, alpha `0.0001`, and rule fallback.
9. **Experimental method** — development-only grouped CV, alpha tuning, fresh
   synthetic final cases, bounded DIMACS samples, repeats, and paired metrics.
10. **Feature ablation and alpha tuning** — development results and
    seed-level variability.
11. **Independent synthetic evaluation** — 672 cases, 92.71% Ridge versus
    79.61% rule winner accuracy.
12. **Runtime comparison** — Ridge's 5.69× median paired advantage over rules,
    alongside the 1.127× adaptive/oracle ratio showing remaining overhead.
13. **Real-world evaluation** — 44% Ridge versus 96% rules on 100 sampled
    queries; five inputs but four independent regions, with Colorado's two
    weight variants sharing topology.
14. **Limitations and future work** — synthetic-to-road generalization,
    bounded samples, cold/warm cache behavior, one host, and broader datasets.
15. **Conclusion and live demonstration** — evidence-supported takeaway and
    local demo walkthrough; present the observed oracle only after benchmarking
    all eligible algorithms.

## Speaker verification checklist

- Recompute slide numbers from the named result CSVs before exporting slides.
- Label the current paper and presentation as drafts until broader literature
  review and independent network coverage are complete.
- Do not describe selector accuracy as a runtime speedup over the oracle.
- Do not present Colorado distance and travel-time inputs as independent
  networks.
- Verify the live demo scenarios and path costs before presenting.
