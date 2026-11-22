# Runtime profiling report

## Objective

Measure the adaptive selector's stages separately, distinguish cold and warm
graph-feature cache behavior, and decompose runtime regret into selector cost
and the cost of choosing an algorithm slower than the measured oracle. This
is a profiling experiment only; it does not modify production selection logic.

## Reproducibility

- Baseline commit: `1f22bf5` on `feature/algorithm-improvements`.
- Model: Model B, Ridge Regression, alpha `0.0001`, trained from
  `backend/results/development_benchmarks.csv` only.
- Nine fixed generated cases span 50, 100, and 200 nodes; sparse and denser
  graphs; unit, positive, and negative weights; directionality; and coordinate
  availability. Seeds and all graph metadata are recorded per row.
- Stage measurements use `time.perf_counter_ns()` and 31 samples per stage.
  Algorithm and end-to-end benchmarks use the existing harness with five
  repeats and a correctness-checked warm-up.
- Cold feature and complete-selection samples use fresh graph instances. Warm
  samples reuse an unchanged graph after selection features have been
  extracted once. Graph generation is outside the timed region.
- The script writes new `runtime_profiling*.csv` files and does not regenerate
  or overwrite historical experiment outputs.

## Measurements

Across the nine cases, medians of each case's stage median were:

| Stage | Median | Approx. 90th percentile | Interpretation |
|---|---:|---:|---|
| Cold selection feature extraction | 0.917 μs | 74.3 μs | High tail when a fresh graph has coordinates and heuristic scale must be calculated |
| Warm selection feature extraction | 0.500 μs | 0.584 μs | Cached heuristic scale and maintained graph summaries |
| Eligibility checks | 1.000 μs | 1.083 μs | Calls existing validity rules for candidate algorithms |
| Ridge prediction after optimization | 3.750 μs | 3.833 μs | Largest repeatable warm selection stage in this sample |
| Candidate comparison | 0.167 μs | 0.208 μs | Lowest measured warm stage |
| Cold complete selection | 9.709 μs | 81.8 μs | Sensitive to cold heuristic-scale computation |
| Warm complete selection after prediction optimization | 5.750 μs | 5.833 μs | Prediction remains the largest repeatable warm stage |

The original predictor took a median 4.416 μs per call across the nine cases.
A narrowly scoped fixed-width dot-product implementation took a paired median
3.750 μs, a 15.22% reduction across case medians; it was faster in all nine
case-level comparisons. Outputs matched the prior implementation within
`1e-12` relative tolerance on every profile case. This implementation is now
in `performance_model.py`; it changes only how the same standardized model
dot products are evaluated. The model, features, alpha, training rows,
eligibility rules, prediction target, and winner policy are unchanged.

These numbers are medians across nine case medians on one macOS ARM64 host;
microsecond values are machine- and load-sensitive. The stage medians are
measured in separate calls and should not be summed as an exact decomposition
of complete selection latency. The p90 is the nearest lower observed case
order statistic from nine cases, not a population confidence bound.

## Runtime regret

The report records, for each case:

`total regret = adaptive end-to-end median - fastest eligible algorithm median`

and separately:

`algorithm-choice regret = selected algorithm median - fastest eligible algorithm median`.

Selector overhead is estimated using the benchmark harness's difference
between the adaptive end-to-end median and the selected algorithm median.
Because these are medians from separate timing samples, the components are
approximate and need not add exactly. The nine-case profiling sample chose a
measured fastest eligible algorithm in every case, so its algorithm-choice
regret was zero at recorded precision. Several total-regret values were near
zero or negative, which reflects timing noise at this scale, not a guaranteed
runtime benefit from selection.

## Findings and optimization decision

Warm Ridge prediction was the largest individually measured decision stage;
candidate comparison was negligible. Cold feature extraction had a
pronounced tail when A* was eligible and its heuristic scale was uncached.
The prediction-loop change was accepted after a paired same-case benchmark,
output-equivalence check, and full existing test run (33 passed, 7 subtests).
The nine-case end-to-end measurements are noisy and small; they are not used
to claim a general application speedup. The prior independent final evaluation
is not reused for tuning or claimed as a post-change test.

## Focused A* heuristic-cache experiment

The earlier profiler showed a cold-start tail when all node positions were
available and the model extracted its heuristic-scale feature. A focused
experiment now measures the scale calculation directly on 24 deterministic
positive-weight generated graphs: four sizes (100, 500, 1,000, 2,000 nodes),
three edge-probability settings (0.002, 0.01, 0.03), and directed/undirected
variants. Graph generation and cache-reset copies are outside timed regions.
For each graph, 101 fresh dirty-cache calls measured cold cost, 101 calls on
one unchanged warmed graph measured cache-hit cost, and 51 paired scans
compared the original loop with a local-binding candidate. The candidate
binds the positions mapping and `math.hypot` once before scanning edges; its
calculation and returned value are unchanged.

| Measurement | Median across 24 graph-case medians | Range across cases |
|---|---:|---:|
| Cold public heuristic-scale calculation | 0.708 ms | 7.8 μs–17.8 ms |
| Warm public heuristic-scale cache hit | 0.166 μs | 0.125–0.167 μs |
| Cold selection-feature extraction | 0.703 ms | 8.2 μs–17.6 ms |
| Warm selection-feature extraction | 0.542 μs | 0.542–0.583 μs |
| Original direct scale scan | 0.719 ms | 7.9 μs–17.7 ms |
| Local-binding scan candidate | 0.705 ms | 7.6 μs–17.6 ms |

The paired candidate scan was faster in all 24 graph cases. Its median change
was **−1.60%** (case-bootstrap 95% interval **−1.97% to −1.02%**, 20,000
resamples of the 24 case-level paired median changes). The improvement is
small but consistent in this controlled sample, so the local-binding change
was applied in `backend/graph/graph.py`. Existing cache invalidation on edge
and coordinate changes remains in place; the existing regression test covers
both mutations. The full suite passed after the change (33 tests and 7
subtests).

At 2,000 nodes, the median cold scale time across six direction/density cases
was 5.96 ms, while the warm cache hit remained 0.166 μs. Cold work therefore
scales with stored arcs; graph size alone is not a sufficient description of
the first-query cost. These generated graph measurements come from one host
and one fixed seed per case. The bootstrap interval reflects variation across
these selected graph cases, not a population-wide performance guarantee.
Full selection-feature timings include only Model B's lightweight feature
extraction and heuristic-scale lookup, not the larger reporting feature set.

The focused runner and per-case outputs are `backend/experiments/heuristic_cache_profiling.py`
and `backend/results/heuristic_cache_profiling.csv`.

The cold-cache scan optimization was evaluated and applied separately from the
earlier Ridge prediction change. Further optimization should be justified by
new measurements; the current scan improvement is modest and has not been
validated across other Python versions or hardware.

## Files

- `backend/experiments/runtime_profiling.py` — controlled profiling runner.
- `backend/results/runtime_profiling.csv` — per-case stage timings and regret.
- `backend/results/runtime_profiling_benchmarks.csv` — individual algorithm
  and adaptive benchmark rows used for regret calculations.
- `backend/results/runtime_profiling_summary.csv` — across-case summaries.
- `backend/results/optimization_baseline_config.json` — captured baseline.
