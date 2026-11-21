# ADSA Project Progress Summary

This file summarizes the work completed in the ADSA project through Steps 7–10.
It is stored in the project root next to `README.md`.

## Step 7 — Ridge alpha tuning

- Ran an initial exploratory sweep for Model B using nine alphas from
  `0.000001` through `100`. Its results remain in
  `backend/results/ridge_alpha_tuning_model_b.csv`.
- Then ran the planned seven-alpha experiment exactly as specified:
  `0.0001`, `0.001`, `0.01`, `0.1`, `1`, `10`, and `100`.
- The planned run kept Model B's eight features, the 252 development cases,
  five seeds, grouped folds, and training/validation procedure constant. Only
  alpha changed.
- Under the stated rule (highest mean non-trivial winner accuracy, then lowest
  mean median absolute log error), alpha `0.0001` was selected. It reached
  93.10% ± 0.53 percentage points winner accuracy and 91.94% ± 0.62 points
  non-trivial winner accuracy across seeds.
- Alpha `1` had the lowest median absolute log error in the seven-value sweep,
  so `0.0001` was selected for accuracy rather than for every metric.
- The reproducible script and full fold/seed results are in
  `backend/experiments/ridge_alpha_cv.py` and
  `backend/results/ridge_alpha_cv.csv`.

## Step 8 — Production integration

- Updated `backend/adaptive/performance_model.py` to use Model B's eight
  features, removing `average_degree` from the model vector, and set its
  default Ridge alpha to `0.0001`.
- Updated `backend/experiments/run_benchmarks.py` so future benchmark runs use
  alpha `0.0001` and record it in the configuration they generate.
- Preserved algorithm eligibility checks, prediction logic, selection logic,
  and the rule-based fallback.
- Added tests for the Model B feature vector and selected alpha, training all
  four algorithm regressors, finite positive predictions, selecting the
  lowest-predicted eligible algorithm, and safe fallback when no trained model
  is eligible.
- Ran the project test suite after integration: **33 tests passed**.
- Updated the README to describe the current selector and its measured
  performance trade-off.

## Step 9 — Independent final evaluation

- Trained the Ridge model from development data only; earlier held-out data was
  not treated as the untouched final test set.
- Evaluated on **672 new synthetic graph cases** with seeds `303`, `404`, and
  `505`, three repeats, graph sizes of 50, 100, 200, and 300 nodes, densities
  of 0.03, 0.10, 0.30, and 0.55, all supported weight modes, and supported
  direction/coordinate settings.
- Compared Ridge and the rule selector on the same cases and against each
  eligible individual algorithm. The benchmark checked path-cost correctness
  as it ran; all adaptive results passed.
- Results:

  | Measure | Ridge, alpha 0.0001 | Rule selector |
  |---|---:|---:|
  | Winner accuracy | 92.71% | 79.61% |
  | Non-trivial winner accuracy | 91.49% | 76.22% |
  | Median selection time | 8.0 μs | 561 μs |
  | Median estimated selection overhead | 8.7 μs | 582 μs |
  | Median peak memory | 11.6 KB | 246.7 KB |

- Ridge was faster than rules in 671 of 672 cases; median rule/Ridge paired
  runtime speedup was 5.69×.
- Ridge's median end-to-end runtime was still slower than directly running the
  fastest eligible individual algorithm: fastest-algorithm/adaptive speedup
  was 0.887× (adaptive-to-best runtime ratio 1.127×). The experiment therefore
  supports an accuracy and runtime advantage over rules, but not a runtime
  advantage over the best individual algorithm.
- Mean absolute log runtime-prediction error on the new cases was:
  BFS 0.524, Dijkstra 0.448, A* 0.736, and Bellman–Ford 0.321. A* had the
  highest prediction error; its median predicted-to-actual runtime ratio was
  0.845, indicating median underprediction.
- Added seed-level results to
  `backend/results/final_evaluation_seed_summary.csv` and summarized them in
  the research report. Ridge winner accuracy ranged from 91.07% to 93.75%
  across the three fresh test seeds; rule accuracy ranged from 78.57% to
  81.70%.
- Main reproducibility and result files:
  `backend/experiments/final_evaluation.py` and
  `backend/results/final_evaluation_config.json`,
  `final_evaluation_benchmarks.csv`,
  `final_evaluation_selection_summary.csv`,
  `final_evaluation_overall_summary.csv`,
  `final_evaluation_seed_summary.csv`,
  `final_evaluation_predictions.csv`,
  `final_evaluation_prediction_summary.csv`, and
  `final_evaluation_model_vs_rules.csv`.

## Step 10 — Research reporting

- Added `docs/RESEARCH_REPORT.md` with the problem, model, methodology,
  focused related work, algorithm complexity, feature-ablation and
  alpha-tuning results, final evaluation by seed, discussion, limitations,
  reproducibility paths, and foundational references.
- Expanded related work to cover Rice's algorithm-selection framework,
  algorithm performance-model portfolios, SATzilla, and Adham et al.'s
  machine-learning study of all-pairs shortest-path algorithm selection. The
  report distinguishes that all-pairs task from ADSA's single-pair setting and
  links the sources in its bibliography.
- Added `docs/OPTIMIZATION_LOG.md` to record decisions, evidence, and
  limitations in the available measurements.
- Updated `docs/PAPER_OUTLINE.md` to use the independent final evaluation and
  treat the prior held-out data as diagnostic only; it now records that the
  literature review is focused rather than comprehensive.
- Corrected and updated the README's benchmark and research-status descriptions.
- Recorded the feature-extraction comparison: on the final workload, median
  lightweight extraction time was 0.709 μs versus 584.7 μs for full extraction
  (about 825× lower by ratio of medians). The Union-Find optimization is noted
  as rejected per the project notes, but its quantitative benchmark artifact
  was not present, so no numeric before/after claim is made.

## Verification and repository state

- After Step 8 integration, the project test suite passed: **33 passed**.
- `git diff --check` passed after the reporting changes.
- Existing benchmark outputs and historical experiment configuration were
  preserved; they were not regenerated or overwritten during these steps.
- No changes were committed.
- Experiment scripts and their result files are untracked additions until
  intentionally staged. Production and documentation edits are currently
  uncommitted.

## Remaining work

- Before submission, document a literature search protocol and broaden the
  focused related-work review, especially coverage of learned graph algorithm
  selection. The current report is an internal draft, not a systematic survey.
- Decide whether another optimization could reduce the remaining selector
  overhead enough to beat direct execution of the fastest eligible algorithm.
- Consider evaluation on real-world graph datasets and more source/destination
  pairs; the current independent test uses the project's synthetic generator
  and one fixed endpoint pair per graph.
- Review and stage the intended files, then commit when ready.
