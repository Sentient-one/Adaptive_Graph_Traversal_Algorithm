# ADSA Project Progress Summary

This file summarizes the original Steps 7–10 and the subsequent profiling,
real-world evaluation, statistical analysis, model comparison, literature
search protocol, and cold A* cache optimization. It is stored in the project
root next to `README.md`.

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

## Optimization roadmap — baseline and selector profiling

- Verified branch `feature/algorithm-improvements`, baseline commit
  `1f22bf5`, and a clean tree before starting this work.
- Captured environment, production model, final-evaluation configuration,
  timing methodology, and historical files to preserve in
  `backend/results/optimization_baseline_config.json`. Python is 3.14.6 and
  pytest is 9.0.3; the repository has no root `requirements.txt`.
- Re-ran the existing suite: **33 passed, 7 subtests passed**.
- Added `backend/experiments/runtime_profiling.py`, which separately measures
  selection-feature extraction, eligibility, Ridge prediction, candidate
  comparison, complete cold/warm selection, algorithm execution, and regret
  on nine reproducible synthetic cases.
- Added the per-case, detailed benchmark, and summary outputs under
  `backend/results/runtime_profiling*.csv`, plus
  `docs/RUNTIME_PROFILING_REPORT.md`.
- Profiling identified runtime prediction as the largest repeatable warm
  decision stage; cold feature extraction has a high tail when A*'s heuristic
  scale must be computed.
- Benchmarked a fixed-width dot-product change against the original predictor:
  median prediction time fell from 4.416 μs to 3.750 μs (15.22%) across these
  nine cases, with equivalent predictions within `1e-12` relative tolerance
  and a faster result in all nine case comparisons. Applied this isolated
  implementation optimization in `backend/adaptive/performance_model.py`;
  the model configuration and selection behavior are unchanged.
- Re-ran all existing tests after the change: **33 passed, 7 subtests passed**.
- The project path does not permit pytest's cache directory to be written, so
  pytest emits a cache warning; tests still pass.

## Optimization roadmap — real-world and statistical evaluation

- Added `backend/experiments/real_world_evaluation.py` and ran it on five
  graph/weight inputs: New York distance, Colorado distance, Colorado travel
  time, Florida distance, and Rome99 distance. These represent four
  geographic networks; Colorado distance/time share a topology.
- Evaluated 20 reachable near/far endpoint pairs per input (100 pairs total)
  using connected 200-node induced samples, with three timing repeats and
  path-cost checks. DIMACS weights are preserved; parallel arcs are reduced to
  their minimum weight per ordered pair because the project Graph rejects
  parallel edges. No arbitrary weights or coordinates were added.
- Results: Ridge winner accuracy was **44%**, rules **96%**. Median
  adaptive/oracle ratio was **1.221×** for Ridge and **3.676×** for rules.
  Ridge's median selected-algorithm/oracle ratio before selector overhead was
  1.079×. All returned path costs were verified.
- The five inputs are not five independent networks: they cover four regions,
  one duplicated Colorado topology with different weight files, and one
  200-node sample per input. Florida and Rome99 lacked coordinates in the
  verified mirrors, so A* was ineligible there. The original DIMACS data host
  returned 404; verified public mirrors and source hashes are recorded in the
  manifest. Data files are stored in ignored `data/real_world/`.
- Added `docs/REAL_WORLD_EVALUATION.md` and the five `real_world_*.csv`
  artifacts. This is evidence that the synthetic-trained Ridge model does
  **not** generalize reliably to these sampled road networks.
- Added `backend/experiments/statistical_analysis.py` and generated confidence
  intervals, paired comparisons, runtime regret/near-optimal rates, and
  feature correlations. On the synthetic final cases, Ridge beat rules by
  13.10 percentage points in winner accuracy (paired 95% bootstrap interval
  +9.52 to +16.67 pp; exact McNemar p = 5.25e-13). Ridge was faster end-to-end
  in 671/672 pairs. On real-world samples the accuracy difference reversed;
  geographic-cluster intervals are broad because only four networks were
  represented.
- Added `docs/STATISTICAL_ANALYSIS.md`, updated README run guidance, and kept
  the prior final evaluation as reporting-only data.

## Optimization roadmap — development model comparison

- Added `backend/experiments/model_comparison_cv.py` to compare the current
  Ridge model with Random Forest and Gradient Boosting, using Model B's eight
  features and a 13-feature endpoint extension. It uses the same five grouped
  CV seeds and folds on the 252 development cases only.
- Added `requirements-experiments.txt`; scikit-learn is optional and is not
  imported by the production selector or existing tests.
- Results are in `backend/results/model_comparison_repeated_cv.csv` and
  `model_comparison_cv_config.json`; analysis is in
  `docs/MODEL_COMPARISON.md`.
- Model B Ridge retained the highest accuracy: **93.10% ± 0.53 pp** winner
  accuracy and **91.94% ± 0.62 pp** non-trivial accuracy. Random Forest had
  lower prediction error but lower winner accuracy, much higher inference
  time, and much larger model files. Adding endpoint features reduced winner
  accuracy for all three regressors.
- Production remains Model B Ridge alpha `0.0001`. No alternative is justified
  by development CV, and the model configuration has not been retuned on the
  independent final set. A fresh untouched evaluation is still required
  before claiming improved real-world generalization.

## Optimization roadmap — initial literature search protocol

- Added `docs/LITERATURE_SEARCH_PROTOCOL.md` with a dated scoping-search
  description, topic queries, inclusion/exclusion rules, screening fields, and
  an initial screened set.
- Added `docs/LITERATURE_REVIEW.md` and `docs/LITERATURE_COMPARISON.csv` with
  a source-checked comparison of algorithm-selection foundations, SATzilla,
  ASlib, the all-pairs shortest-path selection paper, and adjacent
  learning-augmented shortest-path work.
- Expanded the research report to distinguish portfolio selection from
  learning-augmented shortest-path algorithms. The closest located selection
  paper is for all-pairs shortest paths; this search is not sufficient to
  claim no prior single-pair selector exists.
- Added verified bibliography entries for the ASlib benchmark paper, the
  automated algorithm-selection survey, and two adjacent learned graph
  algorithm studies. Broader database searching, citation chasing, and final
  DOI/version checks remain necessary before submission.

## Optimization roadmap — focused cold A* cache study

- Added `backend/experiments/heuristic_cache_profiling.py` and
  `backend/results/heuristic_cache_profiling.csv`. It measures 24 graph cases
  (100–2,000 nodes, three edge probabilities, directed and undirected), cold
  and warm heuristic-scale/feature extraction, and a paired scan candidate.
- Cold heuristic-scale computation had a 0.708 ms median across case medians;
  the warm cache hit was 0.166 μs. The local-binding candidate reduced scan
  medians by 1.60%, with a 95% case-bootstrap interval of −1.97% to −1.02%,
  and was faster in all 24 cases.
- Applied the equivalent local-binding change in `backend/graph/graph.py`.
  Cache invalidation behavior is unchanged. The regression suite passed:
  **33 tests and 7 subtests**. This result is limited to one host and generated
  graph cases.

## Demonstration and presentation preparation

- Added `backend/demo_app.py`, a local Streamlit demo with unit-weight,
  positive-weight geometric, and negative-edge directed DAG scenarios. It
  exposes graph features, algorithm eligibility, Model B predictions, model
  and rule choices, paths, per-algorithm timing, and end-to-end comparisons.
- The demo runs every eligible algorithm on the same graph/query, validates
  path costs, and labels the observed fastest algorithm as an after-run
  benchmark oracle. It does not write benchmark or final-evaluation results.
- Added optional `requirements-demo.txt`, setup and interpretation guidance in
  `docs/DEMONSTRATION_GUIDE.md`, and a 15-slide plan in
  `docs/PRESENTATION_OUTLINE.md`.
- Exercised all three scenarios through Streamlit's app runner. Path costs
  were consistent, eligibility matched the intended cases, and no app
  exceptions occurred.
- Completed and reviewed a 15-slide live Spaces presentation. It covers the
  problem, eligibility, Model B, evaluation design and results, profiling,
  road-network generalization limits, optimizations, and next research work.
  Its live presentation artifact is available in this Codex project session;
  the local source outline remains in `docs/PRESENTATION_OUTLINE.md`.

## Current verification and delivery state

- Re-ran the full project test suite after the latest changes:
  **33 tests passed and 7 subtests passed**.
- `git diff --check` passed.
- The Streamlit demo's three scenarios were exercised without app exceptions;
  path costs and eligibility were checked, including a negative-edge DAG.
- Historical final-evaluation and benchmark result files were preserved.
- No Git commit was created. All project changes remain available for review.
- The production model remains Model B Ridge, alpha `0.0001`; its eligibility,
  correctness checks, selection behavior, and fallback remain intact.

## Remaining work

- **Highest research priority:** expand real-world evaluation with independent
  networks, larger or full graphs, more reproducibly selected source/destination
  pairs, and coordinates compatible with A*. The current five inputs cover
  four regions, reuse one Colorado topology for two weight files, and use one
  200-node sample per input. The synthetic-trained model's 44% winner accuracy
  versus 96% for rules shows that this generalization gap needs investigation.
- The real-world downloader currently lacks reliable access to the original
  DIMACS host in this environment. Add and record a new network only after its
  provenance, license, weights, topology, and any coordinates are verified.
- Complete a broader literature review: search bibliographic databases, chase
  citations, verify publication versions and DOIs, record screening counts,
  and seek a second reviewer where feasible. The new review and comparison
  table are a source-checked scoping pass, not a systematic review.
- Turn the internal research report into the final manuscript: complete the
  planned figures and tables, verify every claim against source artifacts,
  finish citations and formatting, and obtain supervisor review. No submission
  format or supervisor approval has been completed here.
- Keep optimization work evidence-led. The local Ridge and A* scan changes are
  measured and modest; the selector still has a median 1.127× runtime ratio
  versus direct execution of the fastest eligible algorithm. Any further
  optimization should target measured regret without changing correctness.
- The alternative-regressor comparison and initial statistical analysis are
  complete. Do not tune against the already-used synthetic final set; reserve
  it for reporting and evaluate future model changes on a new independent set.
- Before delivery, review the intended file set, clean up generated artifacts
  only with care, then stage and commit after the user has reviewed the changes.

## How to run the current project and studies

From the project root, use Python 3.10+; core runtime code uses the Python
standard library. Install pytest to run tests. Install optional requirements
only for the corresponding experiment or demonstration:

```bash
python -m pip install pytest
PYTHONPATH=backend pytest -p no:cacheprovider backend/tests/ -v
PYTHONPATH=backend python backend/experiments/runtime_profiling.py
PYTHONPATH=backend python backend/experiments/heuristic_cache_profiling.py
PYTHONPATH=backend python backend/experiments/real_world_evaluation.py
PYTHONPATH=backend python backend/experiments/statistical_analysis.py
python -m pip install -r requirements-experiments.txt
PYTHONPATH=backend python backend/experiments/model_comparison_cv.py
python -m pip install -r requirements-demo.txt
PYTHONPATH=backend python -m streamlit run backend/demo_app.py
```

`requirements-experiments.txt` installs scikit-learn for the optional model
comparison. `requirements-demo.txt` installs Streamlit for the local app.
The saved evaluation scripts can overwrite their own named result files;
preserve the current CSVs before rerunning experiments.

The real-world run needs network access on first use. It saves downloaded
inputs in ignored `data/real_world/`. Statistical analysis expects saved
`final_evaluation_*.csv` and `real_world_*.csv` artifacts. The example app is
`cd backend && python main.py`; it overwrites `backend/results/benchmark.csv`.
The model-comparison run is optional, needs scikit-learn, and uses only the
development dataset.
The synthetic final-evaluation and development benchmark scripts rewrite their
result files, so do not rerun them if preserving the current historical
results.
