# ADSA Optimization and Experiment Log

This log records changes and evaluation evidence. Timing figures are specific
to the recorded workloads and machine; they should not be treated as universal
performance guarantees.

| Change or decision | Reason | Evaluation and evidence | Status |
|---|---|---|---|
| Feature extraction optimization: add lightweight selection-feature extraction | Avoid connectivity and reporting calculations that runtime selection does not need | On the 672-case final workload, the median lightweight extraction time was 0.709 μs versus 584.7 μs for full extraction (about 825× lower by ratio of medians). Ridge median decision time was 8.0 μs; the rule baseline's was 561 μs. These are same-run measurements, not a controlled old-version/new-version deployment test. | Accepted; retained in `extract_selection_features` |
| Union-Find connectivity optimization | Proposed to reduce connectivity work | Project notes say the added overhead outweighed its benefit. No code, raw measurements, or before/after result file for this experiment was found in the repository; no quantitative claim is made here. | Rejected, based on recorded project note; recover measurements if this is to be reported as measured evidence |
| Model B feature selection | Reduce the predictor from nine features to eight | Repeated grouped five-fold CV on 252 development cases, five seeds. Model B reached 93.02% ± 0.60 pp winner accuracy and 91.85% ± 0.70 pp non-trivial accuracy. Models C/D had slightly lower median absolute log error. | Provisionally accepted for selection accuracy |
| Ridge alpha tuning | Choose regularization strength while holding features/data/folds constant | Seven alphas, five grouped-CV seeds. α=0.0001 achieved 93.10% ± 0.53 pp winner accuracy and 91.94% ± 0.62 pp non-trivial accuracy, the highest tested values. α=1 had lower median absolute log error. | Accepted under the declared accuracy-first rule |
| Step 8 production integration | Use selected Model B features and α=0.0001 in the runtime selector | Updated model vector/default, benchmark runner, README, and targeted tests. Project suite: 33 passed. Eligibility and rule fallback remain in place. | Integrated; no commit made |
| Step 9 independent final evaluation | Test generalization on new graph cases and compare selectors | 672 generated cases; three new seeds; 3 repeats; path costs verified. Ridge had 92.71% winner accuracy versus 79.61% for rules and was faster than rules on 671/672 paired cases. Median adaptive runtime was still slower than the fastest individual eligible algorithm (0.887× speedup). | Completed; mixed result, report the trade-off |

## Measurement conventions

- Feature and selector timing measurements come from
  `backend/results/final_evaluation_benchmarks.csv`.
- Accuracy and paired runtime comparisons come from
  `backend/results/final_evaluation_overall_summary.csv` and
  `backend/results/final_evaluation_model_vs_rules.csv`.
- Cross-validation values and per-seed variation come from the two experiment
  result files under `backend/results/`.
- The final test uses synthetic graphs from the project generator. Three new
  seeds separate it from previous runs, but do not establish performance on
  real-world graph distributions.
