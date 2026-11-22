# Development-only model comparison

## Purpose

The real-world study showed weak synthetic-to-road generalization, so this
experiment checks whether a different regressor or a small set of additional
endpoint features improves development cross-validation. It reads only
`backend/results/development_benchmarks.csv`; it does not read the synthetic
final or real-world outcome files and does not modify the production model.

The comparison uses the same five repeated grouped folds as the feature
ablation study (252 development cases; seeds 11, 22, 42, 101, and 202):

- Ridge, alpha `0.0001`, standardized inputs.
- Random Forest, 100 trees, `min_samples_leaf=2`.
- Gradient Boosting, 100 estimators, learning rate `0.05`, depth 2, Huber
  loss.
- Model B's eight features, and Model B plus source degree, destination
  degree, endpoint heuristic distance, mean weight, and unique-weight ratio.

Tree settings were fixed before evaluating the cross-validation output. The
candidate features are computed only in the experiment. Tree and Ridge
inference time is measured with scikit-learn prediction calls; it is not a
complete selector-latency measurement.

## Results

Means and sample standard deviations are across the five seed-level summaries.

| Features | Regressor | Winner accuracy | Non-trivial accuracy | Median absolute log error | Prediction calls per case | Model size (pickle) |
|---|---|---:|---:|---:|---:|---:|
| Model B | Ridge | **93.10% ± 0.53 pp** | **91.94% ± 0.62 pp** | 0.3024 ± 0.0035 | 0.253 ms | 4.2 KB |
| Model B | Random Forest | 91.35% ± 1.03 pp | 89.91% ± 1.20 pp | **0.2542 ± 0.0195** | 7.95 ms | 2.33 MB |
| Model B | Gradient Boosting | 89.60% ± 1.55 pp | 87.87% ± 1.81 pp | 0.2601 ± 0.0075 | 0.247 ms | 315 KB |
| Model B + endpoints | Ridge | 89.68% ± 0.63 pp | 87.96% ± 0.73 pp | 0.2943 ± 0.0072 | 0.254 ms | 4.9 KB |
| Model B + endpoints | Random Forest | 89.76% ± 1.24 pp | 88.06% ± 1.44 pp | 0.2553 ± 0.0112 | 7.95 ms | 2.35 MB |
| Model B + endpoints | Gradient Boosting | 89.76% ± 0.91 pp | 88.06% ± 1.06 pp | 0.2575 ± 0.0084 | 0.249 ms | 315 KB |

The random forest had the lowest runtime-prediction error, but lower winner
accuracy, much larger model size, and substantially slower inference. The
endpoint feature set reduced winner accuracy for all three regressors. Ridge
with Model B remained strongest on the accuracy-first metric used for
selection.

## Decision

Keep the production configuration at Model B Ridge alpha `0.0001`. This
development-only comparison does not solve the observed real-world
generalization gap. No model or feature configuration was changed, and the
existing independent final results were not used for this comparison. A new
untouched evaluation is required before claiming that any future configuration
improves out-of-domain performance.

## Reproduce

Install the optional experiment dependency and run:

```bash
python -m pip install -r requirements-experiments.txt
PYTHONPATH=backend python backend/experiments/model_comparison_cv.py
```

Outputs:

- `backend/results/model_comparison_repeated_cv.csv`
- `backend/results/model_comparison_cv_config.json`

The production selector and test suite continue to use the Python standard
library; scikit-learn is required only for this optional comparison.
