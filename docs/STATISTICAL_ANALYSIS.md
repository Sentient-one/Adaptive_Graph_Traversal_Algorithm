# Statistical analysis

## Methods

The analysis uses the already completed 672-case synthetic final evaluation and
the 100 real-world sampled queries; it does not tune the model or regenerate
either evaluation. Winner-accuracy intervals use Wilson score intervals.
Accuracy differences are paired by graph/query case and receive a 5,000-draw
bootstrap interval. Synthetic paired correctness uses an exact two-sided
McNemar test; paired end-to-end runtime uses an exact sign test. Runtime
regret is reported both before and after selection overhead. Near-optimality
was defined before analysis as selected/oracle runtime ratio `<= 1.10`.

For real-world results, endpoint pairs are clustered by the underlying
geographic network. Confidence intervals for paired differences and runtime
ratios use a 5,000-draw geographic-cluster bootstrap. Only four geographic
networks are represented, and Colorado contributes two related cost variants;
these intervals are exploratory and broad. Query-level Wilson intervals are
also reported for reference but do not account for within-network dependence.
No real-world McNemar or runtime sign-test p-value is reported because the 100
queries are clustered within only four networks.

## Paired selector comparison

| Evaluation | Cases | Ridge accuracy | Rule accuracy | Ridge minus rules, 95% interval | McNemar p | Median paired Ridge/rules runtime | Paired runtime result |
|---|---:|---:|---:|---:|---:|---:|---|
| Synthetic final | 672 | 92.71% | 79.61% | +13.10 pp `[+9.52, +16.67]` | `5.25e-13` | 0.176× | Ridge faster in 671/672; exact sign-test p `6.87e-200` |
| Real-world samples | 100 | 44% | 96% | −52 pp `[−86.67, 0]` cluster bootstrap | Not reported | 0.377× `[0.345, 0.414]` cluster interval | Ridge faster in 100/100 paired queries; exploratory four-network sample |

The real-world intervals vary with the very small number of network clusters.
They should not be read as a stable population estimate. The paired outcomes
still provide evidence that Model B's synthetic-trained runtime predictions
do not transfer well to the selected real-world subgraphs.

## Synthetic runtime regret

| Selector | Median selected/oracle, no selection overhead | Median adaptive/oracle, with overhead | Near-optimal no overhead | Near-optimal with overhead | Median regret | 90th percentile regret | Maximum regret |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ridge | 1.000× | 1.127× | 95.98% | 45.68% | 9.13 μs | 27.08 μs | 3.47 ms |
| Rules | 1.000× | 7.664× | 82.89% | 0% | 585 μs | 5.01 ms | 9.74 ms |

The maximum regret is timing-sensitive and comes from the upper tail; the
median and percentile summaries are more representative of typical cases.
Negative per-case regret can also occur when separately measured medians put
the adaptive timing below the measured oracle due to timing noise.

## Prediction error and feature relationships

The algorithm-specific error analysis is written to
`backend/results/statistical_summary.csv`, grouped by algorithm and by graph
size, weight mode, and coordinate availability. The previous aggregate result
remains: A* had the largest mean absolute log error (0.736), followed by BFS
(0.524), Dijkstra (0.448), and Bellman–Ford (0.321).

Pearson correlations across the synthetic final benchmark rows are in
`backend/results/feature_relationships.csv`. They are descriptive and do not
establish causality. For example, node count correlated positively with
measured runtimes for all four algorithms in this generated set; density also
correlated positively. Density and average degree were correlated (r=0.730),
which follows their mathematical relationship with graph size for these graph
families. A* heuristic scale and A* runtime had a weak negative correlation
(r=−0.155).

## Assumptions and limits

- Runtime values come from three timed repeats in the final evaluation and
  five repeats in the real-world experiment; confidence intervals here do not
  model machine-load or timing-instrumentation uncertainty.
- Winner ties count as correct if the selector chose any measured fastest
  eligible algorithm.
- Synthetic case-level resampling treats each generated graph case as one
  observation. Real-world endpoint queries are clustered by graph source.
- The real-world sample contains four geographic networks, not five
  independent networks. Colorado's distance and time files share a topology.
- Correlations are measured on synthetic final data only and are not a basis
  for changing features without a separate development experiment.
- The sign test is used for paired runtime direction; no Wilcoxon test is
  claimed. All p-values are secondary to the effect sizes and data limits.

## Reproduce

```bash
PYTHONPATH=backend python backend/experiments/statistical_analysis.py
```

Outputs:

- `backend/results/statistical_summary.csv`
- `backend/results/confidence_intervals.csv`
- `backend/results/paired_comparisons.csv`
- `backend/results/feature_relationships.csv`
