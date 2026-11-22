# ADSA: Experimental Results and Analysis

## Abstract

This project studies whether graph features can help choose among BFS,
Dijkstra, A*, and Bellman–Ford for a source-to-destination shortest-path
query. A Ridge Regression selector predicts each eligible algorithm's median
runtime from eight graph features, then chooses the algorithm with the lowest
predicted runtime. Repeated grouped cross-validation on 252 development graph
cases provisionally selected Model B and Ridge alpha 0.0001. A separate final
evaluation used 672 newly generated cases, three new graph seeds, and a wider
graph grid. Ridge selected the measured fastest eligible algorithm in 92.71%
of all cases and 91.49% of cases with multiple eligible algorithms, compared
with 79.61% and 76.22% for the rule selector. Both selectors returned costs
verified against the individual algorithms. Ridge was 5.69 times faster than
the rule selector at the median paired end-to-end runtime, but its median
end-to-end time was still 12.7% slower than running the fastest measured
eligible algorithm directly. These findings support better selection accuracy
than the rule baseline on this synthetic workload, but do not establish a
runtime advantage over the best individual algorithm.

## Introduction and problem statement

The shortest-path algorithms in this project make different trade-offs. BFS
works for equal nonnegative edge weights; Dijkstra and A* require nonnegative
weights; Bellman–Ford supports negative weights and detects reachable
negative-weight cycles. No one algorithm is eligible or fastest for every
supported graph. The project asks whether observable graph characteristics can
predict which eligible implementation will run fastest for a query.

This is an instance of algorithm selection: use features of a problem instance
to choose among available algorithms according to a performance measure, as
framed by Rice [5]. The system retains deterministic validity checks and uses
the learned model only to rank algorithms that are valid for the supplied
graph.

The research question is: **Can a lightweight, feature-based Ridge selector
choose the fastest eligible shortest-path algorithm more accurately than a
rule-based selector on generated graphs, while accounting for the cost of
selection?** Speed, decision quality, selection overhead, memory, and
correctness are treated as separate outcomes so better selection accuracy is
not mistaken for faster end-to-end execution.

## Related work

Rice's formulation provides the general setting for this project: characterize
problem instances with measurable features, measure candidate algorithm
performance, and map instances to algorithms [5]. A standard portfolio pattern
is to train algorithm-specific performance models, estimate all candidates'
runtime for a new instance, and execute the predicted fastest one [6]. SATzilla
applies per-instance feature and performance models to SAT solvers and also
measures feature computation cost [7]. It is useful context for both runtime
prediction and selector overhead, but it is not a direct comparator for
shortest-path algorithms.

The closest work found for this project is Adham, De Loera, and Zhang's study
of machine-learning selection for the **all-pairs** shortest-path problem [8].
They compare all-pairs Dijkstra, Floyd–Warshall, and Peng's algorithm on
generated graph families, using features that include graph density, node
count, and sampled degree information. Their study reports that density and
size help distinguish which algorithm performs well. This supports the
project's focus on instance features and measured runtime, but the task differs
in important ways: their problem is all-pairs, their tested edge weights are
positive, and their candidate set differs. ADSA selects one algorithm for a
single source-to-destination query, includes negative-weight cases, and
compares BFS, Dijkstra, A*, and Bellman–Ford. Results from the two studies are
not directly comparable.

Moore described maze-based shortest-path search [1]; Dijkstra presented a
shortest-path method for nonnegative lengths [2]; Bellman studied a routing
problem [3]; and Hart, Nilsson, and Raphael formalized heuristic search with an
optimality result for A* [4]. The code adapts these algorithmic ideas to a
common source-to-destination interface and applies explicit eligibility
checks before performance-based selection.

ASlib provides a benchmark representation for algorithm-selection studies,
while the later survey by Kerschke et al. places feature-based selection
alongside configuration and scheduling [9, 10]. Both reinforce that metrics
and data splits must be reported with the selector, rather than treating a
classifier's choice accuracy as the whole outcome. Learning-augmented SSSP
work is also relevant but addresses a different mechanism: predictions guide
the path algorithm's execution rather than select one algorithm from a
portfolio [11, 12].

The review is a focused scoping review. A first search protocol and screened
set are now recorded in `docs/LITERATURE_SEARCH_PROTOCOL.md`; the protocol
does not establish exhaustive coverage, and additional database searching and
citation checks are still needed before submission.

## System and model

The selector extracts these eight Model B features:

1. Node count, transformed with `log1p`.
2. Density.
3. Directedness.
4. Uniform-weight indicator.
5. Negative-weight indicator.
6. Weight standard deviation.
7. Whether positions are available for all nodes.
8. Scaled heuristic value, when available.

Each algorithm has its own Ridge model fitted to log median runtime. Training
features are standardized using development data. At query time, eligibility
checks remove invalid algorithms; the selector chooses the eligible algorithm
with the lowest positive finite predicted runtime. If none of the eligible
algorithms has a trained prediction, it uses the existing rule-based fallback.

## Algorithms, correctness, and complexity

The implementation uses adjacency-list BFS, heap-based Dijkstra and A*, and
edge relaxation for Bellman–Ford. For `V` vertices and `E` stored adjacency
entries, their conventional worst-case time bounds are O(V+E) for BFS,
O((V+E) log V) for heap-based Dijkstra and A*, and O(VE) for Bellman–Ford.
Lazy heap entries can require O(V+E) auxiliary storage for Dijkstra/A*;
Bellman–Ford also stores a flattened edge list, while BFS uses O(V) traversal
state. A*'s graph-specific heuristic scale may require an O(E) calculation the
first time it is requested; the graph caches that value thereafter.

The regression selector's inference operates on four small models with eight
inputs each; feature extraction and eligibility checks dominate this fixed
model-inference work. The benchmark validates returned path costs against the
individual algorithms for every case. The implementation also has tests for
negative-weight cycle behavior, eligibility, and selected-path costs, but this
report does not claim a formal proof of the implementation.

## Experimental method

### Development experiments

The development benchmark contains 252 generated graph cases. Feature
ablation used grouped five-fold cross-validation repeated with seeds 11, 22,
42, 101, and 202. Each graph case belonged to one validation fold within a
repeat. The alpha experiment kept Model B's features, development cases, seeds,
and folds fixed while varying alpha over 0.0001, 0.001, 0.01, 0.1, 1, 10,
and 100. It selected the highest mean non-trivial winner accuracy, breaking
ties by lower mean median absolute log error.

### Independent final evaluation

The final evaluation trained only from `development_benchmarks.csv`; earlier
held-out cases were treated as diagnostic data, not as the final test set. It
generated 672 new cases using seeds 303, 404, and 505, with 50, 100, 200, and
300 nodes; densities 0.03, 0.10, 0.30, and 0.55; unit, uniform, positive, and
negative weights; and coordinate/directedness settings supported by the
generator. Negative-weight cases were directed. Each case used source `0` and
destination `n-1`, with three timed repeats. The same graph cases were measured
for Ridge, the rule selector, and every eligible individual algorithm.

Winner accuracy counts cases where the selector chose an algorithm whose
measured median runtime was the minimum among eligible algorithms. Non-trivial
winner accuracy excludes cases with only one eligible algorithm. Prediction
error is absolute log(predicted runtime / measured runtime). The reported
algorithm speedup is fastest eligible algorithm runtime divided by adaptive
end-to-end runtime; values above 1 favor the adaptive selector. The paired
rule/Ridge speedup is rule-selector end-to-end runtime divided by Ridge
end-to-end runtime; values above 1 favor Ridge. Timings are machine-dependent.
The benchmark harness extracts full features before timing adaptive execution,
which warms the graph's cached heuristic scale. Therefore selection and
feature-extraction timings here represent a warmed graph state; a cold first
query may incur additional feature work.

## Results

### Feature ablation

Values are mean across the five repeated cross-validation seeds; `±` is the
sample standard deviation across those seeds.

| Model | Winner accuracy | Non-trivial winner accuracy | Median absolute log error |
|---|---:|---:|---:|
| A | 92.14% ± 0.76 pp | 90.83% ± 0.89 pp | 0.2583 ± 0.0037 |
| B | **93.02% ± 0.60 pp** | **91.85% ± 0.70 pp** | 0.2582 ± 0.0058 |
| C | 92.14% ± 0.71 pp | 90.83% ± 0.83 pp | 0.2575 ± 0.0063 |
| D | 92.22% ± 0.53 pp | 90.93% ± 0.62 pp | **0.2574 ± 0.0061** |

Model B had the strongest winner-accuracy results; Models C and D had slightly
lower runtime-prediction error. Model B was therefore a provisional feature
choice based on selection accuracy, not a winner on every metric.

### Ridge alpha tuning

| Alpha | Winner accuracy | Non-trivial winner accuracy | Median absolute log error |
|---:|---:|---:|---:|
| 0.0001 | **93.10% ± 0.53 pp** | **91.94% ± 0.62 pp** | 0.2581 ± 0.0053 |
| 0.001 | 93.02% ± 0.60 pp | 91.85% ± 0.70 pp | 0.2582 ± 0.0058 |
| 0.01 | 92.54% ± 0.43 pp | 91.30% ± 0.51 pp | 0.2582 ± 0.0052 |
| 0.1 | 92.22% ± 0.53 pp | 90.93% ± 0.62 pp | 0.2574 ± 0.0057 |
| 1 | 92.14% ± 0.71 pp | 90.83% ± 0.83 pp | **0.2560 ± 0.0063** |
| 10 | 91.83% ± 1.00 pp | 90.46% ± 1.16 pp | 0.2598 ± 0.0054 |
| 100 | 87.94% ± 0.35 pp | 85.93% ± 0.41 pp | 0.4061 ± 0.0020 |

Alpha 0.0001 was selected by the stated accuracy-first rule. Alpha 1 had the
lowest median absolute log error, so the selected value reflects the declared
priority on choosing the fastest algorithm, not uniformly best prediction
quality.

### Final evaluation

| Measure | Ridge, alpha 0.0001 | Rule selector |
|---|---:|---:|
| Graph cases | 672 | 672 |
| Non-trivial cases | 576 | 576 |
| Winner accuracy | **92.71%** | 79.61% |
| Non-trivial winner accuracy | **91.49%** | 76.22% |
| Median selection decision time | 8.0 μs | 561 μs |
| Median estimated selection overhead | 8.7 μs | 582 μs |
| Median peak memory | 11.6 KB | 246.7 KB |
| All returned costs verified | Yes | Yes |

Ridge was faster than rules in 671 of 672 paired cases. The median paired
rule/Ridge speedup was 5.69×. Against the fastest individual algorithm, Ridge's
median adaptive runtime was 100.8 μs versus 86.1 μs for the fastest algorithm.
Its median fastest-algorithm/adaptive speedup was 0.887× (equivalently, the
adaptive-to-best runtime ratio was 1.127×). Rules had a median adaptive-to-best
runtime ratio of 7.664×. Thus the model improved on the rule baseline but
selection and execution together remained slower than directly running the
fastest eligible algorithm at the median.

Winner accuracy by weight mode was:

| Weight mode | Ridge | Rules | Cases |
|---|---:|---:|---:|
| Unit | 89.06% | 79.69% | 192 |
| Uniform | 91.15% | 79.69% | 192 |
| Positive | 94.27% | 69.27% | 192 |
| Negative | 100.00% | 100.00% | 96 |

All negative-weight cases had only Bellman–Ford eligible, so their 100% rate is
not evidence that the selectors distinguished between competing algorithms.
The category-level CSV provides additional breakdowns by size, density,
directedness, and coordinates.

Accuracy was similar across the three fresh seeds, with some variation:

| Selector | Test seed | Winner accuracy | Non-trivial winner accuracy | Median adaptive-to-best runtime ratio |
|---|---:|---:|---:|---:|
| Ridge | 303 | 93.30% | 92.19% | 1.132× |
| Ridge | 404 | 91.07% | 89.58% | 1.126× |
| Ridge | 505 | 93.75% | 92.71% | 1.129× |
| Rules | 303 | 78.57% | 75.00% | 7.934× |
| Rules | 404 | 81.70% | 78.65% | 7.551× |
| Rules | 505 | 78.57% | 75.00% | 7.659× |

Mean absolute log prediction errors on the fresh test cases were 0.524 for
BFS, 0.448 for Dijkstra, 0.736 for A*, and 0.321 for Bellman–Ford. A* had the
largest error and a median prediction-to-actual ratio of 0.845, indicating
underprediction on the median case. Across all predictions, the median
absolute log error was 0.364.

## Discussion

The final data supports the Ridge selector over the current rule baseline on
winner accuracy, decision latency, and measured peak memory for this generated
workload. The rule baseline's full graph-feature calculation, including
connectivity work, is costly at the tested sizes. The lightweight feature
extractor used by the learned selector had a median measured extraction time
of 0.709 μs, compared with 584.7 μs for full feature extraction (about 825×
lower by ratio of medians in this run).

This does not make the Ridge selector the fastest end-to-end option in this
experiment. Its median overhead exceeded the median time it saved by selecting
among algorithms, so running the fastest algorithm directly was still faster
at the median. Also, negative-weight cases provide no meaningful choice
comparison because only Bellman–Ford is eligible. The positive, uniform, and
unit-weight cases are the informative selection comparisons.

## Conclusion

On this synthetic test set, Model B with alpha 0.0001 chose the measured
fastest eligible algorithm more often than the rule selector, and its
end-to-end execution was substantially faster than the rule selector's. The
results do not show an end-to-end speed advantage over running the fastest
eligible algorithm directly. The supported conclusion is therefore narrower:
the learned selector improves decision accuracy and performance relative to
the implemented rule baseline for these generated cases, while its overhead
still matters when the alternative is direct execution of the best algorithm.

## Limitations and next work

- The final cases are synthetic and use the repository's graph generator; real
  application graphs may differ.
- New seeds make this a separate test set from prior runs, but it uses the
  same generator family. Only the graph-size and density grid was broadened.
- Each graph uses one fixed source/destination pair and three timing repeats.
- Results come from one macOS ARM64 environment (Python 3.14.7); timings may
  change on other machines or workloads.
- Timed adaptive calls use warmed graph feature caches. Cold-start latency
  needs a separate evaluation if first-query cost is important.
- The alpha choice prioritizes winner accuracy. A* prediction error remains
  comparatively high, and other metrics favor other alpha values.
- A measured runtime advantage over the fastest individual algorithm was not
  established. Future work should study whether lower-overhead selection,
  more workload-specific features, or real graph datasets can change this.
- The supplied project notes say a Union-Find connectivity optimization was
  rejected because its overhead outweighed its benefit, but this repository
  contains no quantitative result artifact for that experiment. The claim
  should remain qualitative until its measurements are recovered.
- The related-work section is a focused scoping review, not a systematic
  survey. Broader database searching and citation verification are still
  needed before submission.

## Reproducibility files

- Feature-ablation code and results: `backend/experiments/feature_ablation_cv.py`,
  `backend/results/feature_ablation_repeated_cv.csv`.
- Alpha-tuning code and results: `backend/experiments/ridge_alpha_cv.py`,
  `backend/results/ridge_alpha_cv.csv`.
- Final evaluation code, configuration, case-level data, predictions, and
  summaries: `backend/experiments/final_evaluation.py` and the
  `backend/results/final_evaluation_*.{csv,json}` files, including
  `final_evaluation_seed_summary.csv` for seed-to-seed variation.
- Test suite: 33 tests passed after Step 8 integration.
- Optimization decisions and measurement caveats: `docs/OPTIMIZATION_LOG.md`.

## References

1. E. F. Moore, “The Shortest Path Through a Maze,” *Proceedings of the
   International Symposium on the Theory of Switching*, Part II, pp. 285–292,
   1959. [Bibliographic record](https://cir.nii.ac.jp/crid/1570854174823707776?lang=en).
2. E. W. Dijkstra, “A Note on Two Problems in Connexion with Graphs,”
   *Numerische Mathematik*, vol. 1, pp. 269–271, 1959.
   [doi:10.1007/BF01386390](https://doi.org/10.1007/BF01386390).
3. R. Bellman, “On a Routing Problem,” *Quarterly of Applied Mathematics*,
   vol. 16, pp. 87–90, 1958. [doi:10.1090/qam/102435](https://doi.org/10.1090/qam/102435).
4. P. E. Hart, N. J. Nilsson, and B. Raphael, “A Formal Basis for the
   Heuristic Determination of Minimum Cost Paths,” *IEEE Transactions on
   Systems Science and Cybernetics*, vol. 4, no. 2, pp. 100–107, 1968.
   [doi:10.1109/TSSC.1968.300136](https://doi.org/10.1109/TSSC.1968.300136).
5. J. R. Rice, “The Algorithm Selection Problem,” *Advances in Computers*,
   vol. 15, pp. 65–118, 1976.
   [doi:10.1016/S0065-2458(08)60520-3](https://doi.org/10.1016/S0065-2458(08)60520-3).
6. K. Leyton-Brown, E. Nudelman, G. Andrew, J. McFadden, and Y. Shoham,
   “A Portfolio Approach to Algorithm Selection,” in *Proceedings of the
   18th International Joint Conference on Artificial Intelligence*, pp.
   1542–1543, 2003. [Proceedings paper](https://ijcai.org/Proceedings/03/Papers/268.pdf).
7. L. Xu, F. Hutter, H. H. Hoos, and K. Leyton-Brown, “SATzilla:
   Portfolio-based Algorithm Selection for SAT,” *Journal of Artificial
   Intelligence Research*, vol. 32, pp. 565–606, 2008.
   [doi:10.1613/JAIR.2490](https://doi.org/10.1613/JAIR.2490).
8. I. Adham, J. De Loera, and Z. Zhang, “(Machine) Learning to Improve the
   Empirical Performance of Discrete Algorithms,” arXiv:2109.14271, 2021.
   [arXiv preprint](https://arxiv.org/abs/2109.14271).
9. B. Bischl et al., “ASlib: A Benchmark Library for Algorithm Selection,”
   *Artificial Intelligence*, vol. 237, pp. 41–58, 2016.
   [doi:10.1016/j.artint.2016.04.003](https://doi.org/10.1016/j.artint.2016.04.003).
10. P. Kerschke et al., “Automated Algorithm Selection: Survey and
    Perspectives,” *Evolutionary Computation*, vol. 27, no. 1, pp. 3–45, 2019.
    [doi:10.1162/evco_a_00242](https://doi.org/10.1162/evco_a_00242).
11. J. Y. Chen, S. Silwal, A. Vakilian, and F. Zhang, “Faster Fundamental Graph
    Algorithms via Learned Predictions,” arXiv:2204.12055, 2022.
    [arXiv preprint](https://arxiv.org/abs/2204.12055).
12. W. Feijen and G. Schäfer, “Dijkstra's Algorithm with Predictions to Solve
    the Single-Source Many-Targets Shortest-Path Problem,” arXiv:2112.11927,
    2021. [arXiv preprint](https://arxiv.org/abs/2112.11927).
    [arXiv record](https://arxiv.org/abs/2112.11927).
