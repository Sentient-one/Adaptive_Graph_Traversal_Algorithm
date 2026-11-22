# Literature review: algorithm selection for shortest paths

## Scope and status

This is a focused, source-checked scoping review for the ADSA project, updated
22 November 2026. It is not a systematic review: the project did not have
institutional access to Scopus, Web of Science, ACM Digital Library, or IEEE
Xplore result exports, and screening was not performed by two independent
reviewers. Search scope and limitations are recorded in
`docs/LITERATURE_SEARCH_PROTOCOL.md`. The comparison table is in
`docs/LITERATURE_COMPARISON.csv`.

The question is whether prior work selects among shortest-path algorithms for
an individual single-pair query using graph-level features, and what
methodological precedents apply to learned runtime portfolios.

## Algorithm selection and performance portfolios

Rice's formulation frames algorithm selection as mapping measurable instance
properties to an algorithm expected to perform well. This is the conceptual
basis for ADSA's per-graph feature vector and selection among eligible
algorithms. Portfolio methods then operationalize the idea by estimating
candidate performance and choosing among solvers. SATzilla is an established
example in SAT solving; its evaluation also makes the cost of computing
features part of the practical selector design. These works establish the
general method, but their domains and algorithms differ from shortest paths.

ASlib provides a benchmark representation for algorithm-selection studies,
and the survey by Kerschke et al. describes the field's broader terminology
and evaluation approaches. They motivate recording the instance set, split,
features, candidate portfolio, runtime objective, and selector overhead.
ADSA follows that direction by preserving development/final separation,
reporting paired outcomes, and measuring decision cost separately from
algorithm runtime. Its small project-specific experiment is not an ASlib
benchmark submission.

## Shortest-path and graph-algorithm evidence

The closest located portfolio study is Adham, De Loera, and Zhang's
data-driven selection among algorithms for the all-pairs shortest-path
problem. It is direct evidence that graph features can help distinguish
algorithm performance, but it is not a direct comparator for ADSA: it studies
all-pairs paths, uses a different portfolio, and does not answer whether a
selector helps for one source-destination query. It supports investigating
graph density and size while leaving the single-pair question open.

Feijen and Schäfer study predictions used inside a Dijkstra-style algorithm
for a single-source, many-targets task. Chen et al. study learned predictions
as advice for speeding up graph algorithms, including single-source shortest
paths with negative lengths. Both are relevant to the broader relationship
between learning and shortest paths, but neither is the same mechanism as
ADSA: they augment an algorithm's search rather than rank BFS, Dijkstra, A*,
and Bellman–Ford as a portfolio. These distinctions matter when stating the
research contribution; ADSA does not claim that learning-based shortest-path
methods or algorithm selection are new in general.

## Implications for ADSA

The prior work supports treating selection as a cost-sensitive decision.
Winner accuracy alone does not establish faster end-to-end execution; feature
extraction and prediction costs must be included, and results should be
compared on the same graph/query cases against both a transparent baseline
and the measured fastest eligible algorithm. The current evidence follows
these comparisons on its synthetic final set. Ridge improves winner accuracy
over the project's rule selector there, but its median end-to-end runtime is
still 1.127 times the fastest eligible algorithm. On the bounded road-network
sample, accuracy reverses in favor of rules. Thus the results support a
synthetic-workload finding, not a general claim of selector superiority.

The literature also identifies two separate future questions: whether
single-pair portfolio selection generalizes to independent real graph
distributions, and whether lower-cost decision mechanisms can reduce regret
relative to direct execution. The current evaluation is not large or diverse
enough to answer the first question conclusively.

## Search limitations and next review pass

The focused search identified a close all-pairs selector paper and adjacent
learning-augmented shortest-path work, but cannot establish that no prior
single-pair selector exists. Before submission, search at least two
bibliographic databases using the query variants in the protocol; record
query dates, result counts, and screening decisions; follow citations from
the directly relevant work; verify final publication versions and DOIs; and
have the inclusion decisions checked by a second reviewer if feasible.

## Primary and foundational sources

- Rice, J. R. (1976). “The Algorithm Selection Problem.” *Advances in
  Computers*, 15, 65–118. [DOI](https://doi.org/10.1016/S0065-2458(08)60520-3).
- Leyton-Brown, K. et al. (2003). “A Portfolio Approach to Algorithm
  Selection.” IJCAI. [Proceedings paper](https://ijcai.org/Proceedings/03/Papers/268.pdf).
- Xu, L. et al. (2008). “SATzilla: Portfolio-based Algorithm Selection for
  SAT.” *Journal of Artificial Intelligence Research*, 32, 565–606.
  [DOI](https://doi.org/10.1613/JAIR.2490).
- Bischl, B. et al. (2016). “ASlib: A Benchmark Library for Algorithm
  Selection.” *Artificial Intelligence*, 237, 41–58.
  [DOI](https://doi.org/10.1016/j.artint.2016.04.003).
- Kerschke, P. et al. (2019). “Automated Algorithm Selection: Survey and
  Perspectives.” *Evolutionary Computation*, 27(1), 3–45.
  [DOI](https://doi.org/10.1162/evco_a_00242).
- Adham, I., De Loera, J., and Zhang, Z. (2021). “(Machine) Learning to
  Improve the Empirical Performance of Discrete Algorithms.” [arXiv:2109.14271](https://arxiv.org/abs/2109.14271).
- Feijen, W. and Schäfer, G. (2021). “Dijkstra's Algorithm with Predictions
  to Solve the Single-Source Many-Targets Shortest-Path Problem.”
  [arXiv:2112.11927](https://arxiv.org/abs/2112.11927).
- Chen, J. Y. et al. (2022). “Faster Fundamental Graph Algorithms via Learned
  Predictions.” [arXiv:2204.12055](https://arxiv.org/abs/2204.12055).
