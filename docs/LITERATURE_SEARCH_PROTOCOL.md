# Literature search protocol

## Status and scope

This is a documented, reproducible starting protocol for expanding the ADSA
related-work review. The search was conducted on 22 November 2026. It is a
focused scoping search, not a systematic review: no claim of exhaustive
coverage, duplicate-screening by multiple reviewers, or formal risk-of-bias
assessment is made.

The question is: **What prior work studies per-instance algorithm selection
or learned performance prediction for graph algorithms, particularly
shortest paths, and what methodological guidance applies to measuring
selection overhead and generalization?**

## Sources and search strings

Searches used scholarly web indexing and primary publisher, proceedings, or
author-hosted paper pages. Queries were run on 22 November 2026:

| Search string | Intended coverage |
|---|---|
| `machine learning algorithm selection shortest path algorithms graph single source shortest path paper algorithm selection` | Direct learned shortest-path selection, especially single-source/single-pair work |
| `machine learning algorithm selection graph algorithms shortest path algorithm selection Adham` | Follow-up and closely related graph algorithm-selection research |
| `algorithm selection problem Rice 1976 computational problem solving environments` | Foundational formalization |
| `SATzilla algorithm portfolio per-instance algorithm selection feature computation cost paper` | Established runtime-prediction portfolios and feature-cost treatment |
| `Bischl ASlib benchmark library algorithm selection Artificial Intelligence 2016` | Benchmark/reporting conventions |
| `automated algorithm selection survey perspectives 2019 Kerschke` | Broader terminology and related directions |

The search was supplemented by following references from the located survey,
benchmark paper, and the directly related shortest-path selection study.
Useful starting sources include Rice's foundational chapter, the
Kerschke et al. survey, the ASlib benchmark paper, SATzilla, and Adham et al.'s
all-pairs shortest-path selection study. Stable links and bibliographic
records are listed in `docs/RESEARCH_REPORT.md`.

## Eligibility rules

Include a source when it meets at least one of these criteria:

1. It formulates per-instance algorithm selection or a runtime-model portfolio.
2. It empirically selects among graph algorithms, shortest-path algorithms,
   or related graph-computation methods using instance features.
3. It provides established evaluation or benchmark methodology for learned
   algorithm selectors, especially feature cost, train/test separation,
   paired outcomes, or portfolio baselines.

Prefer peer-reviewed papers, archival proceedings, publisher records, and
author-hosted versions of identifiable scholarly work. Preprints may be used
when they are the accessible version of relevant work; mark them as
preprints and check for a later publication before submission.

## Exclusion rules

Exclude sources that only:

- present a shortest-path algorithm without discussing algorithm selection
  or empirical comparative performance;
- use ML to predict path distances, edge weights, or route quality inside a
  single algorithm without choosing among candidate algorithms;
- study a different task (for example, all-pairs shortest paths) without
  clearly labeling it as adjacent rather than directly comparable;
- are informal tutorials, unverified summaries, or commercial pages when a
  primary paper or authoritative documentation is available.

## Screening and extraction

For each candidate, record: citation and version; problem/query setting;
candidate algorithms; instance distribution; features and their computation
cost; learning or selection method; baseline/oracle; evaluation split and
repetitions; performance objective; and limitations relevant to ADSA.
Screen titles and abstracts first, then inspect the full paper or authoritative
record for every source used to support a technical claim. Record whether the
result is directly comparable or only methodological context.

## Initial screened set and findings

| Work | Relation to ADSA | Use in review |
|---|---|---|
| Rice (1976), *The Algorithm Selection Problem* | Foundational formulation of mapping measurable instance features to algorithm choices | Conceptual basis; not empirical evidence for shortest paths |
| Xu et al. (2008), SATzilla | Per-instance runtime models, learned portfolios, and explicit feature-cost considerations in SAT | Methodological context; different application domain |
| Bischl et al. (2016), ASlib | Benchmark library and common experimental representation for algorithm selection | Evaluation and reproducibility context |
| Kerschke et al. (2019), algorithm-selection survey | Broad survey across discrete/continuous domains and related concepts | Discovery source and terminology; not a shortest-path result |
| Adham, De Loera, and Zhang (2021), learned discrete-algorithm selection | Directly adjacent: selects among algorithms for all-pairs shortest paths using graph features | Closest located work; distinguish APSP from ADSA's single-pair task |
| Dijkstra et al. (2021), learning-augmented SSSP | Uses predictions within a shortest-path algorithm for a single-source/multiple-target setting | Adjacent learned graph algorithm work, not portfolio selection |
| Haeupler et al. (2022), faster graph algorithms via learned predictions | Uses learned predictions to improve graph algorithms, including SSSP | Adjacent learning-augmented algorithms; different claim and setup |

The initial search identifies a clear task distinction: the closest located
selection study addresses all-pairs shortest paths, while learning-augmented
single-source work changes an algorithm using predictions rather than selecting
among BFS, Dijkstra, A*, and Bellman–Ford. The present search is not enough to
claim that no prior single-pair selector exists. That claim requires broader
database searches and citation chaining before submission.

## Search limitations and next review pass

- Search coverage is limited to targeted scholarly web queries and reference
  chaining; no Scopus, Web of Science, ACM DL, or IEEE Xplore institutional
  database export was available in this pass.
- Search results can miss alternate terms such as solver portfolio, graph
  workload classification, adaptive routing, and per-instance performance
  prediction.
- Before submission, search at least two bibliographic databases, record
  exact query syntax and result counts, deduplicate records, screen full texts,
  check publication versions/DOIs, and update the search date.
- Do not describe this document or the current related-work section as a
  systematic review.
