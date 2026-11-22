# Real-world graph evaluation

## Objective and procedure

This experiment applies the frozen Model B Ridge selector (alpha `0.0001`,
trained only on `development_benchmarks.csv`) to five DIMACS graph inputs and
compares it with the rule selector on the same query pairs. It performs no
model fitting or tuning. Each input uses 20 reproducibly selected reachable
near/far source/destination pairs, three timing repeats, a shared set of
measured eligible algorithms, and path-cost checks.

The full DIMACS road networks are too large for the project's pure-Python
Bellman–Ford loop. The evaluator therefore selects the largest weak component
visible among arcs whose endpoint IDs are at most 20,000, traverses from a
deterministic root, keeps a connected prefix of up to 200 nodes, and evaluates
the induced subgraph. All included edge weights are taken from the source
files. When the DIMACS input has parallel arcs, the sample keeps the minimum
weight for each ordered endpoint pair; this preserves shortest-path costs for
these nonnegative graphs and accommodates the project's simple-graph `Graph`
class. These are bounded graph samples, not measurements on the complete
regional networks.

## Data provenance and coverage

The five tested graph inputs cover four regions: New York distance, Colorado
distance, Colorado transit time, Florida distance, and Rome 1999 distance.
Colorado's distance and travel-time inputs share a topology, so they are two
cost variants rather than independent geographic networks. The DIMACS
Challenge 9 page describes its USA road networks and says its files are public
domain unless otherwise stated. It also warns of source-network gaps and
bridges. The original data-file host returned 404 during this run, so the
downloaded graph files came from public GitHub mirrors that identify the
original DIMACS source. SHA-256 hashes and mirror URLs are in
`backend/results/real_world_dataset_manifest.csv`.

Coordinates are available for New York and Colorado. A* is eligible on those
two inputs. Verified mirrors with coordinates were not available for Florida
or Rome99 in this run, so A* is excluded there; Dijkstra and Bellman–Ford are
compared. BFS is ineligible because the distance/time weights are not uniform.

## Results

| Input | Ridge winner accuracy | Rules winner accuracy | Ridge median adaptive/oracle | Rules median adaptive/oracle |
|---|---:|---:|---:|---:|
| New York distance | 15% | 85% | 1.506× | 4.316× |
| Colorado distance | 0% | 100% | 1.296× | 4.407× |
| Colorado transit time | 5% | 95% | 1.390× | 4.586× |
| Florida distance | 100% | 100% | 1.129× | 3.500× |
| Rome99 distance | 100% | 100% | 1.078× | 2.884× |
| All five graph inputs | **44%** | **96%** | **1.221×** | **3.676×** |

All returned path costs matched across the eligible algorithms. The aggregate
accuracy is query-weighted across 100 endpoint pairs. The typical Ridge
selection chose an algorithm 1.079× the measured oracle runtime before
selection overhead; after selection overhead, its median adaptive/oracle
ratio was 1.221×. The rule selector usually chose the measured fastest
algorithm here, but its expensive feature and connectivity calculation made
its end-to-end median 3.676× the oracle runtime.

The results do **not** support synthetic-to-real generalization of the current
Ridge selector on these samples. Performance also varies sharply by input:
Ridge was competitive on Florida and Rome99, and substantially less accurate
on the New York and Colorado samples. This is an exploratory result, not a
five-independent-network estimate: only four geographic networks were
available, Colorado contributes two correlated cost variants, and each
network contributes 20 queries from one 200-node sampled subgraph. The next
evaluation should obtain at least one more independent network, run larger or
full-network queries with an implementation that can measure Bellman–Ford at
that scale, and use graph-clustered uncertainty estimates.

## Reproduce

```bash
PYTHONPATH=backend python backend/experiments/real_world_evaluation.py
```

The script downloads inputs to ignored `data/real_world/dimacs9/` if absent.
It writes new artifacts only:

- `backend/results/real_world_dataset_manifest.csv`
- `backend/results/real_world_benchmarks.csv`
- `backend/results/real_world_predictions.csv`
- `backend/results/real_world_query_results.csv`
- `backend/results/real_world_summary.csv`

## Sources

- [DIMACS Challenge 9 download page](https://www.diag.uniroma1.it/~challenge9/download.shtml)
- [DIMACS Challenge 9 graph format](https://www.diag.uniroma1.it/challenge9/format.shtml)
- [NY/COL/FLA mirror and reconstruction notes](https://github.com/blxrua/ads_project2_ylb/tree/main/data/dimacs)
- [NY coordinate mirror](https://github.com/arnauddesombre/Python-Utilities)
- [COL distance/time and coordinate mirror](https://github.com/wmsnp/cch-rs/tree/master/tests/data)
- [Rome99 graph and attribution](https://github.com/patrickenfuego/rome99-graph-traversal)
