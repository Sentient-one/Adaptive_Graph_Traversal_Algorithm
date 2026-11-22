"""Interactive Streamlit demonstration of the ADSA shortest-path selector."""

import math
import random
import statistics
from pathlib import Path
from time import perf_counter

import streamlit as st

from adaptive.features import extract_features, extract_selection_features
from adaptive.performance_model import BenchmarkRuntimeSelector
from adaptive.selector import (
    ALGORITHM_IMPLEMENTATIONS,
    choose_algorithm,
    eligible_algorithms,
    find_shortest_path,
)
from graph.graph import Graph


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEVELOPMENT_RESULTS = PROJECT_ROOT / "backend" / "results" / "development_benchmarks.csv"
ALGORITHM_LABELS = {
    "astar": "A*",
    "bellman_ford": "Bellman–Ford",
    "bfs": "BFS",
    "dijkstra": "Dijkstra",
}

st.set_page_config(page_title="ADSA shortest-path demo", page_icon="🧭", layout="wide")


@st.cache_resource
def _load_model(csv_path: str, modified_ns: int):
    del modified_ns  # Included in the cache key so updated development data reloads.
    return BenchmarkRuntimeSelector.from_csv(csv_path)


def build_demo_graph(
    scenario: str,
    node_count: int,
    extra_edge_probability: float,
    seed: int,
) -> Graph:
    """Build a reproducible connected mesh or a negative-weight DAG."""
    rng = random.Random(seed)
    negative = scenario == "Negative-edge directed graph"
    graph = Graph(directed=negative)
    width = math.ceil(math.sqrt(node_count))

    for index in range(node_count):
        node = str(index)
        graph.add_node(node)
        if not negative:
            graph.set_position(node, index % width, index // width)

    if negative:
        # A forward-only DAG guarantees a route from 0 to n-1 and cannot have
        # negative cycles. Some edges are negative to exercise Bellman–Ford.
        for index in range(node_count - 1):
            weight = rng.choice((-2.0, -1.0, 2.0, 4.0))
            graph.add_edge(str(index), str(index + 1), weight)
        for start in range(node_count - 2):
            if rng.random() < extra_edge_probability:
                graph.add_edge(
                    str(start), str(start + 2), rng.choice((-1.0, 1.0, 3.0))
                )
        return graph

    unit_weights = scenario == "Unit-weight mesh"
    existing = set()

    def add_undirected_edge(first: int, second: int) -> None:
        key = (min(first, second), max(first, second))
        if key in existing:
            return
        existing.add(key)
        weight = 1.0 if unit_weights else rng.uniform(1.1, 10.0)
        graph.add_edge(str(first), str(second), weight)

    # A grid backbone keeps every node connected and gives A* coordinates.
    for index in range(node_count):
        row, column = divmod(index, width)
        right = index + 1
        below = index + width
        if column + 1 < width and right < node_count:
            add_undirected_edge(index, right)
        if below < node_count:
            add_undirected_edge(index, below)

    possible_pairs = node_count * (node_count - 1) // 2
    extra_count = min(
        possible_pairs - len(existing),
        int(extra_edge_probability * possible_pairs),
    )
    target_edge_count = len(existing) + extra_count
    attempts = 0
    while len(existing) < target_edge_count and attempts < extra_count * 20 + 1:
        first, second = rng.sample(range(node_count), 2)
        add_undirected_edge(first, second)
        attempts += 1

    return graph


def _matches(first, second) -> bool:
    if first is None or second is None:
        return first is second
    return math.isclose(first, second, rel_tol=1e-9, abs_tol=1e-9)


def _path_cost(graph: Graph, path):
    if path is None:
        return None
    cost = 0.0
    for source, destination in zip(path, path[1:]):
        matches = [
            weight
            for neighbor, weight in graph.neighbors(source)
            if neighbor == destination
        ]
        if not matches:
            raise AssertionError("Returned path contains a nonexistent edge")
        cost += matches[0]
    return cost


def _median_call(function, repeats: int):
    function()  # Correctness-checked separately; this warms interpreter state.
    elapsed = []
    result = None
    for _ in range(repeats):
        started = perf_counter()
        result = function()
        elapsed.append(perf_counter() - started)
    return statistics.median(elapsed), result


def run_comparison(
    graph: Graph,
    source: str,
    destination: str,
    repeats: int,
    model: BenchmarkRuntimeSelector,
):
    features = extract_features(graph)
    selection_features = extract_selection_features(graph)
    eligible = eligible_algorithms(
        graph, source, destination, features=selection_features
    )
    if not eligible:
        raise ValueError("No supported algorithm is eligible for this query.")

    predictions = model.predict_runtimes_from_features(selection_features)
    choices = {
        name: (
            "Eligible"
            if name in eligible
            else "Not eligible for these graph weights/coordinates"
        )
        for name in ALGORITHM_IMPLEMENTATIONS
    }
    model_choice = model.choose_algorithm(graph, source, destination)
    rule_choice = choose_algorithm(graph, source, destination)

    algorithm_timings = {name: [] for name in eligible}
    algorithm_results = {}
    reference_cost = None
    have_reference = False
    ordered_names = list(eligible)

    # Warm each implementation and verify costs before collecting timings.
    for name in ordered_names:
        result = eligible[name](graph, source, destination)
        if not have_reference:
            reference_cost = result["cost"]
            have_reference = True
        elif not _matches(reference_cost, result["cost"]):
            raise AssertionError(f"{name} disagrees on shortest-path cost")
        if not _matches(result["cost"], _path_cost(graph, result["path"])):
            raise AssertionError(f"{name} returned a path with an invalid cost")
        algorithm_results[name] = result

    # Rotate execution order across repeats to reduce fixed ordering effects.
    for repeat in range(repeats):
        order = ordered_names[repeat % len(ordered_names):] + ordered_names[:repeat % len(ordered_names)]
        for name in order:
            started = perf_counter()
            result = eligible[name](graph, source, destination)
            algorithm_timings[name].append(perf_counter() - started)
            if not _matches(reference_cost, result["cost"]):
                raise AssertionError(f"{name} changed shortest-path cost")

    model_decision_seconds, _ = _median_call(
        lambda: model.choose_algorithm(graph, source, destination), repeats
    )
    rule_decision_seconds, _ = _median_call(
        lambda: choose_algorithm(graph, source, destination), repeats
    )
    model_end_to_end_seconds, model_result = _median_call(
        lambda: find_shortest_path(
            graph, source, destination, performance_model=model
        ),
        repeats,
    )
    rule_end_to_end_seconds, rule_result = _median_call(
        lambda: find_shortest_path(graph, source, destination), repeats
    )
    if not _matches(reference_cost, model_result["cost"]):
        raise AssertionError("Model selector returned an inconsistent path cost")
    if not _matches(reference_cost, rule_result["cost"]):
        raise AssertionError("Rule selector returned an inconsistent path cost")

    timings = {
        name: statistics.median(samples)
        for name, samples in algorithm_timings.items()
    }
    oracle = min(timings, key=timings.get)
    return {
        "features": features,
        "selection_features": selection_features,
        "eligible": list(eligible),
        "predictions": predictions,
        "choices": choices,
        "model_choice": model_choice,
        "rule_choice": rule_choice,
        "algorithm_results": algorithm_results,
        "algorithm_timings": timings,
        "oracle": oracle,
        "reference_cost": reference_cost,
        "model_result": model_result,
        "rule_result": rule_result,
        "model_decision_seconds": model_decision_seconds,
        "rule_decision_seconds": rule_decision_seconds,
        "model_end_to_end_seconds": model_end_to_end_seconds,
        "rule_end_to_end_seconds": rule_end_to_end_seconds,
    }


def _format_seconds(seconds: float) -> str:
    if seconds < 1e-3:
        return f"{seconds * 1e6:.2f} μs"
    if seconds < 1:
        return f"{seconds * 1e3:.2f} ms"
    return f"{seconds:.3f} s"


def main():
    st.title("🧭 Adaptive shortest-path selection")
    st.write(
        "Explore how graph properties affect eligibility, runtime predictions, "
        "and the selected shortest-path algorithm."
    )

    if not DEVELOPMENT_RESULTS.exists():
        st.error(f"Development benchmark data not found: {DEVELOPMENT_RESULTS}")
        st.stop()

    with st.sidebar:
        st.header("Graph and query")
        scenario = st.selectbox(
            "Graph scenario",
            (
                "Unit-weight mesh",
                "Positive weighted mesh",
                "Negative-edge directed graph",
            ),
        )
        node_count = st.slider("Number of nodes", 8, 100, 36, 1)
        extra_edge_probability = st.slider(
            "Extra edges as share of possible pairs",
            0.0,
            0.20,
            0.04,
            0.01,
            help="For mesh graphs, this is the share of all possible pairs added on top of the backbone. For the negative DAG, it is the chance of adding each forward skip edge.",
        )
        seed = st.number_input("Generation seed", 0, 2**31 - 1, 42)
        source = st.selectbox("Source", [str(i) for i in range(node_count)], index=0)
        destination = st.selectbox(
            "Destination", [str(i) for i in range(node_count)], index=node_count - 1
        )
        repeats = st.slider("Timing repeats", 3, 15, 7, 2)
        submit = st.button("Generate graph and compare", type="primary", use_container_width=True)

    st.caption(
        "The graph is generated locally. Each eligible algorithm is run on the "
        "same graph and query, and its returned path cost is checked."
    )

    if submit:
        st.session_state.pop("demo_result", None)
        try:
            graph = build_demo_graph(
                scenario, node_count, extra_edge_probability, int(seed)
            )
            model = _load_model(
                str(DEVELOPMENT_RESULTS), DEVELOPMENT_RESULTS.stat().st_mtime_ns
            )
            result = run_comparison(graph, source, destination, repeats, model)
            st.session_state["demo_result"] = result
            st.session_state["demo_config"] = {
                "scenario": scenario,
                "nodes": node_count,
                "seed": int(seed),
                "source": source,
                "destination": destination,
                "repeats": repeats,
            }
        except Exception as error:
            st.error(f"Could not complete the comparison: {error}")

    result = st.session_state.get("demo_result")
    config = st.session_state.get("demo_config")
    if result is None:
        st.info("Choose a graph scenario and run the comparison to see results.")
        return

    st.subheader(f"{config['scenario']} · seed {config['seed']}")
    st.caption(
        f"{config['nodes']} nodes · source {config['source']} → "
        f"destination {config['destination']} · {config['repeats']} timing repeats"
    )

    left, middle, right = st.columns(3)
    left.metric("Model-selected algorithm", ALGORITHM_LABELS[result["model_choice"]])
    middle.metric("Rule-selected algorithm", ALGORITHM_LABELS[result["rule_choice"]])
    right.metric("Verified path cost", str(result["reference_cost"]))

    feature_col, eligibility_col = st.columns(2)
    with feature_col:
        st.markdown("#### Graph features")
        st.json(result["features"], expanded=True)
    with eligibility_col:
        st.markdown("#### Eligibility and runtime predictions")
        status_rows = []
        for name in sorted(result["choices"]):
            status_rows.append({
                "Algorithm": ALGORITHM_LABELS[name],
                "Status": result["choices"][name],
                "Predicted runtime": (
                    _format_seconds(result["predictions"][name])
                    if name in result["eligible"] and name in result["predictions"]
                    else "—"
                ),
            })
        st.dataframe(status_rows, hide_index=True, width="stretch")

    st.markdown("#### Path and measured comparison")
    chosen_path = result["model_result"]["path"]
    if chosen_path is None:
        st.warning("No path exists between these selected endpoints.")
    else:
        st.code(" → ".join(chosen_path), language=None)

    timing_rows = []
    for name in result["eligible"]:
        timing_rows.append({
            "Eligible algorithm": ALGORITHM_LABELS[name],
            "Median execution": _format_seconds(result["algorithm_timings"][name]),
            "Model predicted": _format_seconds(result["predictions"][name])
            if name in result["predictions"] else "—",
        })
    timing_rows.extend([
        {
            "Eligible algorithm": "Model selector end to end",
            "Median execution": _format_seconds(result["model_end_to_end_seconds"]),
            "Model predicted": "—",
        },
        {
            "Eligible algorithm": "Rule selector end to end",
            "Median execution": _format_seconds(result["rule_end_to_end_seconds"]),
            "Model predicted": "—",
        },
    ])
    st.dataframe(timing_rows, hide_index=True, width="stretch")
    st.caption(
        f"The observed fastest eligible algorithm was "
        f"{ALGORITHM_LABELS[result['oracle']]} after all eligible algorithms "
        "had been benchmarked. The selector does not know this oracle choice "
        "before execution. Timings are local and can vary between runs."
    )
    st.markdown("#### Selection overhead")
    st.write(
        f"Model decision: {_format_seconds(result['model_decision_seconds'])} · "
        f"Rule decision: {_format_seconds(result['rule_decision_seconds'])}"
    )
    st.caption(
        "The shown feature table is extracted before the timing run, so the "
        "heuristic-scale cache is warm for these comparison timings."
    )


if __name__ == "__main__":
    main()
