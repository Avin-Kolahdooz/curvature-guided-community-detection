
"""
Shared utilities for curvature-guided community detection experiments.

Common functions used by Amazon, DBLP, Facebook, and YouTube notebooks.

Dataset-specific constants, dataset selection, plots, and final tables
should remain in notebooks. Experiment parameters are passed explicitly.
"""

from __future__ import annotations

import gzip
import os
import time
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import (
    Any, Dict, Hashable, Iterable, Iterator, List, Mapping,
    Optional, Sequence, Set, Tuple, Union
)

import igraph as ig
import leidenalg
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from GraphRicciCurvature.OllivierRicci import OllivierRicci
from scipy.stats import ttest_rel, wilcoxon
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
)
import community.community_louvain as community_louvain


Node = Hashable
Partition = Mapping[Node, int]
PathLike = Union[str, os.PathLike]


# ============================================================
# 1. FILE AND DATA LOADING
# ============================================================

def download_if_missing(url: str, path: PathLike) -> None:
    """Download a file only if it does not already exist."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        print("Already exists:", path)
        return

    print("Downloading:", url)
    urllib.request.urlretrieve(url, path)
    print("Saved:", path)


def stream_edges(
    edges_file: PathLike,
    chunksize: int = 1_000_000,
) -> Iterator[pd.DataFrame]:
    """Read SNAP edges in chunks."""
    for chunk in pd.read_csv(
        edges_file,
        sep="\t",
        comment="#",
        header=None,
        names=["u", "v"],
        dtype={"u": "int64", "v": "int64"},
        chunksize=chunksize,
    ):
        yield chunk


def load_communities(
    community_file: PathLike,
    min_size: int = 1,
) -> List[List[int]]:
    """Load SNAP community membership data."""
    communities = []

    with gzip.open(
        community_file,
        "rt",
        encoding="utf-8",
        errors="replace",
    ) as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            nodes = [
                int(value)
                for value in line.split("\t")
                if value != ""
            ]

            if len(nodes) >= min_size:
                communities.append(nodes)

    return communities


def load_facebook_combined(path: PathLike) -> nx.Graph:
    """Load Facebook combined graph."""
    graph = nx.Graph()

    with gzip.open(
        path, "rt", encoding="utf-8", errors="replace"
    ) as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            u, v = map(int, line.split())
            graph.add_edge(u, v)

    graph.remove_edges_from(nx.selfloop_edges(graph))
    return graph


def load_all_facebook_circles(
    extract_dir: PathLike,
    min_size: int = 1,
) -> List[Dict[str, Any]]:
    """Load Facebook circle files."""
    communities = []
    circle_files = sorted(Path(extract_dir).glob("*.circles"))

    print("Circle files:", len(circle_files))

    for file in circle_files:
        ego_id = int(file.stem)

        with open(
            file, "r", encoding="utf-8", errors="replace"
        ) as handle:
            for line in handle:
                parts = line.strip().split()

                if len(parts) <= 1:
                    continue

                circle_name = parts[0]
                nodes = [int(value) for value in parts[1:]]

                if len(nodes) >= min_size:
                    communities.append({
                        "ego_id": ego_id,
                        "circle_name": circle_name,
                        "nodes": nodes,
                        "size": len(nodes),
                    })

    return communities


# ============================================================
# 2. GRAPH PREPROCESSING
# ============================================================

def get_gcc(graph: nx.Graph) -> nx.Graph:
    """Return giant connected component."""
    if (
        graph.number_of_nodes() == 0
        or graph.number_of_edges() == 0
    ):
        return graph.copy()

    if nx.is_connected(graph):
        return graph.copy()

    gcc_nodes = max(nx.connected_components(graph), key=len)
    return graph.subgraph(gcc_nodes).copy()


def ensure_edge_weights(
    graph: nx.Graph,
    weight: str = "weight",
    default_weight: float = 1.0,
    copy_graph: bool = True,
) -> nx.Graph:
    """Assign default weight to edges lacking one."""
    output = graph.copy() if copy_graph else graph

    for u, v in output.edges():
        output[u][v][weight] = (
            output[u][v].get(weight, default_weight)
        )

    return output


# ============================================================
# 3. COMMUNITY DETECTION
# ============================================================

def run_louvain(
    graph: nx.Graph,
    weight: str = "weight",
    seed: int = 42,
) -> Dict[Node, int]:

    if graph.number_of_nodes() == 0:
        return {}

    graph_run = ensure_edge_weights(
        graph, weight=weight,
        default_weight=1.0, copy_graph=True
    )

    return community_louvain.best_partition(
        graph_run,
        weight=weight,
        random_state=seed,
    )


def nx_to_igraph(
    graph: nx.Graph,
    weight: str = "weight",
) -> Tuple[ig.Graph, List[Node]]:

    nodes = list(graph.nodes())
    node_to_idx = {
        node: index for index, node in enumerate(nodes)
    }

    edges = [
        (node_to_idx[u], node_to_idx[v])
        for u, v in graph.edges()
    ]

    weights = [
        graph[u][v].get(weight, 1.0)
        for u, v in graph.edges()
    ]

    ig_graph = ig.Graph()
    ig_graph.add_vertices(len(nodes))
    ig_graph.add_edges(edges)

    ig_graph.vs["name"] = nodes
    ig_graph.es["weight"] = weights

    return ig_graph, nodes


def run_leiden(
    graph: nx.Graph,
    weight: str = "weight",
    seed: int = 42,
) -> Dict[Node, int]:

    if graph.number_of_nodes() == 0:
        return {}

    graph_run = ensure_edge_weights(
        graph, weight=weight,
        default_weight=1.0, copy_graph=True
    )

    ig_graph, nodes = nx_to_igraph(
        graph_run, weight=weight
    )

    partition = leidenalg.find_partition(
        ig_graph,
        leidenalg.ModularityVertexPartition,
        weights=ig_graph.es["weight"],
        seed=seed,
    )

    result = {}

    for community_id, community_nodes in enumerate(partition):
        for index in community_nodes:
            result[nodes[index]] = community_id

    return result


# ============================================================
# 4. CLUSTERING EVALUATION
# ============================================================

def eval_partition(
    graph: nx.Graph,
    partition: Partition,
    label_attribute: str = "label_gt",
) -> Tuple[float, float, int]:

    nodes = [
        node for node in graph.nodes()
        if (
            label_attribute in graph.nodes[node]
            and node in partition
        )
    ]

    if len(nodes) == 0:
        return np.nan, np.nan, 0

    y_true = [
        graph.nodes[node][label_attribute]
        for node in nodes
    ]
    y_pred = [partition[node] for node in nodes]

    if len(set(y_true)) < 2:
        return 0.0, 0.0, len(nodes)

    return (
        adjusted_rand_score(y_true, y_pred),
        normalized_mutual_info_score(y_true, y_pred),
        len(nodes),
    )


# ============================================================
# 5. LOWER RICCI CURVATURE AND EDGE PRUNING
# ============================================================

def compute_LRC(
    graph: nx.Graph,
    attribute: str = "LRC",
) -> nx.Graph:
    """Compute Lower Ricci Curvature for every edge."""

    for u, v in graph.edges():
        neighbors_u = set(graph.neighbors(u))
        neighbors_v = set(graph.neighbors(v))

        degree_u = len(neighbors_u)
        degree_v = len(neighbors_v)
        common_neighbors = len(neighbors_u & neighbors_v)

        if degree_u == 0 or degree_v == 0:
            lrc = -np.inf
        else:
            lrc = (
                2.0 / degree_u
                + 2.0 / degree_v
                - 2.0
                + 2.0 * common_neighbors / max(
                    degree_u, degree_v
                )
                + common_neighbors / min(
                    degree_u, degree_v
                )
            )

        graph[u][v][attribute] = float(lrc)

    return graph


def prune_by_percentile(
    graph: nx.Graph,
    attr: str,
    pct: float,
    remove_equal_to_cutoff: bool = True,
) -> Tuple[nx.Graph, Optional[float], int]:
    """Prune edges based on an attribute percentile."""

    if not 0 <= pct <= 100:
        raise ValueError("pct must be between 0 and 100.")

    if pct == 0:
        return graph.copy(), None, 0

    values = np.array(
        [
            data.get(attr, 0.0)
            for _, _, data in graph.edges(data=True)
        ],
        dtype=float,
    )

    values = values[np.isfinite(values)]

    if len(values) == 0:
        return graph.copy(), np.nan, 0

    cutoff = float(np.percentile(values, pct))
    output = graph.copy()

    if remove_equal_to_cutoff:
        to_drop = [
            (u, v)
            for u, v, data in output.edges(data=True)
            if data.get(attr, 0.0) <= cutoff
        ]
    else:
        to_drop = [
            (u, v)
            for u, v, data in output.edges(data=True)
            if data.get(attr, 0.0) < cutoff
        ]

    output.remove_edges_from(to_drop)
    output.remove_nodes_from(list(nx.isolates(output)))

    return output, cutoff, len(to_drop)


# ============================================================
# 6. OLLIVIER-RICCI CURVATURE AND RICCI FLOW
# ============================================================

def run_orc_flow_gcc(
    graph: nx.Graph,
    alpha: float = 0.5,
    iterations: int = 5,
    method: str = "Sinkhorn",
    weight: str = "weight",
    verbose: str = "ERROR",
    proc: Optional[int] = None,
) -> nx.Graph:
    """Run Ollivier-Ricci flow on the supplied graph."""

    graph_orc = ensure_edge_weights(
        graph,
        weight=weight,
        default_weight=1.0,
        copy_graph=True,
    )

    kwargs = {
        "alpha": alpha,
        "method": method,
        "weight": weight,
        "verbose": verbose,
    }

    if proc is not None:
        if proc < 1:
            raise ValueError("proc must be >= 1")
        kwargs["proc"] = proc

    orc = OllivierRicci(
        graph_orc,
        **kwargs,
    )

    orc.compute_ricci_flow(
        iterations=iterations
    )

    return orc.G.copy()


def run_orc_no_flow(
    graph: nx.Graph,
    alpha: float = 0.5,
    method: str = "Sinkhorn",
    weight: str = "weight",
    verbose: str = "ERROR",
    proc: Optional[int] = None,
) -> nx.Graph:
    """Compute ORC without Ricci flow."""

    graph_orc = ensure_edge_weights(
        graph,
        weight=weight,
        default_weight=1.0,
        copy_graph=True,
    )

    kwargs = {
        "alpha": alpha,
        "method": method,
        "weight": weight,
        "verbose": verbose,
    }

    if proc is not None:
        if proc < 1:
            raise ValueError("proc must be >= 1")
        kwargs["proc"] = proc

    orc = OllivierRicci(graph_orc, **kwargs)
    orc.compute_ricci_curvature()

    return orc.G.copy()


# ============================================================
# 7. PREPARE EXPERIMENTAL GRAPHS
# ============================================================

def prepare_lrc_graph(
    graph: nx.Graph,
    lrc_pct: float,
    lrc_attribute: str = "LRC",
    print_distribution: bool = False,
) -> Dict[str, Any]:

    start = time.perf_counter()

    graph_lrc = graph.copy()
    compute_LRC(graph_lrc, attribute=lrc_attribute)

    lrc_values = [
        data[lrc_attribute]
        for _, _, data in graph_lrc.edges(data=True)
        if np.isfinite(data[lrc_attribute])
    ]

    if print_distribution:
        print("LRC distribution:")
        print(pd.Series(lrc_values).describe())

    pruned_graph, cutoff, removed_lrc = prune_by_percentile(
        graph_lrc,
        attr=lrc_attribute,
        pct=lrc_pct,
    )

    final_graph = pruned_graph.copy()
    prep_time = time.perf_counter() - start

    return {
        "method_base": "LRC-only",
        "graph": final_graph,
        "prep_time": prep_time,
        "removed_lrc": removed_lrc,
        "removed_orc": 0,
        "pct_lrc": lrc_pct,
        "pct_orc": np.nan,
        "alpha": np.nan,
        "iterations": np.nan,
        "cutoff_lrc": cutoff,
        "cutoff_orc": np.nan,
    }


def prepare_orc_graph(
    graph: nx.Graph,
    orc_pct: float,
    alpha: float = 0.5,
    iterations: int = 5,
    method: str = "Sinkhorn",
    weight: str = "weight",
    orc_attribute: str = "ricciCurvature",
    proc: Optional[int] = None,
    verbose: str = "ERROR",
) -> Dict[str, Any]:
    """Run ORC flow and prune low-curvature edges."""

    start = time.perf_counter()

    flow_graph = run_orc_flow_gcc(
        graph,
        alpha=alpha,
        iterations=iterations,
        method=method,
        weight=weight,
        proc=proc,
        verbose=verbose,
    )

    pruned_graph, cutoff_orc, removed_orc = prune_by_percentile(
        flow_graph,
        attr=orc_attribute,
        pct=orc_pct,
    )

    final_graph = pruned_graph.copy()
    prep_time = time.perf_counter() - start

    return {
        "method_base": "ORC-only",
        "graph": final_graph,
        "prep_time": prep_time,
        "removed_lrc": 0,
        "removed_orc": removed_orc,
        "pct_lrc": np.nan,
        "pct_orc": orc_pct,
        "alpha": alpha,
        "iterations": iterations,
        "cutoff_lrc": np.nan,
        "cutoff_orc": cutoff_orc,
    }


def prepare_combo_graph(
    graph: nx.Graph,
    lrc_pct: float,
    orc_pct: float,
    alpha: float = 0.5,
    iterations: int = 5,
    method: str = "Sinkhorn",
    weight: str = "weight",
    lrc_attribute: str = "LRC",
    orc_attribute: str = "ricciCurvature",
    use_ricci_flow: bool = True,
    proc: Optional[int] = None,
    verbose: str = "ERROR",
) -> Dict[str, Any]:
    """
    LRC pruning on full graph.
    ORC is applied only to the largest connected component.
    Non-GCC components are preserved and reattached.
    """

    start = time.perf_counter()

    method_base = (
        "LRC→ORC combo"
        if use_ricci_flow
        else "LRC→ORC combo (No Flow)"
    )

    graph_lrc = graph.copy()
    compute_LRC(
        graph_lrc,
        attribute=lrc_attribute,
    )

    after_lrc, cutoff_lrc, removed_lrc = prune_by_percentile(
        graph_lrc,
        attr=lrc_attribute,
        pct=lrc_pct,
    )

    if after_lrc.number_of_nodes() == 0:
        prep_time = time.perf_counter() - start
        return {
            "method_base": method_base,
            "graph": after_lrc.copy(),
            "prep_time": prep_time,
            "removed_lrc": removed_lrc,
            "removed_orc": 0,
            "pct_lrc": lrc_pct,
            "pct_orc": orc_pct,
            "alpha": alpha,
            "iterations": iterations,
            "cutoff_lrc": cutoff_lrc,
            "cutoff_orc": np.nan,
        }

    components = sorted(
        nx.connected_components(after_lrc),
        key=len,
        reverse=True,
    )

    gcc_nodes = components[0]

    gcc_graph = after_lrc.subgraph(
        gcc_nodes
    ).copy()

    non_gcc_nodes = (
        set(after_lrc.nodes()) - set(gcc_nodes)
    )

    non_gcc_graph = after_lrc.subgraph(
        non_gcc_nodes
    ).copy()

    if use_ricci_flow:
        flow_graph = run_orc_flow_gcc(
            gcc_graph,
            alpha=alpha,
            iterations=iterations,
            method=method,
            weight=weight,
            proc=proc,
            verbose=verbose,
        )
    else:
        flow_graph = run_orc_no_flow(
            gcc_graph,
            alpha=alpha,
            method=method,
            weight=weight,
            proc=proc,
            verbose=verbose,
        )

    processed_gcc, cutoff_orc, removed_orc = prune_by_percentile(
        flow_graph,
        attr=orc_attribute,
        pct=orc_pct,
    )

    final_graph = nx.compose(
        processed_gcc,
        non_gcc_graph,
    )

    prep_time = time.perf_counter() - start

    return {
        "method_base": method_base,
        "graph": final_graph,
        "prep_time": prep_time,
        "removed_lrc": removed_lrc,
        "removed_orc": removed_orc,
        "pct_lrc": lrc_pct,
        "pct_orc": orc_pct,
        "alpha": alpha,
        "iterations": iterations,
        "cutoff_lrc": cutoff_lrc,
        "cutoff_orc": cutoff_orc,
    }


# ============================================================
# 8. EVALUATION
# ============================================================

def evaluate_prepared_graph(
    info: Mapping[str, Any],
    algorithm: str = "louvain",
    seed: int = 42,
    weight: str = "weight",
    label_attribute: str = "label_gt",
) -> Dict[str, Any]:

    graph = info["graph"]
    start = time.perf_counter()

    algorithm_lower = algorithm.lower()

    if algorithm_lower == "louvain":
        partition = run_louvain(
            graph, weight=weight, seed=seed
        )
        method_name = info["method_base"]

    elif algorithm_lower == "leiden":
        partition = run_leiden(
            graph, weight=weight, seed=seed
        )
        method_name = info["method_base"] + " + Leiden"

    else:
        raise ValueError(
            "algorithm must be 'louvain' or 'leiden'."
        )

    community_time = time.perf_counter() - start

    ari, nmi, n_eval = eval_partition(
        graph, partition,
        label_attribute=label_attribute,
    )

    return {
        "method": method_name,
        "algorithm": algorithm_lower,
        "seed": seed,
        "ARI": ari,
        "NMI": nmi,
        "n_eval": n_eval,
        "V": graph.number_of_nodes(),
        "E": graph.number_of_edges(),
        "components": (
            nx.number_connected_components(graph)
            if graph.number_of_nodes() > 0 else 0
        ),
        "runtime_sec": info["prep_time"],
        "prep_time": info["prep_time"],
        "community_time": community_time,
        "removed_lrc": info["removed_lrc"],
        "removed_orc": info["removed_orc"],
        "pct_lrc": info["pct_lrc"],
        "pct_orc": info["pct_orc"],
        "alpha": info["alpha"],
        "iterations": info["iterations"],
        "cutoff_lrc": info["cutoff_lrc"],
        "cutoff_orc": info["cutoff_orc"],
    }


def evaluate_plain_graph(
    graph: nx.Graph,
    algorithm: str = "louvain",
    seed: int = 42,
    weight: str = "weight",
    label_attribute: str = "label_gt",
    method_name: Optional[str] = None,
) -> Dict[str, Any]:

    start = time.perf_counter()
    algorithm_lower = algorithm.lower()

    if algorithm_lower == "louvain":
        partition = run_louvain(
            graph, weight=weight, seed=seed
        )
        default_name = "Plain Louvain"

    elif algorithm_lower == "leiden":
        partition = run_leiden(
            graph, weight=weight, seed=seed
        )
        default_name = "Plain Leiden"

    else:
        raise ValueError(
            "algorithm must be 'louvain' or 'leiden'."
        )

    runtime = time.perf_counter() - start

    ari, nmi, n_eval = eval_partition(
        graph, partition,
        label_attribute=label_attribute,
    )

    return {
        "method": method_name or default_name,
        "algorithm": algorithm_lower,
        "seed": seed,
        "ARI": ari,
        "NMI": nmi,
        "n_eval": n_eval,
        "V": graph.number_of_nodes(),
        "E": graph.number_of_edges(),
        "components": (
            nx.number_connected_components(graph)
            if graph.number_of_nodes() > 0 else 0
        ),
        "runtime_sec": runtime,
        "prep_time": 0.0,
        "community_time": runtime,
        "removed_lrc": 0,
        "removed_orc": 0,
        "pct_lrc": np.nan,
        "pct_orc": np.nan,
        "alpha": np.nan,
        "iterations": np.nan,
        "cutoff_lrc": np.nan,
        "cutoff_orc": np.nan,
    }


# ============================================================
# 9. SUMMARIES AND STATISTICAL TESTS
# ============================================================

def mean_pm_std(
    series: Union[pd.Series, Sequence[float], np.ndarray],
    digits: int = 3,
) -> str:

    values = pd.Series(series, dtype=float)
    mean = values.mean()
    std = values.std(ddof=1)

    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def summarize_results(
    results: pd.DataFrame,
    method_column: str = "method",
) -> pd.DataFrame:

    required = {
        method_column, "ARI", "NMI",
        "runtime_sec", "n_eval", "V", "E",
    }

    missing = required.difference(results.columns)

    if missing:
        raise KeyError(
            f"Missing required columns: {sorted(missing)}"
        )

    summary = (
        results.groupby(method_column)
        .agg(
            ARI_mean=("ARI", "mean"),
            ARI_std=("ARI", "std"),
            NMI_mean=("NMI", "mean"),
            NMI_std=("NMI", "std"),
            Runtime_mean=("runtime_sec", "mean"),
            Runtime_std=("runtime_sec", "std"),
            n_eval_mean=("n_eval", "mean"),
            V_mean=("V", "mean"),
            E_mean=("E", "mean"),
        )
        .reset_index()
    )

    summary["ARI"] = (
        summary["ARI_mean"].map(
            lambda value: f"{value:.3f}"
        )
        + " ± "
        + summary["ARI_std"].map(
            lambda value: f"{value:.3f}"
        )
    )

    summary["NMI"] = (
        summary["NMI_mean"].map(
            lambda value: f"{value:.3f}"
        )
        + " ± "
        + summary["NMI_std"].map(
            lambda value: f"{value:.3f}"
        )
    )

    summary["Runtime"] = (
        summary["Runtime_mean"].map(
            lambda value: f"{value:.2f}"
        )
        + " ± "
        + summary["Runtime_std"].map(
            lambda value: f"{value:.2f}"
        )
    )

    final_table = summary[
        [
            method_column,
            "ARI", "NMI", "Runtime",
            "n_eval_mean", "V_mean", "E_mean",
        ]
    ].copy()

    return final_table


def paired_tests(
    results: pd.DataFrame,
    baseline_method: str,
    metric: str = "ARI",
) -> pd.DataFrame:

    required = {"method", "seed", metric}
    missing = required.difference(results.columns)

    if missing:
        raise KeyError(
            f"Missing required columns: {sorted(missing)}"
        )

    baseline_df = (
        results[
            results["method"] == baseline_method
        ][["seed", metric]]
        .dropna()
        .rename(columns={metric: "baseline_value"})
    )

    rows = []

    for method in sorted(
        results["method"].dropna().unique()
    ):
        if method == baseline_method:
            continue

        method_df = (
            results[
                results["method"] == method
            ][["seed", metric]]
            .dropna()
            .rename(columns={metric: "method_value"})
        )

        paired = baseline_df.merge(
            method_df, on="seed", how="inner"
        )

        if paired.empty:
            continue

        base = paired["baseline_value"].to_numpy(
            dtype=float
        )
        values = paired["method_value"].to_numpy(
            dtype=float
        )

        try:
            t_p = ttest_rel(values, base).pvalue
        except Exception:
            t_p = np.nan

        try:
            w_p = wilcoxon(values, base).pvalue
        except Exception:
            w_p = np.nan

        rows.append({
            "baseline": baseline_method,
            "method": method,
            "metric": metric,
            "n_pairs": len(paired),
            "baseline_mean": np.mean(base),
            "method_mean": np.mean(values),
            "mean_diff": np.mean(values - base),
            "paired_t_p": t_p,
            "wilcoxon_p": w_p,
        })

    return pd.DataFrame(rows)


# ============================================================
# 10. GRAPH AND LABEL STRUCTURAL ANALYSIS
# ============================================================

def basic_graph_stats(
    graph: nx.Graph,
    dataset_name: Optional[str] = None,
    include_path_metrics: bool = False,
) -> Dict[str, Any]:

    n = graph.number_of_nodes()
    m = graph.number_of_edges()

    degrees = np.array(
        [degree for _, degree in graph.degree()],
        dtype=float,
    )

    gcc = get_gcc(graph)

    stats = {
        "nodes": n,
        "edges": m,
        "density": nx.density(graph) if n > 0 else np.nan,
        "avg_degree": (
            degrees.mean() if len(degrees) else np.nan
        ),
        "min_degree": (
            degrees.min() if len(degrees) else np.nan
        ),
        "max_degree": (
            degrees.max() if len(degrees) else np.nan
        ),
        "avg_clustering": (
            nx.average_clustering(graph)
            if n > 0 else np.nan
        ),
        "components": (
            nx.number_connected_components(graph)
            if n > 0 else 0
        ),
        "gcc_nodes": gcc.number_of_nodes(),
        "gcc_edges": gcc.number_of_edges(),
    }

    if dataset_name is not None:
        stats = {"dataset": dataset_name, **stats}

    if include_path_metrics:
        if gcc.number_of_nodes() > 0:
            stats["avg_shortest_path"] = (
                nx.average_shortest_path_length(gcc)
            )
            stats["diameter"] = nx.diameter(gcc)
        else:
            stats["avg_shortest_path"] = np.nan
            stats["diameter"] = np.nan

    return stats


def label_structure_stats(
    graph: nx.Graph,
    label_attribute: str = "label_gt",
    overlap_attribute: str = "is_overlap",
) -> Tuple[Dict[str, Any], pd.Series]:

    labeled_nodes = [
        node for node in graph.nodes()
        if label_attribute in graph.nodes[node]
    ]

    labels = [
        graph.nodes[node][label_attribute]
        for node in labeled_nodes
    ]

    label_counts = pd.Series(labels).value_counts()

    overlap_flags = [
        graph.nodes[node].get(overlap_attribute, 0)
        for node in labeled_nodes
    ]

    if label_counts.empty:
        return {
            "labeled_nodes": 0,
            "num_labels": 0,
            "largest_label_size": np.nan,
            "smallest_label_size": np.nan,
            "mean_label_size": np.nan,
            "median_label_size": np.nan,
            "overlap_fraction": np.nan,
        }, label_counts

    return {
        "labeled_nodes": len(labeled_nodes),
        "num_labels": len(set(labels)),
        "largest_label_size": label_counts.max(),
        "smallest_label_size": label_counts.min(),
        "mean_label_size": label_counts.mean(),
        "median_label_size": label_counts.median(),
        "overlap_fraction": (
            np.mean(overlap_flags)
            if overlap_flags else np.nan
        ),
    }, label_counts



def compute_community_metrics(
    graph: nx.Graph,
    nodes: Iterable[Node],
    *,
    total_edges: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Undirected community metrics with correct cut-edge counts."""

    community = {
        node for node in nodes if node in graph
    }

    if not community:
        return None

    internal_edges = 0
    boundary_edges = 0
    internal_triangles = 0
    boundary_triangles = 0
    counted_internal = set()

    for u in community:
        neighbors_u = set(graph.neighbors(u))

        for v in neighbors_u:

            if v in community:
                edge_key = frozenset((u, v))

                if edge_key in counted_internal:
                    continue

                counted_internal.add(edge_key)
                internal_edges += 1

                internal_triangles += len(
                    neighbors_u & set(graph.neighbors(v))
                )

            else:
                boundary_edges += 1

                boundary_triangles += len(
                    neighbors_u & set(graph.neighbors(v))
                )

    volume_s = 2 * internal_edges + boundary_edges

    m = (
        graph.number_of_edges()
        if total_edges is None
        else total_edges
    )

    volume_complement = 2 * m - volume_s

    denominator = min(
        volume_s,
        volume_complement
    )

    conductance = (
        boundary_edges / denominator
        if denominator > 0
        else np.nan
    )

    btd = (
        boundary_triangles / boundary_edges
        if boundary_edges > 0
        else np.nan
    )

    return {
        "size": len(community),
        "internal_edges": internal_edges,
        "boundary_edges": boundary_edges,
        "boundary_triangles": boundary_triangles,
        "internal_triangles": internal_triangles,
        "conductance": conductance,
        "BTD": btd,
    }


def compute_label_metrics(
    graph: nx.Graph,
    label: Any,
    label_attribute: str = "label_gt",
) -> Optional[Dict[str, Any]]:

    community = {
        node for node in graph.nodes()
        if graph.nodes[node].get(label_attribute) == label
    }

    if len(community) == 0:
        return None

    result = compute_community_metrics(
        graph, community
    )

    if result is None:
        return None

    possible_internal_edges = (
        len(community) * (len(community) - 1) / 2
    )

    internal_density = (
        result["internal_edges"] / possible_internal_edges
        if possible_internal_edges > 0 else np.nan
    )

    return {
        "label": label,
        **result,
        "internal_density": internal_density,
    }


def weighted_mean(
    dataframe: pd.DataFrame,
    value_col: str,
    weight_col: str = "size",
) -> float:

    valid = dataframe[
        np.isfinite(dataframe[value_col])
        &
        np.isfinite(dataframe[weight_col])
    ]

    if len(valid) == 0:
        return np.nan

    return float(
        np.average(
            valid[value_col],
            weights=valid[weight_col],
        )
    )


def ground_truth_partition(
    graph: nx.Graph,
    label_attribute: str = "label_gt",
) -> List[Set[Node]]:

    communities_by_label = defaultdict(set)

    for node in graph.nodes():
        if label_attribute in graph.nodes[node]:
            label = graph.nodes[node][label_attribute]
            communities_by_label[label].add(node)

    return list(communities_by_label.values())


# ============================================================
# 11. PARAMETER SENSITIVITY ANALYSIS
# ============================================================

def run_combo_sensitivity(
    graph: nx.Graph,
    lrc_pcts: Sequence[float],
    orc_pcts: Sequence[float],
    alphas: Sequence[float] = (0.5,),
    iterations: int = 5,
    seeds: Sequence[int] = (0, 1, 2, 3, 4, 42),
    algorithm: str = "louvain",
    method: str = "Sinkhorn",
    weight: str = "weight",
    label_attribute: str = "label_gt",
    verbose: bool = True,
) -> pd.DataFrame:

    rows = []

    total_configs = (
        len(lrc_pcts)
        * len(orc_pcts)
        * len(alphas)
    )

    config_index = 0

    for alpha in alphas:
        for lrc_pct in lrc_pcts:
            for orc_pct in orc_pcts:
                config_index += 1

                if verbose:
                    print(
                        f"[{config_index}/{total_configs}] "
                        f"LRC={lrc_pct}%, "
                        f"ORC={orc_pct}%, alpha={alpha}"
                    )

                try:
                    info = prepare_combo_graph(
                        graph=graph,
                        lrc_pct=lrc_pct,
                        orc_pct=orc_pct,
                        alpha=alpha,
                        iterations=iterations,
                        method=method,
                        weight=weight,
                    )
                except Exception as error:
                    if verbose:
                        print(
                            "Preparation failed:",
                            repr(error)
                        )

                    for seed in seeds:
                        rows.append({
                            "method": "LRC→ORC combo",
                            "algorithm": algorithm,
                            "seed": seed,
                            "lrc_pct": lrc_pct,
                            "orc_pct": orc_pct,
                            "alpha": alpha,
                            "iterations": iterations,
                            "ARI": np.nan,
                            "NMI": np.nan,
                            "runtime_sec": np.nan,
                            "prep_time": np.nan,
                            "community_time": np.nan,
                            "n_eval": 0,
                            "V": 0,
                            "E": 0,
                            "removed_lrc": np.nan,
                            "removed_orc": np.nan,
                            "status": "failed",
                            "error": repr(error),
                        })
                    continue

                for seed in seeds:
                    try:
                        result = evaluate_prepared_graph(
                            info=info,
                            algorithm=algorithm,
                            seed=seed,
                            weight=weight,
                            label_attribute=label_attribute,
                        )

                        result.update({
                            "lrc_pct": lrc_pct,
                            "orc_pct": orc_pct,
                            "alpha": alpha,
                            "iterations": iterations,
                            "status": "ok",
                            "error": "",
                        })

                        rows.append(result)

                    except Exception as error:
                        rows.append({
                            "method": "LRC→ORC combo",
                            "algorithm": algorithm,
                            "seed": seed,
                            "lrc_pct": lrc_pct,
                            "orc_pct": orc_pct,
                            "alpha": alpha,
                            "iterations": iterations,
                            "ARI": np.nan,
                            "NMI": np.nan,
                            "runtime_sec": np.nan,
                            "prep_time": info.get(
                                "prep_time", np.nan
                            ),
                            "community_time": np.nan,
                            "n_eval": 0,
                            "V": info["graph"].number_of_nodes(),
                            "E": info["graph"].number_of_edges(),
                            "removed_lrc": info.get(
                                "removed_lrc", np.nan
                            ),
                            "removed_orc": info.get(
                                "removed_orc", np.nan
                            ),
                            "status": "failed",
                            "error": repr(error),
                        })

    return pd.DataFrame(rows)


def summarize_sensitivity(
    results: pd.DataFrame,
    group_columns: Sequence[str] = (
        "lrc_pct",
        "orc_pct",
        "alpha",
        "iterations",
    ),
) -> pd.DataFrame:

    required = {
        *group_columns,
        "ARI", "NMI",
        "runtime_sec", "V", "E",
    }

    missing = required.difference(results.columns)

    if missing:
        raise KeyError(
            f"Missing required columns: {sorted(missing)}"
        )

    valid = results.copy()

    if "status" in valid.columns:
        valid = valid[
            valid["status"] == "ok"
        ].copy()

    summary = (
        valid.groupby(
            list(group_columns),
            dropna=False,
            as_index=False,
        )
        .agg(
            ARI_mean=("ARI", "mean"),
            ARI_std=("ARI", "std"),
            NMI_mean=("NMI", "mean"),
            NMI_std=("NMI", "std"),
            Runtime_mean=("runtime_sec", "mean"),
            Runtime_std=("runtime_sec", "std"),
            V_mean=("V", "mean"),
            E_mean=("E", "mean"),
            n_runs=("seed", "nunique"),
        )
    )

    summary["ARI"] = (
        summary["ARI_mean"].map(
            lambda value: f"{value:.3f}"
        )
        + " ± "
        + summary["ARI_std"].map(
            lambda value: f"{value:.3f}"
        )
    )

    summary["NMI"] = (
        summary["NMI_mean"].map(
            lambda value: f"{value:.3f}"
        )
        + " ± "
        + summary["NMI_std"].map(
            lambda value: f"{value:.3f}"
        )
    )

    summary["Runtime"] = (
        summary["Runtime_mean"].map(
            lambda value: f"{value:.2f}"
        )
        + " ± "
        + summary["Runtime_std"].map(
            lambda value: f"{value:.2f}"
        )
    )

    return summary.sort_values(
        ["alpha", "lrc_pct", "orc_pct"]
    ).reset_index(drop=True)


def select_best_sensitivity_configs(
    summary: pd.DataFrame,
    top_n: int = 10,
    metric: str = "ARI_mean",
    ascending: bool = False,
) -> pd.DataFrame:

    if metric not in summary.columns:
        raise KeyError(
            f"{metric!r} is not in the summary table."
        )

    return (
        summary.sort_values(
            metric, ascending=ascending
        )
        .head(top_n)
        .reset_index(drop=True)
    )


def plot_sensitivity_metric(
    summary: pd.DataFrame,
    x: str,
    y: str,
    group: Optional[str] = None,
    error: Optional[str] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    figsize: Tuple[float, float] = (7, 5),
):

    required = {x, y}

    if group is not None:
        required.add(group)
    if error is not None:
        required.add(error)

    missing = required.difference(summary.columns)

    if missing:
        raise KeyError(
            f"Missing columns: {sorted(missing)}"
        )

    fig, ax = plt.subplots(figsize=figsize)

    if group is None:
        plot_df = summary.sort_values(x)
        ax.errorbar(
            plot_df[x],
            plot_df[y],
            yerr=(
                plot_df[error]
                if error is not None else None
            ),
            marker="o",
            capsize=3,
        )
    else:
        for group_value, plot_df in summary.groupby(
            group, dropna=False
        ):
            plot_df = plot_df.sort_values(x)
            ax.errorbar(
                plot_df[x],
                plot_df[y],
                yerr=(
                    plot_df[error]
                    if error is not None else None
                ),
                marker="o",
                capsize=3,
                label=f"{group}={group_value}",
            )
        ax.legend()

    ax.set_title(title or f"{y} versus {x}")
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    return fig, ax


# ============================================================
# 12. REAL-DATASET SENSITIVITY HEATMAPS
# ============================================================

LRC_HEAT_VALUES = [0, 5, 10, 20, 30]
ORC_HEAT_VALUES = [0, 5, 10, 20, 30]

HEATMAP_ALPHA = 0.5
LOUVAIN_SEEDS = [0, 1]


def estimate_empirical_pin_pout(
    graph,
    label_attribute="label_gt",
):

    labeled_nodes = [
        node for node in graph.nodes()
        if label_attribute in graph.nodes[node]
    ]

    labels = {
        node: graph.nodes[node][label_attribute]
        for node in labeled_nodes
    }

    label_counts = (
        pd.Series(list(labels.values()))
        .value_counts()
        .to_dict()
    )

    possible_within = sum(
        n * (n - 1) / 2
        for n in label_counts.values()
    )

    n = len(labeled_nodes)
    possible_total = n * (n - 1) / 2
    possible_between = possible_total - possible_within

    observed_within = 0
    observed_between = 0

    labeled_set = set(labeled_nodes)

    for u, v in graph.edges():
        if (
            u not in labeled_set
            or v not in labeled_set
        ):
            continue

        if labels[u] == labels[v]:
            observed_within += 1
        else:
            observed_between += 1

    p_in = (
        observed_within / possible_within
        if possible_within > 0 else np.nan
    )

    p_out = (
        observed_between / possible_between
        if possible_between > 0 else np.nan
    )

    return p_in, p_out


def run_real_dataset_heatmap(
    graph,
    dataset_name,
    label_attribute="label_gt",
    orc_iterations=5,
):
    """Compute joint LRC–ORC sensitivity heatmaps."""

    p_in, p_out = estimate_empirical_pin_pout(
        graph,
        label_attribute=label_attribute,
    )

    print("\n" + "=" * 70)
    print(dataset_name)
    print("=" * 70)
    print(f"empirical p_in  = {p_in:.6f}")
    print(f"empirical p_out = {p_out:.6f}")

    rows = []

    for lrc_pct in LRC_HEAT_VALUES:
        for orc_pct in ORC_HEAT_VALUES:

            print(
                f"\n{dataset_name} | "
                f"LRC={lrc_pct}% | ORC={orc_pct}%"
            )

            try:
                prep_start = time.perf_counter()

                combo_info = prepare_combo_graph(
                    graph=graph,
                    lrc_pct=lrc_pct,
                    orc_pct=orc_pct,
                    alpha=HEATMAP_ALPHA,
                    iterations=orc_iterations,
                    method="Sinkhorn",
                    weight="weight",
                )

                prep_runtime = (
                    time.perf_counter() - prep_start
                )

                for seed in LOUVAIN_SEEDS:
                    community_start = time.perf_counter()

                    result = evaluate_prepared_graph(
                        info=combo_info,
                        algorithm="louvain",
                        seed=seed,
                        weight="weight",
                        label_attribute=label_attribute,
                    )

                    community_runtime = (
                        time.perf_counter() - community_start
                    )

                    # Preserve original heatmap runtime definition:
                    # T measures preprocessing only.
                    total_runtime = prep_runtime

                    rows.append({
                        "dataset": dataset_name,
                        "seed": seed,
                        "lrc_pct": lrc_pct,
                        "orc_pct": orc_pct,
                        "p_in": p_in,
                        "p_out": p_out,
                        "ARI": result["ARI"],
                        "NMI": result["NMI"],
                        "Prep_time": prep_runtime,
                        "Community_time": community_runtime,
                        "Runtime": total_runtime,
                        "status": "ok",
                    })

                    print(
                        f"seed={seed} | "
                        f"ARI={result['ARI']:.3f} | "
                        f"NMI={result['NMI']:.3f} | "
                        f"T={total_runtime:.2f}s"
                    )

            except Exception as error:
                print("FAILED:", repr(error))

    raw_df = pd.DataFrame(rows)

    summary = (
        raw_df.groupby(
            ["lrc_pct", "orc_pct"],
            as_index=False
        )
        .agg(
            ARI_mean=("ARI", "mean"),
            NMI_mean=("NMI", "mean"),
            Runtime_mean=("Runtime", "mean"),
            Prep_time_mean=("Prep_time", "mean"),
            Community_time_mean=("Community_time", "mean"),
        )
    )

    def make_matrix(column):
        return (
            summary.pivot(
                index="lrc_pct",
                columns="orc_pct",
                values=column,
            )
            .reindex(
                index=LRC_HEAT_VALUES,
                columns=ORC_HEAT_VALUES,
            )
        )

    ari_matrix = make_matrix("ARI_mean")
    nmi_matrix = make_matrix("NMI_mean")
    runtime_matrix = make_matrix("Runtime_mean")

    def plot_heatmap(metric_matrix, metric_name):

        if metric_name == "ARI":
            vmin, vmax = -0.5, 1.0
            colorbar_ticks = [-0.5, 1.0]
        else:
            vmin, vmax = 0.0, 1.0
            colorbar_ticks = [0.0, 1.0]

        fig, ax = plt.subplots(figsize=(11, 9))

        im = ax.imshow(
            metric_matrix.values,
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
        )

        ax.set_xticks(
            np.arange(len(ORC_HEAT_VALUES))
        )
        ax.set_xticklabels(
            [f"{x}%" for x in ORC_HEAT_VALUES],
            fontsize=12,
        )

        ax.set_yticks(
            np.arange(len(LRC_HEAT_VALUES))
        )
        ax.set_yticklabels(
            [f"{x}%" for x in LRC_HEAT_VALUES],
            fontsize=12,
        )

        ax.set_xlabel(
            "ORC pruning ratio (%)",
            fontsize=14,
        )
        ax.set_ylabel(
            "LRC pruning ratio (%)",
            fontsize=14,
        )

        ax.set_title(
            f"{dataset_name}: "
            f"Joint LRC–ORC Sensitivity ({metric_name})\n"
            rf"empirical "
            rf"$\hat{{p}}_{{in}}={p_in:.5f}$, "
            rf"$\hat{{p}}_{{out}}={p_out:.5f}$",
            fontsize=16,
        )

        for i, lrc_pct in enumerate(LRC_HEAT_VALUES):
            for j, orc_pct in enumerate(ORC_HEAT_VALUES):

                metric_value = metric_matrix.loc[
                    lrc_pct, orc_pct
                ]
                runtime_value = runtime_matrix.loc[
                    lrc_pct, orc_pct
                ]

                if (
                    pd.isna(metric_value)
                    or pd.isna(runtime_value)
                ):
                    cell_text = "failed"
                else:
                    cell_text = (
                        f"{metric_name}={metric_value:.3f}\n"
                        f"T={runtime_value:.2f}s"
                    )

                ax.text(
                    j, i, cell_text,
                    ha="center",
                    va="center",
                    fontsize=10,
                )

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_ticks(colorbar_ticks)
        cbar.set_label(
            f"Mean {metric_name}",
            fontsize=13,
        )

        fig.tight_layout()
        plt.show()

    plot_heatmap(ari_matrix, "ARI")
    plot_heatmap(nmi_matrix, "NMI")

    return raw_df, summary


# ============================================================
# PUBLIC FUNCTIONS
# ============================================================

__all__ = [
    "download_if_missing",
    "stream_edges",
    "load_communities",
    "load_facebook_combined",
    "load_all_facebook_circles",
    "get_gcc",
    "ensure_edge_weights",
    "run_louvain",
    "nx_to_igraph",
    "run_leiden",
    "eval_partition",
    "compute_LRC",
    "prune_by_percentile",
    "run_orc_flow_gcc",
    "run_orc_no_flow",
    "prepare_lrc_graph",
    "prepare_orc_graph",
    "prepare_combo_graph",
    "evaluate_prepared_graph",
    "evaluate_plain_graph",
    "mean_pm_std",
    "summarize_results",
    "paired_tests",
    "basic_graph_stats",
    "label_structure_stats",
    "compute_community_metrics",
    "compute_label_metrics",
    "weighted_mean",
    "ground_truth_partition",
    "run_combo_sensitivity",
    "summarize_sensitivity",
    "select_best_sensitivity_configs",
    "plot_sensitivity_metric",
    "estimate_empirical_pin_pout",
    "run_real_dataset_heatmap",
]

# ============================================================
# 13. SHARED HEATMAP AND BASELINE UTILITIES
# ============================================================

def restore_reference_nodes(graph, reference_graph):
    """Restore missing nodes without restoring removed edges."""
    if not set(graph).issubset(set(reference_graph)):
        raise ValueError("Graph contains unknown nodes.")

    result = graph.copy()
    for node, attrs in reference_graph.nodes(data=True):
        if node not in result:
            result.add_node(node, **attrs)
        else:
            for key, value in attrs.items():
                result.nodes[node].setdefault(key, value)
    return result


def prepare_pruning_configuration(
    graph, lrc_pct, orc_pct, *, alpha=0.5, iterations=5,
    method="Sinkhorn", weight="weight", preserve_nodes=True
):
    """Original / LRC-only / ORC-RF-only / Hybrid."""

    lrc_pct = float(lrc_pct)
    orc_pct = float(orc_pct)

    if not (0 <= lrc_pct <= 100 and 0 <= orc_pct <= 100):
        raise ValueError("Pruning percentages must be in [0, 100].")

    if lrc_pct == 0 and orc_pct == 0:
        info = dict(
            method_base="Original",
            graph=graph.copy(),
            prep_time=0.0,
            removed_lrc=0,
            removed_orc=0,
            pct_lrc=0.0,
            pct_orc=0.0,
            alpha=np.nan,
            iterations=0,
            cutoff_lrc=np.nan,
            cutoff_orc=np.nan
        )

    elif orc_pct == 0:
        info = prepare_lrc_graph(graph, lrc_pct=lrc_pct)

    elif lrc_pct == 0:
        info = prepare_orc_graph(
            graph, orc_pct=orc_pct,
            alpha=alpha, iterations=iterations,
            method=method, weight=weight
        )

    else:
        info = prepare_combo_graph(
            graph, lrc_pct=lrc_pct, orc_pct=orc_pct,
            alpha=alpha, iterations=iterations,
            method=method, weight=weight
        )

    if preserve_nodes:
        info = {
            **info,
            "graph": restore_reference_nodes(info["graph"], graph)
        }

    return info


def _shared_eval(
    graph, reference_graph, algorithm, seed,
    label_attribute="label_gt", weight="weight"
):
    """Evaluate on the same labeled nodes for every method."""

    eval_nodes = [
        node for node in reference_graph
        if label_attribute in reference_graph.nodes[node]
    ]

    if len(eval_nodes) < 2:
        raise ValueError("At least two labeled nodes required.")

    start = time.perf_counter()

    if algorithm.lower() == "louvain":
        partition = run_louvain(
            graph, seed=int(seed), weight=weight
        )
    elif algorithm.lower() == "leiden":
        partition = run_leiden(
            graph, seed=int(seed), weight=weight
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    cluster_time = time.perf_counter() - start

    if not all(node in partition for node in eval_nodes):
        raise ValueError("Partition does not contain all labeled nodes.")

    y_true = [
        reference_graph.nodes[node][label_attribute]
        for node in eval_nodes
    ]
    y_pred = [partition[node] for node in eval_nodes]

    return (
        float(adjusted_rand_score(y_true, y_pred)),
        float(normalized_mutual_info_score(y_true, y_pred)),
        float(cluster_time),
        len(eval_nodes)
    )


# ============================================================
# SHARED HEATMAP
# ============================================================

def run_shared_pruning_heatmap(
    graph, *, dataset_name,
    lrc_values=(0, 2, 5, 10, 20, 30),
    orc_values=(0, 2, 5, 10, 20, 30),
    alpha=0.5, iterations=5,
    seeds=(0, 1, 2, 3, 4, 42),
    label_attribute="label_gt",
    weight="weight",
    preserve_nodes=True,
    verbose=True
):
    """Heatmap runtime includes preprocessing only."""

    rows = []

    for lrc_pct in lrc_values:
        for orc_pct in orc_values:

            try:
                info = prepare_pruning_configuration(
                    graph, lrc_pct, orc_pct,
                    alpha=alpha,
                    iterations=iterations,
                    weight=weight,
                    preserve_nodes=preserve_nodes
                )

                for seed in seeds:
                    ari, nmi, cluster_time, n_eval = _shared_eval(
                        info["graph"], graph, "Louvain", seed,
                        label_attribute=label_attribute,
                        weight=weight
                    )

                    rows.append(dict(
                        Dataset=dataset_name,
                        seed=int(seed),
                        lrc_pct=float(lrc_pct),
                        orc_pct=float(orc_pct),
                        alpha=float(alpha),
                        ARI=ari,
                        NMI=nmi,
                        Runtime=float(info["prep_time"]),
                        Prep_time=float(info["prep_time"]),
                        Community_time=cluster_time,
                        V=info["graph"].number_of_nodes(),
                        E=info["graph"].number_of_edges(),
                        n_eval=n_eval,
                        status="ok",
                        error=""
                    ))

            except Exception as exc:
                if verbose:
                    print(
                        f"{dataset_name}: LRC={lrc_pct}, "
                        f"ORC={orc_pct} FAILED: {exc!r}"
                    )

                for seed in seeds:
                    rows.append(dict(
                        Dataset=dataset_name,
                        seed=int(seed),
                        lrc_pct=float(lrc_pct),
                        orc_pct=float(orc_pct),
                        alpha=float(alpha),
                        ARI=np.nan,
                        NMI=np.nan,
                        Runtime=np.nan,
                        Prep_time=np.nan,
                        Community_time=np.nan,
                        V=np.nan,
                        E=np.nan,
                        n_eval=np.nan,
                        status="failed",
                        error=repr(exc)
                    ))

            else:
                if verbose:
                    print(
                        f"{dataset_name}: LRC={lrc_pct:g}%, "
                        f"ORC={orc_pct:g}% | "
                        f"Tprep={info['prep_time']:.3f}s"
                    )

    raw = pd.DataFrame(rows)
    valid = raw.loc[raw["status"] == "ok"].copy()

    if valid.empty:
        raise RuntimeError(
            "No successful heatmap configurations."
        )

    summary = valid.groupby(
        ["lrc_pct", "orc_pct"], as_index=False
    ).agg(
        ARI_mean=("ARI", "mean"),
        ARI_std=("ARI", "std"),
        NMI_mean=("NMI", "mean"),
        NMI_std=("NMI", "std"),
        Runtime_mean=("Prep_time", "mean"),
        Runtime_std=("Prep_time", "std"),
        Prep_time_mean=("Prep_time", "mean"),
        Community_time_mean=("Community_time", "mean"),
        V_mean=("V", "mean"),
        E_mean=("E", "mean"),
        n_eval_mean=("n_eval", "mean"),
        n_runs=("seed", "count")
    )

    return raw, summary


def plot_shared_pruning_heatmap(
    summary, *, dataset_name, metric,
    lrc_values, orc_values,
    pin=None, pout=None,
    figsize=(9, 7),
    runtime_column="Prep_time_mean"
):
    """Plot accuracy and pipeline-only runtime."""

    key = {
        "ARI": "ARI_mean",
        "NMI": "NMI_mean"
    }[metric]

    matrix = summary.pivot(
        index="lrc_pct",
        columns="orc_pct",
        values=key
    ).reindex(
        index=list(map(float, lrc_values)),
        columns=list(map(float, orc_values))
    )

    runtime = summary.pivot(
        index="lrc_pct",
        columns="orc_pct",
        values=runtime_column
    ).reindex(
        index=list(map(float, lrc_values)),
        columns=list(map(float, orc_values))
    )

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(
        np.ma.masked_invalid(matrix.to_numpy(dtype=float)),
        vmin=-0.5 if metric == "ARI" else 0,
        vmax=1,
        aspect="auto"
    )

    ax.set_xticks(
        range(len(orc_values)),
        [f"{p:g}%" for p in orc_values]
    )
    ax.set_yticks(
        range(len(lrc_values)),
        [f"{p:g}%" for p in lrc_values]
    )

    ax.set_xlabel("ORC pruning ratio (%)")
    ax.set_ylabel("LRC pruning ratio (%)")

    title = f"{dataset_name}: Joint LRC–ORC Sensitivity ({metric})"

    if pin is not None and pout is not None:
        title += (
            f"\nempirical $\\hat{{p}}_{{in}}={pin:.4f}$, "
            f"$\\hat{{p}}_{{out}}={pout:.2e}$"
        )

    ax.set_title(title)

    for i in range(len(lrc_values)):
        for j in range(len(orc_values)):
            value = matrix.iat[i, j]
            seconds = runtime.iat[i, j]

            if np.isfinite(value) and np.isfinite(seconds):
                label = (
                    f"{metric}={value:.3f}\n"
                    f"Tprep={seconds:.2f}s"
                )
            else:
                label = "failed"

            ax.text(
                j, i, label,
                ha="center", va="center", fontsize=8
            )

    fig.colorbar(im, ax=ax).set_label(f"Mean {metric}")
    fig.tight_layout()

    return fig


# ============================================================
# SHARED PRUNING BASELINES
# ============================================================

def prune_exact_edge_budget(
    graph, k, criterion, *, seed=42
):
    """Remove exactly k edges; preserve all nodes."""

    result = graph.copy()
    edges = list(result.edges())
    k = int(k)

    if not (0 <= k <= len(edges)):
        raise ValueError("Invalid edge removal count.")

    if k == 0:
        return result

    rng = np.random.default_rng(seed)

    if criterion == "Random":
        chosen = rng.choice(len(edges), k, replace=False)

    else:
        if criterion == "LRC":
            compute_LRC(result, attribute="LRC")

        neighbors = {
            node: set(result.neighbors(node))
            for node in result.nodes()
        }

        scores = []

        for u, v in edges:
            common = len(neighbors[u] & neighbors[v])

            if criterion == "LRC":
                score = result[u][v]["LRC"]

            elif criterion == "Jaccard":
                union = len(neighbors[u] | neighbors[v])
                score = common / union if union else 0.0

            elif criterion == "Triangle":
                score = float(common)

            elif criterion == "Forman":
                # Triangle-augmented unweighted Forman proxy
                score = float(
                    4 - len(neighbors[u])
                    - len(neighbors[v])
                    + 3 * common
                )

            else:
                raise ValueError(f"Unknown criterion: {criterion}")

            scores.append(score)

        scores = np.asarray(scores, dtype=float)

        if not np.isfinite(scores).all():
            raise ValueError("Non-finite edge scores.")

        chosen = np.lexsort(
            (rng.random(len(edges)), scores)
        )[:k]

    result.remove_edges_from(
        [edges[int(index)] for index in chosen]
    )

    return result


def prune_orc_exact_edge_budget(
    graph, k, *,
    alpha=0.5,
    iterations=5,
    use_flow=True,
    seed=42,
    method="Sinkhorn",
    weight="weight"
):
    """ORC on GCC; exactly k edges removed."""

    output = graph.copy()
    k = int(k)

    if k == 0 and output.number_of_edges() == 0:
        return output

    gcc = get_gcc(output)

    if not (0 <= k <= gcc.number_of_edges()):
        raise ValueError(
            "ORC removal budget exceeds GCC edge count."
        )

    if use_flow:
        processed = run_orc_flow_gcc(
            gcc, alpha=alpha,
            iterations=iterations,
            method=method,
            weight=weight
        )
    else:
        processed = run_orc_no_flow(
            gcc, alpha=alpha,
            method=method,
            weight=weight
        )

    original_edges = set(
        map(frozenset, gcc.edges())
    )
    processed_edges = set(
        map(frozenset, processed.edges())
    )

    if (
        set(processed.nodes()) != set(gcc.nodes())
        or processed_edges != original_edges
    ):
        raise RuntimeError(
            "ORC changed graph topology before pruning."
        )

    for u, v, attrs in processed.edges(data=True):
        output[u][v].update(attrs)

    if k:
        edges = list(processed.edges())
        scores = np.asarray([
            processed[u][v].get("ricciCurvature", np.nan)
            for u, v in edges
        ], dtype=float)

        if not np.isfinite(scores).all():
            raise ValueError("Invalid ORC curvature values.")

        rng = np.random.default_rng(seed)

        chosen = np.lexsort(
            (rng.random(len(edges)), scores)
        )[:k]

        output.remove_edges_from(
            [edges[int(index)] for index in chosen]
        )

    return output


# ============================================================
# SHARED BASELINE AND ABLATION COMPARISON
# ============================================================

def run_shared_baseline_comparison(
    graph, *, prepared_info, dataset_name,
    seeds=(0, 1, 2, 3, 4, 42),
    alpha=0.5,
    iterations=5,
    lrc_pct=2.5,
    orc_pct=2.5,
    edge_seed=42,
    label_attribute="label_gt",
    weight="weight",
    verbose=True
):
    """Run original methods, matched baselines, and ablations.

    TotalTime = PrepTime + ClusterTime.
    """

    required = {
        "LRC_only",
        "ORC_RF_only",
        "Hybrid_LRC_ORC_RF"
    }

    if not required.issubset(prepared_info):
        raise ValueError(
            "Missing required prepared_info methods."
        )

    original = graph.copy()
    n_edges = original.number_of_edges()

    if n_edges == 0:
        raise ValueError("The graph has no edges.")

    for u, v in original.edges():
        original[u][v].setdefault(weight, 1.0)

    graphs = {"Original": original}
    prep_times = {"Original": 0.0}

    for name in (
        "LRC_only",
        "ORC_RF_only",
        "Hybrid_LRC_ORC_RF"
    ):
        info = prepared_info[name]

        graphs[name] = restore_reference_nodes(
            info["graph"], original
        )
        prep_times[name] = float(info["prep_time"])

    hybrid_edges = graphs[
        "Hybrid_LRC_ORC_RF"
    ].number_of_edges()

    total_remove = n_edges - hybrid_edges

    lrc_remove = int(
        prepared_info["Hybrid_LRC_ORC_RF"]["removed_lrc"]
    )

    if not (0 <= lrc_remove <= total_remove):
        raise ValueError("Inconsistent Hybrid removal budget.")

    def register(name, builder):
        if verbose:
            print("Preparing", name, "...")

        start = time.perf_counter()
        prepared_graph = restore_reference_nodes(
            builder(), original
        )
        elapsed = time.perf_counter() - start

        if set(prepared_graph) != set(original):
            raise RuntimeError("Node mismatch: " + name)

        graphs[name] = prepared_graph
        prep_times[name] = elapsed

        if verbose:
            print(
                f"  Edges={prepared_graph.number_of_edges()}, "
                f"PrepTime={elapsed:.4f}s"
            )

    for criterion in (
        "Random",
        "Jaccard",
        "Triangle",
        "Forman",
        "LRC"
    ):
        register(
            f"{criterion}_only_matched",
            lambda c=criterion: prune_exact_edge_budget(
                original, total_remove, c,
                seed=edge_seed
            )
        )

    register(
        "ORC_RF_only_matched",
        lambda: prune_orc_exact_edge_budget(
            original, total_remove,
            alpha=alpha,
            iterations=iterations,
            seed=edge_seed,
            weight=weight
        )
    )

    for criterion in ("Random", "Jaccard", "Forman"):

        def build_two_stage(c=criterion):
            first = prune_exact_edge_budget(
                original, lrc_remove, c,
                seed=edge_seed
            )
            second_remove = (
                first.number_of_edges() - hybrid_edges
            )
            return prune_orc_exact_edge_budget(
                first, second_remove,
                alpha=alpha,
                iterations=iterations,
                seed=edge_seed,
                weight=weight
            )

        register(
            f"{criterion}_ORC_RF",
            build_two_stage
        )

    register(
        "Hybrid_without_flow",
        lambda: prepare_combo_graph(
            original,
            lrc_pct=lrc_pct,
            orc_pct=orc_pct,
            alpha=alpha,
            iterations=iterations,
            method="Sinkhorn",
            weight=weight,
            use_ricci_flow=False
        )["graph"]
    )

    def unit_weight_graph():
        result = graphs[
            "Hybrid_LRC_ORC_RF"
        ].copy()
        nx.set_edge_attributes(result, 1.0, weight)
        return result

    register("Hybrid_unit_weights", unit_weight_graph)

    prep_times["Hybrid_unit_weights"] += (
        prep_times["Hybrid_LRC_ORC_RF"]
    )

    rows = []

    for method_name, prepared_graph in graphs.items():
        for algorithm in ("Louvain", "Leiden"):
            for seed in seeds:

                ari, nmi, cluster_time, n_eval = _shared_eval(
                    prepared_graph,
                    original,
                    algorithm,
                    seed,
                    label_attribute=label_attribute,
                    weight=weight
                )

                rows.append(dict(
                    Dataset=dataset_name,
                    Method=method_name,
                    Algorithm=algorithm,
                    Seed=int(seed),
                    ARI=ari,
                    NMI=nmi,
                    PrepTime=prep_times[method_name],
                    ClusterTime=cluster_time,
                    TotalTime=(
                        prep_times[method_name] + cluster_time
                    ),
                    OriginalEdges=n_edges,
                    RemainingEdges=prepared_graph.number_of_edges(),
                    RemovedEdges=(
                        n_edges - prepared_graph.number_of_edges()
                    ),
                    EvaluatedNodes=n_eval,
                    Components=nx.number_connected_components(
                        prepared_graph
                    ),
                    IsolatedNodes=nx.number_of_isolates(
                        prepared_graph
                    )
                ))

        if verbose:
            print("Evaluated:", method_name)

    raw = pd.DataFrame(rows)

    summary = raw.groupby(
        ["Method", "Algorithm"], as_index=False
    ).agg(
        ARI_mean=("ARI", "mean"),
        ARI_std=("ARI", "std"),
        NMI_mean=("NMI", "mean"),
        NMI_std=("NMI", "std"),
        Time_mean=("TotalTime", "mean"),
        Time_std=("TotalTime", "std"),
        PrepTime=("PrepTime", "first"),
        ClusterTime_mean=("ClusterTime", "mean"),
        Edges=("RemainingEdges", "first"),
        Nodes=("EvaluatedNodes", "first"),
        Components=("Components", "first"),
        IsolatedNodes=("IsolatedNodes", "first")
    ).sort_values(
        ["Algorithm", "ARI_mean"],
        ascending=[True, False]
    ).reset_index(drop=True)

    if not np.allclose(
        summary["Time_mean"],
        summary["PrepTime"] + summary["ClusterTime_mean"]
    ):
        raise RuntimeError("Runtime accounting mismatch.")

    return raw, summary


# ============================================================
# SAVE BASELINE TABLES AND FIGURES
# ============================================================

def save_shared_baseline_outputs(
    raw, summary, *,
    output_dir, dataset_name,
    show_plots=True
):
    """Save tables, LaTeX, and two runtime figures."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw.to_csv(
        output_dir / "all_baseline_runs.csv",
        index=False
    )
    summary.to_csv(
        output_dir / "baseline_summary.csv",
        index=False
    )

    wide = raw.pivot(
        index=["Seed", "Algorithm"],
        columns="Method",
        values=[
            "ARI", "NMI",
            "TotalTime", "RemainingEdges"
        ]
    )

    wide.columns = [
        f"{method}__{metric}"
        for metric, method in wide.columns
    ]

    wide.reset_index().to_csv(
        output_dir / "side_by_side_columns.csv",
        index=False
    )

    paper = summary.assign(
        ARI=summary.apply(
            lambda r: f"{r.ARI_mean:.3f} ± {r.ARI_std:.3f}",
            axis=1
        ),
        NMI=summary.apply(
            lambda r: f"{r.NMI_mean:.3f} ± {r.NMI_std:.3f}",
            axis=1
        ),
        **{
            "PrepTime (s)": summary["PrepTime"].round(4),
            "ClusterTime (s)": summary["ClusterTime_mean"].round(4),
            "Total Runtime (s)": summary.apply(
                lambda r: (
                    f"{r.Time_mean:.3f} ± {r.Time_std:.3f}"
                ),
                axis=1
            )
        }
    )[
        [
            "Method", "Algorithm", "ARI", "NMI",
            "PrepTime (s)", "ClusterTime (s)",
            "Total Runtime (s)", "Edges", "Nodes"
        ]
    ]

    paper.to_csv(
        output_dir / "paper_ready_comparison.csv",
        index=False
    )

    paper.to_latex(
        output_dir / "paper_ready_comparison.tex",
        index=False,
        escape=True,
        caption=(
            f"{dataset_name} baseline comparison: "
            "preparation plus clustering."
        ),
        label=f"tab:{dataset_name.lower().replace(' ', '_')}_baselines"
    )

    # Accuracy vs total runtime
    fig, axes = plt.subplots(
        1, 2, figsize=(17, 7), sharey=True
    )

    for ax, algorithm in zip(
        axes, ("Louvain", "Leiden")
    ):
        group = summary.loc[
            summary.Algorithm == algorithm
        ]

        for i, (_, row) in enumerate(group.iterrows()):
            hybrid = (
                row.Method == "Hybrid_LRC_ORC_RF"
            )

            ax.scatter(
                row.Time_mean,
                row.ARI_mean,
                s=190 if hybrid else 60,
                marker="*" if hybrid else "o"
            )

            ax.annotate(
                row.Method,
                (row.Time_mean, row.ARI_mean),
                xytext=(5, 7 if i % 2 else -10),
                textcoords="offset points",
                fontsize=7
            )

        ax.set_xscale("log")
        ax.set_title(
            f"{dataset_name} — {algorithm}"
        )
        ax.set_xlabel(
            "Mean Total Runtime (seconds, log scale)"
        )
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("Mean ARI")
    fig.tight_layout()

    fig.savefig(
        output_dir / "accuracy_vs_runtime.png",
        dpi=300,
        bbox_inches="tight"
    )

    if show_plots:
        from IPython.display import display
        display(fig)

    plt.close(fig)

    # Runtime breakdown
    fig, axes = plt.subplots(
        1, 2, figsize=(17, 8)
    )

    for ax, algorithm in zip(
        axes, ("Louvain", "Leiden")
    ):
        group = summary.loc[
            summary.Algorithm == algorithm
        ].sort_values("Time_mean")

        positions = np.arange(len(group))

        ax.barh(
            positions,
            group.PrepTime,
            label="Preparation"
        )

        ax.barh(
            positions,
            group.ClusterTime_mean,
            left=group.PrepTime,
            label="Community detection"
        )

        ax.set_yticks(positions)
        ax.set_yticklabels(
            group.Method, fontsize=8
        )
        ax.set_xlabel(
            "Mean Total Runtime (seconds)"
        )
        ax.set_title(algorithm)
        ax.legend(loc="lower right")

    fig.tight_layout()

    fig.savefig(
        output_dir / "runtime_breakdown.png",
        dpi=300,
        bbox_inches="tight"
    )

    if show_plots:
        from IPython.display import display
        display(fig)

    plt.close(fig)

    return paper


# ============================================================
# PUBLIC FUNCTIONS
# ============================================================

__all__.extend([
    "restore_reference_nodes",
    "prepare_pruning_configuration",
    "run_shared_pruning_heatmap",
    "plot_shared_pruning_heatmap",
    "prune_exact_edge_budget",
    "prune_orc_exact_edge_budget",
    "run_shared_baseline_comparison",
    "save_shared_baseline_outputs"
])


# ============================================================
# SHARED BASELINE VISUALIZATION — PNG ONLY
# ============================================================

def plot_shared_baseline_figures(
    summary,
    *,
    dataset_name,
    output_dir=None,
    show=True,
    annotate_methods=(
        "Original",
        "LRC_only",
        "ORC_RF_only",
        "Hybrid_LRC_ORC_RF",
        "Random_ORC_RF",
    ),
):
    """Plot baseline accuracy and runtime without CSV/LaTeX."""

    if summary.empty:
        raise ValueError("Baseline summary is empty.")

    required = {
        "Method", "Algorithm", "ARI_mean", "Time_mean",
        "PrepTime", "ClusterTime_mean",
    }

    missing = required - set(summary.columns)

    if missing:
        raise KeyError(
            f"Missing baseline columns: {sorted(missing)}"
        )

    output_path = (
        Path(output_dir)
        if output_dir is not None
        else None
    )

    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Accuracy versus total runtime
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1, 2, figsize=(14, 6), sharey=True
    )

    for ax, algorithm in zip(
        axes, ("Louvain", "Leiden")
    ):
        subset = summary.loc[
            summary["Algorithm"] == algorithm
        ]

        for _, result in subset.iterrows():
            is_hybrid = (
                result["Method"] == "Hybrid_LRC_ORC_RF"
            )

            ax.scatter(
                result["Time_mean"],
                result["ARI_mean"],
                marker="*" if is_hybrid else "o",
                s=125 if is_hybrid else 40,
                alpha=0.9,
            )

            if result["Method"] in annotate_methods:
                ax.annotate(
                    result["Method"],
                    (
                        result["Time_mean"],
                        result["ARI_mean"]
                    ),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                )

        if (
            not subset.empty
            and (subset["Time_mean"] > 0).all()
        ):
            ax.set_xscale("log")

        ax.set_title(f"{dataset_name} — {algorithm}")
        ax.set_xlabel("Mean total runtime (seconds)")
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("Mean ARI")
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(
            output_path / "accuracy_vs_runtime.png",
            dpi=300,
            bbox_inches="tight"
        )

    if show:
        from IPython.display import display
        display(fig)

    plt.close(fig)

    # --------------------------------------------------------
    # Preparation versus clustering runtime
    # --------------------------------------------------------

    fig, axes = plt.subplots(1, 2, figsize=(15, 9))

    for ax, algorithm in zip(
        axes, ("Louvain", "Leiden")
    ):
        subset = summary.loc[
            summary["Algorithm"] == algorithm
        ].sort_values("Time_mean")

        positions = np.arange(len(subset))

        ax.barh(
            positions,
            subset["PrepTime"],
            label="Preparation"
        )

        ax.barh(
            positions,
            subset["ClusterTime_mean"],
            left=subset["PrepTime"],
            label="Clustering"
        )

        ax.set_yticks(positions)
        ax.set_yticklabels(
            subset["Method"], fontsize=8
        )

        ax.set_title(f"{dataset_name} — {algorithm}")
        ax.set_xlabel("Mean total runtime (seconds)")
        ax.grid(axis="x", alpha=0.25)
        ax.legend(loc="lower right")

    fig.tight_layout()

    if output_path is not None:
        fig.savefig(
            output_path / "runtime_breakdown.png",
            dpi=300,
            bbox_inches="tight"
        )

    if show:
        from IPython.display import display
        display(fig)

    plt.close(fig)


__all__.append("plot_shared_baseline_figures")

# ============================================================
# SHARED BASELINE FIGURES — AMAZON / DBLP / OTHER DATASETS
# ============================================================

def plot_shared_baseline_figures(
    summary, *, dataset_name, output_dir=None, show=True
):
    """Display and optionally save baseline figures; no CSV or LaTeX."""
    from pathlib import Path
    import numpy as np
    import matplotlib.pyplot as plt

    required = {
        "Method", "Algorithm", "ARI_mean", "Time_mean",
        "PrepTime", "ClusterTime_mean",
    }
    missing = required - set(summary.columns)
    if missing:
        raise KeyError(f"Missing columns: {sorted(missing)}")
    if summary.empty:
        raise ValueError("Baseline summary is empty")

    output_path = Path(output_dir) if output_dir is not None else None
    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)

    algorithms = ("Louvain", "Leiden")
    highlight = {
        "Original", "LRC_only", "ORC_RF_only",
        "Hybrid_LRC_ORC_RF", "Random_ORC_RF",
    }

    # --------------------------------------------------------
    # Figure 1: Accuracy versus total runtime
    # --------------------------------------------------------

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

    for ax, algorithm in zip(axes, algorithms):
        part = summary[summary["Algorithm"] == algorithm]

        for _, row in part.iterrows():
            hybrid = row["Method"] == "Hybrid_LRC_ORC_RF"

            ax.scatter(
                row["Time_mean"],
                row["ARI_mean"],
                s=140 if hybrid else 45,
                marker="*" if hybrid else "o",
            )

            if row["Method"] in highlight:
                ax.annotate(
                    row["Method"],
                    (row["Time_mean"], row["ARI_mean"]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                )

        if not part.empty and (part["Time_mean"] > 0).all():
            ax.set_xscale("log")

        ax.set_title(f"{dataset_name} — {algorithm}")
        ax.set_xlabel("Mean total runtime (seconds)")
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("Mean ARI")
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(
            output_path / "accuracy_vs_runtime.png",
            dpi=300,
            bbox_inches="tight",
        )

    if show:
        from IPython.display import display
        display(fig)

    plt.close(fig)

    # --------------------------------------------------------
    # Figure 2: Preparation versus clustering runtime
    # --------------------------------------------------------

    fig, axes = plt.subplots(1, 2, figsize=(15, 8))

    for ax, algorithm in zip(axes, algorithms):
        part = summary[
            summary["Algorithm"] == algorithm
        ].sort_values("Time_mean")

        y = np.arange(len(part))

        ax.barh(
            y,
            part["PrepTime"],
            label="Preparation",
        )

        ax.barh(
            y,
            part["ClusterTime_mean"],
            left=part["PrepTime"],
            label="Clustering",
        )

        ax.set_yticks(y)
        ax.set_yticklabels(part["Method"], fontsize=8)
        ax.set_title(f"{dataset_name} — {algorithm}")
        ax.set_xlabel("Mean total runtime (seconds)")
        ax.grid(axis="x", alpha=0.25)
        ax.legend()

    fig.tight_layout()

    if output_path is not None:
        fig.savefig(
            output_path / "runtime_breakdown.png",
            dpi=300,
            bbox_inches="tight",
        )

    if show:
        from IPython.display import display
        display(fig)

    plt.close(fig)


# Make the new function publicly available
if "__all__" in globals():
    if "plot_shared_baseline_figures" not in __all__:
        __all__.append("plot_shared_baseline_figures")

