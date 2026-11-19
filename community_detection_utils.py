"""
Shared utilities for curvature-guided community detection experiments.

This module collects the common functions used by the Amazon, DBLP,
Facebook, and YouTube notebooks.

Keep dataset-specific constants, dataset selection, plots, and final tables
inside the notebooks. Pass experiment parameters explicitly to these
functions instead of relying on notebook-level global variables.
"""

from __future__ import annotations

import gzip
import os
import time
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Hashable, Iterable, Iterator, List, Mapping, Optional, Sequence, Set, Tuple, Union

import igraph as ig
import leidenalg
import networkx as nx
import numpy as np
import pandas as pd
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from scipy.stats import ttest_rel, wilcoxon
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

import community.community_louvain as community_louvain


Node = Hashable
Partition = Mapping[Node, int]
PathLike = Union[str, os.PathLike]


# ============================================================
# 1. FILE AND DATA LOADING
# ============================================================

def download_if_missing(url: str, path: PathLike) -> None:
    """Download a file only when it does not already exist."""
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
    """Read a tab-separated SNAP edge list in chunks."""
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
    """Load tab-separated communities from a gzip-compressed SNAP file."""
    communities: List[List[int]] = []

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

            nodes = [int(value) for value in line.split("\t") if value != ""]

            if len(nodes) >= min_size:
                communities.append(nodes)

    return communities


def load_facebook_combined(path: PathLike) -> nx.Graph:
    """Load the gzip-compressed Facebook combined edge list."""
    graph = nx.Graph()

    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as file:
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
    """Load all Facebook ego-network circle files from a directory."""
    communities: List[Dict[str, Any]] = []
    circle_files = sorted(Path(extract_dir).glob("*.circles"))

    print("Circle files:", len(circle_files))

    for file in circle_files:
        ego_id = int(file.stem)

        with open(file, "r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                parts = line.strip().split()

                if len(parts) <= 1:
                    continue

                circle_name = parts[0]
                nodes = [int(value) for value in parts[1:]]

                if len(nodes) >= min_size:
                    communities.append(
                        {
                            "ego_id": ego_id,
                            "circle_name": circle_name,
                            "nodes": nodes,
                            "size": len(nodes),
                        }
                    )

    return communities


# ============================================================
# 2. GRAPH PREPROCESSING
# ============================================================

def get_gcc(graph: nx.Graph) -> nx.Graph:
    """Return the graph's giant connected component as a copy."""
    if graph.number_of_nodes() == 0 or graph.number_of_edges() == 0:
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
    """Ensure every edge has a numeric weight attribute."""
    output = graph.copy() if copy_graph else graph

    for u, v in output.edges():
        output[u][v][weight] = output[u][v].get(weight, default_weight)

    return output


# ============================================================
# 3. COMMUNITY DETECTION
# ============================================================

def run_louvain(
    graph: nx.Graph,
    weight: str = "weight",
    seed: int = 42,
) -> Dict[Node, int]:
    """Run Louvain and return a node-to-community dictionary."""
    if graph.number_of_nodes() == 0:
        return {}

    graph_run = ensure_edge_weights(
        graph,
        weight=weight,
        default_weight=1.0,
        copy_graph=True,
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
    """Convert a NetworkX graph to an igraph graph."""
    nodes = list(graph.nodes())
    node_to_idx = {node: index for index, node in enumerate(nodes)}

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
    """Run Leiden and return a node-to-community dictionary."""
    if graph.number_of_nodes() == 0:
        return {}

    graph_run = ensure_edge_weights(
        graph,
        weight=weight,
        default_weight=1.0,
        copy_graph=True,
    )

    ig_graph, nodes = nx_to_igraph(graph_run, weight=weight)

    partition = leidenalg.find_partition(
        ig_graph,
        leidenalg.ModularityVertexPartition,
        weights=ig_graph.es["weight"],
        seed=seed,
    )

    result: Dict[Node, int] = {}

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
    """Evaluate a partition using ARI and NMI on labeled nodes."""
    nodes = [
        node
        for node in graph.nodes()
        if label_attribute in graph.nodes[node] and node in partition
    ]

    if len(nodes) == 0:
        return np.nan, np.nan, 0

    y_true = [graph.nodes[node][label_attribute] for node in nodes]
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
    """
    Compute and store the lower Ricci curvature (LRC) score for each edge.

    For an edge ``(u, v)``, let ``d_u`` and ``d_v`` be the endpoint degrees
    and ``c_uv`` the number of common neighbors. The unweighted LRC score is

    ``2/d_u + 2/d_v - 2 + 2*c_uv/max(d_u, d_v) + c_uv/min(d_u, d_v)``.

    Args:
        graph: NetworkX graph whose topology defines the neighborhoods and
            degrees. Edge weights, if present, are not used. The graph's
            topology must remain unchanged during this call.
        attribute: Edge-attribute name in which to store each score. Existing
            values under this name are overwritten; defaults to ``"LRC"``.

    Returns:
        The same graph object, annotated with the requested attribute on each
        edge. Isolated nodes have no incident edge to annotate.

    Notes:
        Neighbor sets and degrees are cached for this call, using additional
        ``O(V + E)`` memory to avoid rebuilding neighborhoods for every edge.
        An edge with a zero-degree endpoint is assigned ``-inf`` to avoid
        division by zero.
    """
    neighborhoods = {
        node: set(graph.neighbors(node))
        for node in graph
    }
    degrees = {
        node: len(neighbors)
        for node, neighbors in neighborhoods.items()
    }

    for u, v in graph.edges():
        degree_u = degrees[u]
        degree_v = degrees[v]
        common_neighbors = len(neighborhoods[u] & neighborhoods[v])

        if degree_u == 0 or degree_v == 0:
            lrc = -np.inf
        else:
            lrc = (
                2.0 / degree_u
                + 2.0 / degree_v
                - 2.0
                + 2.0 * common_neighbors / max(degree_u, degree_v)
                + common_neighbors / min(degree_u, degree_v)
            )

        graph[u][v][attribute] = float(lrc)

    return graph


def prune_by_percentile(
    graph: nx.Graph,
    attr: str,
    pct: float,
    remove_equal_to_cutoff: bool = True,
) -> Tuple[nx.Graph, Optional[float], int]:
    """
    Remove low-valued edges according to an attribute percentile.

    The original notebooks removed edges with value <= cutoff. Set
    remove_equal_to_cutoff=False to remove only values < cutoff.
    """
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
) -> nx.Graph:
    """Run Ollivier-Ricci flow and return the resulting graph."""
    graph_orc = ensure_edge_weights(
        graph,
        weight=weight,
        default_weight=1.0,
        copy_graph=True,
    )

    orc = OllivierRicci(
        graph_orc,
        alpha=alpha,
        method=method,
        weight=weight,
        verbose=verbose,
    )

    orc.compute_ricci_flow(iterations=iterations)
    return orc.G.copy()
def run_orc_no_flow(
    graph: nx.Graph,
    alpha: float = 0.5,
    method: str = "Sinkhorn",
    weight: str = "weight",
    verbose: str = "ERROR",
) -> nx.Graph:

    graph_orc = ensure_edge_weights(
        graph,
        weight=weight,
        default_weight=1.0,
        copy_graph=True,
    )

    orc = OllivierRicci(
        graph_orc,
        alpha=alpha,
        method=method,
        weight=weight,
        verbose=verbose,
    )

    orc.compute_ricci_curvature()

    return orc.G.copy()
# ============================================================
# 7. PREPARE THE THREE EXPERIMENTAL GRAPHS
# ============================================================

def prepare_lrc_graph(
    graph: nx.Graph,
    lrc_pct: float,
    lrc_attribute: str = "LRC",
    print_distribution: bool = False,
) -> Dict[str, Any]:
    """Compute LRC, prune low-LRC edges, and retain all remaining components"""
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
) -> Dict[str, Any]:

    """Run ORC Ricci flow, prune low-curvature edges, and return the sparsified graph."""

    start = time.perf_counter()

    # 1. Run ORC Ricci flow
    flow_graph = run_orc_flow_gcc(
        graph,
        alpha=alpha,
        iterations=iterations,
        method=method,
        weight=weight,
    )

    # 2. Prune low-ORC edges
    pruned_graph, cutoff_orc, removed_orc = prune_by_percentile(
        flow_graph,
        attr=orc_attribute,
        pct=orc_pct,
    )

    # 3. Do NOT take GCC again after pruning
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

def _split_lrc_pruned_graph(
    graph: nx.Graph,
) -> Tuple[nx.Graph, nx.Graph]:
    """Split an LRC-pruned graph so ORC can run only on its GCC.

    Args:
        graph: Graph after LRC edge pruning.

    Returns:
        The GCC copy and a copy containing all other components. Keeping the
        latter separate lets the sweep reattach it after processing the GCC.
    """
    if graph.number_of_nodes() == 0:
        return graph.copy(), graph.copy()

    gcc_nodes = max(nx.connected_components(graph), key=len)
    gcc_graph = graph.subgraph(gcc_nodes).copy()

    if len(gcc_nodes) == graph.number_of_nodes():
        non_gcc_graph = graph.subgraph([]).copy()
    else:
        non_gcc_nodes = set(graph.nodes()) - gcc_nodes
        non_gcc_graph = graph.subgraph(non_gcc_nodes).copy()

    return gcc_graph, non_gcc_graph


def _run_combo_orc(
    gcc_graph: nx.Graph,
    alpha: float,
    iterations: int,
    method: str,
    weight: str,
    use_ricci_flow: bool,
) -> nx.Graph:
    """Run the selected ORC operation on the GCC.

    Args:
        gcc_graph: LRC-pruned giant component to process.
        alpha: ORC idleness parameter.
        iterations: Ricci-flow iterations; used only when flow is enabled.
        method: Optimal-transport method for the ORC calculation.
        weight: Edge-weight attribute name.
        use_ricci_flow: Whether to run iterative Ricci flow or curvature only.

    Returns:
        A processed copy of the GCC with ORC attributes.
    """
    if use_ricci_flow:
        return run_orc_flow_gcc(
            gcc_graph,
            alpha=alpha,
            iterations=iterations,
            method=method,
            weight=weight,
        )

    return run_orc_no_flow(
        gcc_graph,
        alpha=alpha,
        method=method,
        weight=weight,
    )


def _prune_and_reassemble_combo_graph(
    flow_graph: nx.Graph,
    non_gcc_graph: nx.Graph,
    orc_pct: float,
    orc_attribute: str,
) -> Tuple[nx.Graph, Optional[float], int]:
    """Apply one ORC cutoff and restore components excluded from ORC.

    Args:
        flow_graph: ORC-processed GCC.
        non_gcc_graph: LRC-pruned components that bypassed ORC.
        orc_pct: Percentile cutoff for ORC edge pruning.
        orc_attribute: Edge attribute containing ORC values.

    Returns:
        The reassembled graph, ORC cutoff, and number of removed GCC edges.
        Keeping this stage separate allows multiple cutoffs to reuse one flow.
    """
    if flow_graph.number_of_nodes() == 0:
        return flow_graph.copy(), np.nan, 0

    processed_gcc, cutoff_orc, removed_orc = prune_by_percentile(
        flow_graph,
        attr=orc_attribute,
        pct=orc_pct,
    )

    final_graph = nx.compose(processed_gcc, non_gcc_graph)

    return final_graph, cutoff_orc, removed_orc


def _amortized_combo_prep_time(
    lrc_compute_time: float,
    lrc_stage_time: float,
    flow_time: float,
    orc_stage_time: float,
    total_configs: int,
    configs_per_lrc_stage: int,
    orc_variants: int,
) -> float:
    """Amortize shared sweep stages into one configuration's prep time.

    Args:
        lrc_compute_time: Time to annotate the input graph with LRC.
        lrc_stage_time: Time to prune LRC edges and split components.
        flow_time: Time to process the GCC with ORC.
        orc_stage_time: Time for this cutoff's ORC pruning and reassembly.
        total_configs: Number of configurations sharing LRC computation.
        configs_per_lrc_stage: Configurations sharing this LRC-pruned graph.
        orc_variants: ORC cutoffs sharing this flow result.

    Returns:
        Amortized preprocessing seconds for one configuration.
    """
    return (
        lrc_compute_time / total_configs
        + lrc_stage_time / configs_per_lrc_stage
        + flow_time / orc_variants
        + orc_stage_time
    )


def _combo_preparation_info(
    graph: nx.Graph,
    prep_time: float,
    lrc_pct: float,
    orc_pct: float,
    alpha: float,
    iterations: int,
    cutoff_lrc: Optional[float],
    cutoff_orc: Optional[float],
    removed_lrc: int,
    removed_orc: int,
    use_ricci_flow: bool,
) -> Dict[str, Any]:
    """Build the preparation metadata consumed by graph evaluation.

    Args:
        graph: Final graph for this parameter configuration.
        prep_time: Actual or amortized preprocessing time in seconds.
        lrc_pct: LRC pruning percentile.
        orc_pct: ORC pruning percentile.
        alpha: ORC idleness parameter.
        iterations: Ricci-flow iteration count.
        cutoff_lrc: LRC pruning threshold.
        cutoff_orc: ORC pruning threshold.
        removed_lrc: Number of edges removed during LRC pruning.
        removed_orc: Number of GCC edges removed during ORC pruning.
        use_ricci_flow: Whether the flow or curvature-only method was used.

    Returns:
        A dictionary with the graph and metadata expected by
        ``evaluate_prepared_graph``.
    """
    return {
        "method_base": (
            "LRC→ORC combo"
            if use_ricci_flow
            else "LRC→ORC combo (No Flow)"
        ),
        "graph": graph,
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
) -> Dict[str, Any]:
    """
    Run LRC pruning on the full graph.

    After LRC pruning:
    - apply Ricci flow + ORC pruning only to the GCC,
    - keep all non-GCC components unchanged,
    - reattach them before community detection.
    """
    start = time.perf_counter()

    graph_lrc = graph.copy()
    compute_LRC(graph_lrc, attribute=lrc_attribute)

    after_lrc, cutoff_lrc, removed_lrc = prune_by_percentile(
        graph_lrc,
        attr=lrc_attribute,
        pct=lrc_pct,
    )

    gcc_graph, non_gcc_graph = _split_lrc_pruned_graph(after_lrc)

    if after_lrc.number_of_nodes() == 0:
        final_graph, cutoff_orc, removed_orc = (
            after_lrc.copy(),
            np.nan,
            0,
        )
    else:
        flow_graph = _run_combo_orc(
            gcc_graph,
            alpha=alpha,
            iterations=iterations,
            method=method,
            weight=weight,
            use_ricci_flow=use_ricci_flow,
        )
        final_graph, cutoff_orc, removed_orc = (
            _prune_and_reassemble_combo_graph(
                flow_graph,
                non_gcc_graph,
                orc_pct=orc_pct,
                orc_attribute=orc_attribute,
            )
        )

    return _combo_preparation_info(
        graph=final_graph,
        prep_time=time.perf_counter() - start,
        lrc_pct=lrc_pct,
        orc_pct=orc_pct,
        alpha=alpha,
        iterations=iterations,
        cutoff_lrc=cutoff_lrc,
        cutoff_orc=cutoff_orc,
        removed_lrc=removed_lrc,
        removed_orc=removed_orc,
        use_ricci_flow=use_ricci_flow,
    )

# ============================================================
# 8. EVALUATE A PREPARED GRAPH
# ============================================================

def evaluate_prepared_graph(
    info: Mapping[str, Any],
    algorithm: str = "louvain",
    seed: int = 42,
    weight: str = "weight",
    label_attribute: str = "label_gt",
) -> Dict[str, Any]:
    """Run Louvain or Leiden on a prepared graph and evaluate it."""
    graph = info["graph"]

    start = time.perf_counter()

    algorithm_lower = algorithm.lower()

    if algorithm_lower == "louvain":
        partition = run_louvain(
            graph,
            weight=weight,
            seed=seed,
        )
        method_name = info["method_base"]

    elif algorithm_lower == "leiden":
        partition = run_leiden(
            graph,
            weight=weight,
            seed=seed,
        )
        method_name = info["method_base"] + " + Leiden"

    else:
        raise ValueError("algorithm must be 'louvain' or 'leiden'.")

    community_time = time.perf_counter() - start

    ari, nmi, n_eval = eval_partition(
        graph,
        partition,
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
            if graph.number_of_nodes() > 0
            else 0
        ),
        "runtime_sec": info["prep_time"] + community_time,
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
    """Evaluate an unmodified graph with Louvain or Leiden."""
    start = time.perf_counter()
    algorithm_lower = algorithm.lower()

    if algorithm_lower == "louvain":
        partition = run_louvain(graph, weight=weight, seed=seed)
        default_name = "Plain Louvain"
    elif algorithm_lower == "leiden":
        partition = run_leiden(graph, weight=weight, seed=seed)
        default_name = "Plain Leiden"
    else:
        raise ValueError("algorithm must be 'louvain' or 'leiden'.")

    runtime = time.perf_counter() - start

    ari, nmi, n_eval = eval_partition(
        graph,
        partition,
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
            if graph.number_of_nodes() > 0
            else 0
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
# 9. STATISTICAL SUMMARIES AND PAIRED TESTS
# ============================================================

def mean_pm_std(
    series: Union[pd.Series, Sequence[float], np.ndarray],
    digits: int = 3,
) -> str:
    """Format a sequence as sample mean ± sample standard deviation."""
    values = pd.Series(series, dtype=float)
    mean = values.mean()
    std = values.std(ddof=1)

    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def summarize_results(
    results: pd.DataFrame,
    method_column: str = "method",
) -> pd.DataFrame:
    """Create the final mean ± std table used in the notebooks."""
    required = {
        method_column,
        "ARI",
        "NMI",
        "runtime_sec",
        "n_eval",
        "V",
        "E",
    }

    missing = required.difference(results.columns)

    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

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
        summary["ARI_mean"].map(lambda value: f"{value:.3f}")
        + " ± "
        + summary["ARI_std"].map(lambda value: f"{value:.3f}")
    )

    summary["NMI"] = (
        summary["NMI_mean"].map(lambda value: f"{value:.3f}")
        + " ± "
        + summary["NMI_std"].map(lambda value: f"{value:.3f}")
    )

    summary["Runtime"] = (
        summary["Runtime_mean"].map(lambda value: f"{value:.2f}")
        + " ± "
        + summary["Runtime_std"].map(lambda value: f"{value:.2f}")
    )

    final_table = summary[
        [
            method_column,
            "ARI",
            "NMI",
            "Runtime",
            "n_eval_mean",
            "V_mean",
            "E_mean",
        ]
    ].copy()

    return final_table.sort_values(
        "ARI_mean" if "ARI_mean" in final_table.columns else method_column
    ) if False else final_table


def paired_tests(
    results: pd.DataFrame,
    baseline_method: str,
    metric: str = "ARI",
) -> pd.DataFrame:
    """Run paired t-tests and Wilcoxon tests against a baseline method."""
    required = {"method", "seed", metric}
    missing = required.difference(results.columns)

    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    baseline_df = (
        results[results["method"] == baseline_method][["seed", metric]]
        .dropna()
        .rename(columns={metric: "baseline_value"})
    )

    rows: List[Dict[str, Any]] = []

    for method in sorted(results["method"].dropna().unique()):
        if method == baseline_method:
            continue

        method_df = (
            results[results["method"] == method][["seed", metric]]
            .dropna()
            .rename(columns={metric: "method_value"})
        )

        paired = baseline_df.merge(method_df, on="seed", how="inner")

        if paired.empty:
            continue

        base = paired["baseline_value"].to_numpy(dtype=float)
        values = paired["method_value"].to_numpy(dtype=float)

        try:
            t_p = ttest_rel(values, base).pvalue
        except Exception:
            t_p = np.nan

        try:
            w_p = wilcoxon(values, base).pvalue
        except Exception:
            w_p = np.nan

        rows.append(
            {
                "baseline": baseline_method,
                "method": method,
                "metric": metric,
                "n_pairs": len(paired),
                "baseline_mean": np.mean(base),
                "method_mean": np.mean(values),
                "mean_diff": np.mean(values - base),
                "paired_t_p": t_p,
                "wilcoxon_p": w_p,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# 10. GRAPH AND GROUND-TRUTH STRUCTURAL ANALYSIS
# ============================================================

def basic_graph_stats(
    graph: nx.Graph,
    dataset_name: Optional[str] = None,
    include_path_metrics: bool = False,
) -> Dict[str, Any]:
    """Compute basic graph statistics."""
    n = graph.number_of_nodes()
    m = graph.number_of_edges()

    degrees = np.array(
        [degree for _, degree in graph.degree()],
        dtype=float,
    )

    gcc = get_gcc(graph)

    stats: Dict[str, Any] = {
        "nodes": n,
        "edges": m,
        "density": nx.density(graph) if n > 0 else np.nan,
        "avg_degree": degrees.mean() if len(degrees) > 0 else np.nan,
        "min_degree": degrees.min() if len(degrees) > 0 else np.nan,
        "max_degree": degrees.max() if len(degrees) > 0 else np.nan,
        "avg_clustering": (
            nx.average_clustering(graph)
            if n > 0
            else np.nan
        ),
        "components": (
            nx.number_connected_components(graph)
            if n > 0
            else 0
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
    """Summarize the ground-truth labels attached to graph nodes."""
    labeled_nodes = [
        node
        for node in graph.nodes()
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
        return (
            {
                "labeled_nodes": 0,
                "num_labels": 0,
                "largest_label_size": np.nan,
                "smallest_label_size": np.nan,
                "mean_label_size": np.nan,
                "median_label_size": np.nan,
                "overlap_fraction": np.nan,
            },
            label_counts,
        )

    return (
        {
            "labeled_nodes": len(labeled_nodes),
            "num_labels": len(set(labels)),
            "largest_label_size": label_counts.max(),
            "smallest_label_size": label_counts.min(),
            "mean_label_size": label_counts.mean(),
            "median_label_size": label_counts.median(),
            "overlap_fraction": (
                np.mean(overlap_flags)
                if overlap_flags
                else np.nan
            ),
        },
        label_counts,
    )


def compute_community_metrics(
    graph: nx.Graph,
    nodes: Iterable[Node],
) -> Optional[Dict[str, Any]]:
    """Compute structural metrics for an explicitly supplied community."""
    community = {node for node in nodes if node in graph}

    if len(community) == 0:
        return None

    internal_edges = 0
    boundary_edges = 0
    boundary_triangles = 0
    internal_triangles = 0

    for u in community:
        neighbors_u = set(graph.neighbors(u))

        for v in neighbors_u:
            if u < v:
                neighbors_v = set(graph.neighbors(v))
                common = len(neighbors_u & neighbors_v)

                if v in community:
                    internal_edges += 1
                    internal_triangles += common
                else:
                    boundary_edges += 1
                    boundary_triangles += common

    volume = 2 * internal_edges + boundary_edges

    conductance = (
        boundary_edges / volume
        if volume > 0
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
    """Compute structural metrics for one ground-truth label."""
    community = {
        node
        for node in graph.nodes()
        if graph.nodes[node].get(label_attribute) == label
    }

    if len(community) == 0:
        return None

    result = compute_community_metrics(graph, community)

    if result is None:
        return None

    possible_internal_edges = (
        len(community) * (len(community) - 1) / 2
    )

    internal_density = (
        result["internal_edges"] / possible_internal_edges
        if possible_internal_edges > 0
        else np.nan
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
    """Compute a weighted mean after removing non-finite rows."""
    valid = dataframe[
        np.isfinite(dataframe[value_col])
        & np.isfinite(dataframe[weight_col])
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
    """Convert node label attributes into a list of node sets."""
    communities_by_label: Dict[Any, Set[Node]] = defaultdict(set)

    for node in graph.nodes():
        if label_attribute in graph.nodes[node]:
            label = graph.nodes[node][label_attribute]
            communities_by_label[label].add(node)

    return list(communities_by_label.values())


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
]
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
    """
    Evaluate the LRC→ORC pipeline over combinations of pruning ratios,
    idleness values, and random seeds.

    Parameters
    ----------
    graph
        Input graph containing ground-truth labels.
    lrc_pcts
        LRC pruning percentages.
    orc_pcts
        ORC pruning percentages.
    alphas
        ORC idleness parameters.
    iterations
        Number of Ricci-flow iterations.
    seeds
        Random seeds used by Louvain or Leiden.
    algorithm
        'louvain' or 'leiden'.
    method
        Optimal-transport solver used by GraphRicciCurvature.
    weight
        Edge-weight attribute.
    label_attribute
        Ground-truth label attribute.
    verbose
        Print progress information.

    Returns
    -------
    pd.DataFrame
        One row per parameter configuration and random seed.

    Preparation time is amortized across configurations that share cached
    LRC and ORC-flow work.
    """

    rows: List[Dict[str, Any]] = []
    total_configs = len(lrc_pcts) * len(orc_pcts) * len(alphas)

    if total_configs == 0:
        return pd.DataFrame(rows)

    def append_row(
        row: Dict[str, Any],
        alpha_index: int,
        lrc_index: int,
        orc_index: int,
        seed_index: int,
    ) -> None:
        """Queue one result row with indices for restoring input order.

        Args:
            row: Result or failure record to append.
            alpha_index: Position of the alpha value in the input sequence.
            lrc_index: Position of the LRC cutoff in the input sequence.
            orc_index: Position of the ORC cutoff in the input sequence.
            seed_index: Position of the seed in the input sequence.

        Returns:
            None. The row is added to the enclosing result list.
        """
        row["_sweep_order"] = (
            alpha_index,
            lrc_index,
            orc_index,
            seed_index,
        )
        rows.append(row)

    def append_preparation_failures(
        lrc_pct: float,
        orc_pct: float,
        alpha: float,
        error: Exception,
        alpha_index: int,
        lrc_index: int,
        orc_index: int,
    ) -> None:
        """Append one failed result per seed for a preparation error.

        Args:
            lrc_pct: LRC cutoff for the failed configuration.
            orc_pct: ORC cutoff for the failed configuration.
            alpha: ORC idleness value for the failed configuration.
            error: Exception raised while preparing the graph.
            alpha_index: Position of alpha in its input sequence.
            lrc_index: Position of the LRC cutoff in its input sequence.
            orc_index: Position of the ORC cutoff in its input sequence.

        Returns:
            None. Failed rows are added to the enclosing result list.
        """
        for seed_index, seed in enumerate(seeds):
            append_row(
                {
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
                },
                alpha_index,
                lrc_index,
                orc_index,
                seed_index,
            )

    lrc_start = time.perf_counter()
    try:
        graph_lrc = graph.copy()
        compute_LRC(graph_lrc)
    except Exception as error:
        if verbose:
            print("LRC preparation failed:", repr(error))
        for alpha_index, alpha in enumerate(alphas):
            for lrc_index, lrc_pct in enumerate(lrc_pcts):
                for orc_index, orc_pct in enumerate(orc_pcts):
                    append_preparation_failures(
                        lrc_pct,
                        orc_pct,
                        alpha,
                        error,
                        alpha_index,
                        lrc_index,
                        orc_index,
                    )
        return pd.DataFrame(rows)
    lrc_compute_time = time.perf_counter() - lrc_start

    config_index = 0

    for lrc_index, lrc_pct in enumerate(lrc_pcts):
        lrc_stage_start = time.perf_counter()
        try:
            after_lrc, cutoff_lrc, removed_lrc = prune_by_percentile(
                graph_lrc,
                attr="LRC",
                pct=lrc_pct,
            )
            gcc_graph, non_gcc_graph = _split_lrc_pruned_graph(after_lrc)
        except Exception as error:
            if verbose:
                print("LRC pruning failed:", repr(error))
            for alpha_index, alpha in enumerate(alphas):
                for orc_index, orc_pct in enumerate(orc_pcts):
                    config_index += 1
                    if verbose:
                        print(
                            f"[{config_index}/{total_configs}] "
                            f"LRC={lrc_pct}%, ORC={orc_pct}%, alpha={alpha}"
                        )
                    append_preparation_failures(
                        lrc_pct,
                        orc_pct,
                        alpha,
                        error,
                        alpha_index,
                        lrc_index,
                        orc_index,
                    )
            continue
        lrc_stage_time = time.perf_counter() - lrc_stage_start

        for alpha_index, alpha in enumerate(alphas):
            flow_start = time.perf_counter()
            try:
                if after_lrc.number_of_nodes() == 0:
                    flow_graph = gcc_graph
                else:
                    flow_graph = _run_combo_orc(
                        gcc_graph,
                        alpha=alpha,
                        iterations=iterations,
                        method=method,
                        weight=weight,
                        use_ricci_flow=True,
                    )
            except Exception as error:
                if verbose:
                    print("ORC preparation failed:", repr(error))
                for orc_index, orc_pct in enumerate(orc_pcts):
                    config_index += 1
                    if verbose:
                        print(
                            f"[{config_index}/{total_configs}] "
                            f"LRC={lrc_pct}%, ORC={orc_pct}%, alpha={alpha}"
                        )
                    append_preparation_failures(
                        lrc_pct,
                        orc_pct,
                        alpha,
                        error,
                        alpha_index,
                        lrc_index,
                        orc_index,
                    )
                continue
            flow_time = time.perf_counter() - flow_start

            for orc_index, orc_pct in enumerate(orc_pcts):
                config_index += 1
                if verbose:
                    print(
                        f"[{config_index}/{total_configs}] "
                        f"LRC={lrc_pct}%, ORC={orc_pct}%, alpha={alpha}"
                    )

                orc_stage_start = time.perf_counter()
                try:
                    final_graph, cutoff_orc, removed_orc = (
                        _prune_and_reassemble_combo_graph(
                            flow_graph,
                            non_gcc_graph,
                            orc_pct=orc_pct,
                            orc_attribute="ricciCurvature",
                        )
                    )
                except Exception as error:
                    if verbose:
                        print("ORC pruning failed:", repr(error))
                    append_preparation_failures(
                        lrc_pct,
                        orc_pct,
                        alpha,
                        error,
                        alpha_index,
                        lrc_index,
                        orc_index,
                    )
                    continue
                orc_stage_time = time.perf_counter() - orc_stage_start

                prep_time = _amortized_combo_prep_time(
                    lrc_compute_time=lrc_compute_time,
                    lrc_stage_time=lrc_stage_time,
                    flow_time=flow_time,
                    orc_stage_time=orc_stage_time,
                    total_configs=total_configs,
                    configs_per_lrc_stage=len(alphas) * len(orc_pcts),
                    orc_variants=len(orc_pcts),
                )
                info = _combo_preparation_info(
                    graph=final_graph,
                    prep_time=prep_time,
                    lrc_pct=lrc_pct,
                    orc_pct=orc_pct,
                    alpha=alpha,
                    iterations=iterations,
                    cutoff_lrc=cutoff_lrc,
                    cutoff_orc=cutoff_orc,
                    removed_lrc=removed_lrc,
                    removed_orc=removed_orc,
                    use_ricci_flow=True,
                )

                for seed_index, seed in enumerate(seeds):
                    try:
                        result = evaluate_prepared_graph(
                            info=info,
                            algorithm=algorithm,
                            seed=seed,
                            weight=weight,
                            label_attribute=label_attribute,
                        )
                        result.update(
                            {
                                "lrc_pct": lrc_pct,
                                "orc_pct": orc_pct,
                                "alpha": alpha,
                                "iterations": iterations,
                                "status": "ok",
                                "error": "",
                            }
                        )
                        append_row(
                            result,
                            alpha_index,
                            lrc_index,
                            orc_index,
                            seed_index,
                        )
                    except Exception as error:
                        append_row(
                            {
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
                                "prep_time": info["prep_time"],
                                "community_time": np.nan,
                                "n_eval": 0,
                                "V": final_graph.number_of_nodes(),
                                "E": final_graph.number_of_edges(),
                                "removed_lrc": removed_lrc,
                                "removed_orc": removed_orc,
                                "status": "failed",
                                "error": repr(error),
                            },
                            alpha_index,
                            lrc_index,
                            orc_index,
                            seed_index,
                        )

    rows.sort(key=lambda row: row["_sweep_order"])
    for row in rows:
        del row["_sweep_order"]

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
    """
    Summarize sensitivity results as mean ± standard deviation.
    """

    required = {
        *group_columns,
        "ARI",
        "NMI",
        "runtime_sec",
        "V",
        "E",
    }

    missing = required.difference(
        results.columns
    )

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
        valid
        .groupby(
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
        [
            "alpha",
            "lrc_pct",
            "orc_pct",
        ]
    ).reset_index(drop=True)


def select_best_sensitivity_configs(
    summary: pd.DataFrame,
    top_n: int = 10,
    metric: str = "ARI_mean",
    ascending: bool = False,
) -> pd.DataFrame:
    """
    Return the best parameter configurations according to a metric.
    """

    if metric not in summary.columns:
        raise KeyError(
            f"{metric!r} is not in the summary table."
        )

    return (
        summary
        .sort_values(
            metric,
            ascending=ascending,
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
    """
    Plot a sensitivity metric from a summarized results table.

    Examples
    --------
    x='lrc_pct', y='ARI_mean', group='orc_pct', error='ARI_std'
    x='alpha', y='Runtime_mean', group=None, error='Runtime_std'
    """

    import matplotlib.pyplot as plt

    required = {x, y}

    if group is not None:
        required.add(group)

    if error is not None:
        required.add(error)

    missing = required.difference(
        summary.columns
    )

    if missing:
        raise KeyError(
            f"Missing columns: {sorted(missing)}"
        )

    fig, ax = plt.subplots(
        figsize=figsize
    )

    if group is None:

        plot_df = summary.sort_values(x)

        ax.errorbar(
            plot_df[x],
            plot_df[y],
            yerr=(
                plot_df[error]
                if error is not None
                else None
            ),
            marker="o",
            capsize=3,
        )

    else:

        for group_value, plot_df in summary.groupby(
            group,
            dropna=False,
        ):

            plot_df = plot_df.sort_values(x)

            ax.errorbar(
                plot_df[x],
                plot_df[y],
                yerr=(
                    plot_df[error]
                    if error is not None
                    else None
                ),
                marker="o",
                capsize=3,
                label=f"{group}={group_value}",
            )

        ax.legend()

    ax.set_title(
        title or f"{y} versus {x}"
    )

    ax.set_xlabel(
        xlabel or x
    )

    ax.set_ylabel(
        ylabel or y
    )

    ax.grid(
        alpha=0.3
    )

    fig.tight_layout()

    return fig, ax 

# ============================================================
# REAL DATASETS: JOINT LRC–ORC SENSITIVITY HEATMAPS
#
# LRC pruning: 0, 5, 10, 20, 30 %
# ORC pruning: 0, 5, 10, 20, 30 %
#
# T = preprocessing + Louvain community detection
# ============================================================

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. SETTINGS
# ============================================================

LRC_HEAT_VALUES = [0, 5, 10, 20, 30]

ORC_HEAT_VALUES = [0, 5, 10, 20, 30]

HEATMAP_ALPHA = 0.5

# Keep small for faster heatmap experiments
LOUVAIN_SEEDS = [0, 1]


# ============================================================
# 2. EMPIRICAL p_in / p_out
# ============================================================

def estimate_empirical_pin_pout(
    graph,
    label_attribute="label_gt"
):

    labeled_nodes = [
        node
        for node in graph.nodes()
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

    # possible within-community pairs
    possible_within = sum(
        n * (n - 1) / 2
        for n in label_counts.values()
    )

    n = len(labeled_nodes)

    possible_total = n * (n - 1) / 2

    possible_between = (
        possible_total
        - possible_within
    )

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
        if possible_within > 0
        else np.nan
    )

    p_out = (
        observed_between / possible_between
        if possible_between > 0
        else np.nan
    )

    return p_in, p_out


# ============================================================
# 3. MAIN EXPERIMENT
# ============================================================

def run_real_dataset_heatmap(
    graph,
    dataset_name,
    label_attribute="label_gt",
    orc_iterations=5,
):
    """Run a heatmap sweep with shared preprocessing time amortized per cell."""

    # --------------------------------------------------------
    # Empirical structure
    # --------------------------------------------------------

    p_in, p_out = estimate_empirical_pin_pout(
        graph,
        label_attribute=label_attribute
    )

    print("\n" + "=" * 70)
    print(dataset_name)
    print("=" * 70)

    print(
        f"empirical p_in  = {p_in:.6f}"
    )

    print(
        f"empirical p_out = {p_out:.6f}"
    )


    rows = []
    total_configs = len(LRC_HEAT_VALUES) * len(ORC_HEAT_VALUES)

    lrc_start = time.perf_counter()
    graph_lrc = graph.copy()
    compute_LRC(graph_lrc)
    lrc_compute_time = time.perf_counter() - lrc_start

    # ========================================================
    # 4. GRID
    # ========================================================

    for lrc_pct in LRC_HEAT_VALUES:
        lrc_stage_start = time.perf_counter()
        try:
            after_lrc, cutoff_lrc, removed_lrc = prune_by_percentile(
                graph_lrc,
                attr="LRC",
                pct=lrc_pct,
            )
            gcc_graph, non_gcc_graph = _split_lrc_pruned_graph(after_lrc)
        except Exception as error:
            for orc_pct in ORC_HEAT_VALUES:
                print(
                    f"\n{dataset_name} | "
                    f"LRC={lrc_pct}% | ORC={orc_pct}%"
                )
                print("FAILED:", repr(error))
            continue
        lrc_stage_time = time.perf_counter() - lrc_stage_start

        flow_start = time.perf_counter()
        try:
            if after_lrc.number_of_nodes() == 0:
                flow_graph = gcc_graph
            else:
                flow_graph = _run_combo_orc(
                    gcc_graph,
                    alpha=HEATMAP_ALPHA,
                    iterations=orc_iterations,
                    method="Sinkhorn",
                    weight="weight",
                    use_ricci_flow=True,
                )
        except Exception as error:
            for orc_pct in ORC_HEAT_VALUES:
                print(
                    f"\n{dataset_name} | "
                    f"LRC={lrc_pct}% | ORC={orc_pct}%"
                )
                print("FAILED:", repr(error))
            continue
        flow_time = time.perf_counter() - flow_start

        for orc_pct in ORC_HEAT_VALUES:
            print(
                f"\n{dataset_name} | "
                f"LRC={lrc_pct}% | "
                f"ORC={orc_pct}%"
            )

            try:
                orc_stage_start = time.perf_counter()
                final_graph, cutoff_orc, removed_orc = (
                    _prune_and_reassemble_combo_graph(
                        flow_graph,
                        non_gcc_graph,
                        orc_pct=orc_pct,
                        orc_attribute="ricciCurvature",
                    )
                )
                orc_stage_time = time.perf_counter() - orc_stage_start

                prep_runtime = _amortized_combo_prep_time(
                    lrc_compute_time=lrc_compute_time,
                    lrc_stage_time=lrc_stage_time,
                    flow_time=flow_time,
                    orc_stage_time=orc_stage_time,
                    total_configs=total_configs,
                    configs_per_lrc_stage=len(ORC_HEAT_VALUES),
                    orc_variants=len(ORC_HEAT_VALUES),
                )
                combo_info = _combo_preparation_info(
                    graph=final_graph,
                    prep_time=prep_runtime,
                    lrc_pct=lrc_pct,
                    orc_pct=orc_pct,
                    alpha=HEATMAP_ALPHA,
                    iterations=orc_iterations,
                    cutoff_lrc=cutoff_lrc,
                    cutoff_orc=cutoff_orc,
                    removed_lrc=removed_lrc,
                    removed_orc=removed_orc,
                    use_ricci_flow=True,
                )

                for seed in LOUVAIN_SEEDS:
                    result = evaluate_prepared_graph(
                        info=combo_info,
                        algorithm="louvain",
                        seed=seed,
                        weight="weight",
                        label_attribute=label_attribute,
                    )
                    community_runtime = result["community_time"]
                    total_runtime = result["runtime_sec"]

                    rows.append(
                        {
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
                        }
                    )

                    print(
                        f"seed={seed} | "
                        f"ARI={result['ARI']:.3f} | "
                        f"NMI={result['NMI']:.3f} | "
                        f"T={total_runtime:.2f}s"
                    )
            except Exception as error:
                print("FAILED:", repr(error))


    # ========================================================
    # 5. DATAFRAME
    # ========================================================

    raw_df = pd.DataFrame(rows)


    summary = (
        raw_df
        .groupby(
            [
                "lrc_pct",
                "orc_pct"
            ],
            as_index=False
        )
        .agg(

            ARI_mean=(
                "ARI",
                "mean"
            ),

            NMI_mean=(
                "NMI",
                "mean"
            ),

            Runtime_mean=(
                "Runtime",
                "mean"
            ),

            Prep_time_mean=(
                "Prep_time",
                "mean"
            ),

            Community_time_mean=(
                "Community_time",
                "mean"
            ),
        )
    )


    # ========================================================
    # 6. MATRICES
    # ========================================================

    def make_matrix(column):

        return (
            summary
            .pivot(
                index="lrc_pct",
                columns="orc_pct",
                values=column
            )
            .reindex(
                index=LRC_HEAT_VALUES,
                columns=ORC_HEAT_VALUES
            )
        )


    ari_matrix = make_matrix(
        "ARI_mean"
    )

    nmi_matrix = make_matrix(
        "NMI_mean"
    )

    runtime_matrix = make_matrix(
        "Runtime_mean"
    )


    # ========================================================
    # 7. HEATMAP FUNCTION
    # ========================================================

    def plot_heatmap(
        metric_matrix,
        metric_name
    ):

        # Same color scales for every dataset

        if metric_name == "ARI":

            vmin = -0.5
            vmax = 1.0

            colorbar_ticks = [
                -0.5,
                1.0
            ]

        else:

            vmin = 0.0
            vmax = 1.0

            colorbar_ticks = [
                0.0,
                1.0
            ]


        fig, ax = plt.subplots(
            figsize=(11, 9)
        )


        im = ax.imshow(
            metric_matrix.values,
            aspect="auto",
            vmin=vmin,
            vmax=vmax
        )


        # ----------------------------------------------------
        # X axis = ORC pruning
        # ----------------------------------------------------

        ax.set_xticks(
            np.arange(
                len(ORC_HEAT_VALUES)
            )
        )

        ax.set_xticklabels(
            [
                f"{x}%"
                for x in ORC_HEAT_VALUES
            ],
            fontsize=12
        )


        # ----------------------------------------------------
        # Y axis = LRC pruning
        # ----------------------------------------------------

        ax.set_yticks(
            np.arange(
                len(LRC_HEAT_VALUES)
            )
        )

        ax.set_yticklabels(
            [
                f"{x}%"
                for x in LRC_HEAT_VALUES
            ],
            fontsize=12
        )


        ax.set_xlabel(
            "ORC pruning ratio (%)",
            fontsize=14
        )

        ax.set_ylabel(
            "LRC pruning ratio (%)",
            fontsize=14
        )


        # ----------------------------------------------------
        # TITLE — same style as your example
        # ----------------------------------------------------

        ax.set_title(
            f"{dataset_name}: "
            f"Joint LRC–ORC Sensitivity ({metric_name})\n"
            rf"empirical "
            rf"$\hat{{p}}_{{in}}={p_in:.5f}$, "
            rf"$\hat{{p}}_{{out}}={p_out:.5f}$",
            fontsize=16
        )


        # ----------------------------------------------------
        # CELL TEXT
        # ----------------------------------------------------

        for i, lrc_pct in enumerate(
            LRC_HEAT_VALUES
        ):

            for j, orc_pct in enumerate(
                ORC_HEAT_VALUES
            ):

                metric_value = (
                    metric_matrix.loc[
                        lrc_pct,
                        orc_pct
                    ]
                )

                runtime_value = (
                    runtime_matrix.loc[
                        lrc_pct,
                        orc_pct
                    ]
                )


                if (
                    pd.isna(metric_value)
                    or pd.isna(runtime_value)
                ):

                    text = "failed"

                else:

                    text = (
                        f"{metric_name}="
                        f"{metric_value:.3f}\n"
                        f"T={runtime_value:.2f}s"
                    )


                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    fontsize=10
                )


        # ----------------------------------------------------
        # COLORBAR
        # ----------------------------------------------------

        cbar = fig.colorbar(
            im,
            ax=ax
        )


        cbar.set_ticks(
            colorbar_ticks
        )


        cbar.set_label(
            f"Mean {metric_name}",
            fontsize=13
        )


        fig.tight_layout()

        plt.show()


    # ========================================================
    # 8. ARI + NMI
    # ========================================================

    plot_heatmap(
        ari_matrix,
        "ARI"
    )


    plot_heatmap(
        nmi_matrix,
        "NMI"
    )


    return (
        raw_df,
        summary
    )   