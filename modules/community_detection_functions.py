import networkx as nx
import numpy as np
import community as community_louvain
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from collections import defaultdict

def compute_community_metrics(G, nodes):
    C = set(n for n in nodes if n in G)

    if len(C) == 0:
        return None

    internal_edges = 0
    boundary_edges = 0
    boundary_triangles = 0
    internal_triangles = 0

    for u in C:
        Nu = set(G.neighbors(u))

        for v in Nu:
            if u < v:
                Nv = set(G.neighbors(v))
                common = len(Nu & Nv)

                if v in C:
                    internal_edges += 1
                    internal_triangles += common
                else:
                    boundary_edges += 1
                    boundary_triangles += common

    volume = 2 * internal_edges + boundary_edges

    conductance = boundary_edges / volume if volume > 0 else np.nan
    btd = boundary_triangles / boundary_edges if boundary_edges > 0 else np.nan

    return {
        "size": len(C),
        "internal_edges": internal_edges,
        "boundary_edges": boundary_edges,
        "boundary_triangles": boundary_triangles,
        "internal_triangles": internal_triangles,
        "conductance": conductance,
        "BTD": btd,
    }

def eval_partition(G, part):
    nodes = [
        n for n in G.nodes()
        if "label_gt" in G.nodes[n] and n in part
    ]

    if len(nodes) == 0:
        return np.nan, np.nan, 0

    y_true = [G.nodes[n]["label_gt"] for n in nodes]
    y_pred = [part[n] for n in nodes]

    if len(set(y_true)) < 2:
        return 0.0, 0.0, len(nodes)

    return (
        adjusted_rand_score(y_true, y_pred),
        normalized_mutual_info_score(y_true, y_pred),
        len(nodes)
    )

def run_louvain(G, weight="weight"):
    H_run = G.copy()

    for u, v in H_run.edges():
        if weight not in H_run[u][v]:
            H_run[u][v][weight] = 1.0

    return community_louvain.best_partition(
        H_run,
        weight=weight,
        random_state=RANDOM_SEED
    )

def prune_by_percentile(G, attr, pct):
    if pct == 0:
        return G.copy(), None, 0

    values = np.array(
        [d.get(attr, 0.0) for _, _, d in G.edges(data=True)],
        dtype=float
    )

    values = values[np.isfinite(values)]

    if len(values) == 0:
        return G.copy(), np.nan, 0

    cutoff = float(np.percentile(values, pct))

    G2 = G.copy()

    to_drop = [
        (u, v)
        for u, v, d in G2.edges(data=True)
        if d.get(attr, 0.0) <= cutoff
    ]

    G2.remove_edges_from(to_drop)
    G2.remove_nodes_from(list(nx.isolates(G2)))

    return G2, cutoff, len(to_drop)

def get_gcc(G):
    if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
        return G.copy()

    if nx.is_connected(G):
        return G.copy()

    gcc_nodes = max(nx.connected_components(G), key=len)
    return G.subgraph(gcc_nodes).copy()

## Functions below this line not 100% verified.

def labels_from_partition(partition: dict, n: int) -> np.ndarray:
    """Convert node->cid dict into y_pred aligned to 0..n-1; missing nodes get -1."""
    y = np.full(n, -1, dtype=int)
    for u, c in partition.items():
        y[u] = c
    return y

def exact_community_accuracy(G: nx.Graph, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    % of ground-truth communities exactly recovered:
    count those GT communities for which there exists a predicted community
    with exactly the same member set.
    """
    # Build GT community sets
    gt = defaultdict(set)
    for u, c in enumerate(y_true):
        if c >= 0 and u in G:
            gt[c].add(u)
    # Build predicted community sets
    pr = defaultdict(set)
    for u, c in enumerate(y_pred):
        if c >= 0 and u in G:
            pr[c].add(u)
    gt_sets = [frozenset(s) for s in gt.values() if s]
    pr_sets = set(frozenset(s) for s in pr.values() if s)

    exact = sum(1 for s in gt_sets if s in pr_sets)
    total = len(gt_sets)
    return exact / total if total else 0.0

def evaluate_partition(G: nx.Graph, y_true: np.ndarray, y_pred: np.ndarray):
    nodes = [u for u in G.nodes() if y_true[u] >= 0 and y_pred[u] >= 0]
    if not nodes:
        return {"ARI": None, "NMI": None, "Modularity": None, "ExactComm": None}
    y_t = y_true[nodes]
    y_p = y_pred[nodes]
    ari = adjusted_rand_score(y_t, y_p)
    nmi = normalized_mutual_info_score(y_t, y_p)
    # Modularity on the pruned graph
    part = {u: y_pred[u] for u in nodes}
    Q = community_louvain.modularity(part, G)
    acc = exact_community_accuracy(G, y_true, y_pred)
    return {"ARI": ari, "NMI": nmi, "Modularity": Q, "ExactComm": acc}

# ---------- New: MAIN/ALL evaluation functions (from YouTube notebook) ----------
def get_eval_nodes(G: nx.Graph, y_true: np.ndarray, y_pred: np.ndarray, mode: str = "main"):
    """
    Get nodes for evaluation based on mode.
    - 'main': nodes with unambiguous ground-truth (no overlaps in GT communities)
    - 'all': all nodes with valid labels (projected for overlaps)
    """
    if mode == "main":
        # MAIN: nodes that belong to exactly one ground-truth community
        gt_membership_count = defaultdict(int)
        for u, c in enumerate(y_true):
            if c >= 0 and u in G:
                gt_membership_count[u] += 1
        return [u for u in G.nodes() if gt_membership_count.get(u, 0) == 1 and y_pred[u] >= 0]
    elif mode == "all":
        # ALL: all nodes with valid predicted and ground-truth labels
        return [u for u in G.nodes() if y_true[u] >= 0 and y_pred[u] >= 0]
    else:
        raise ValueError(f"Unknown mode: {mode}")