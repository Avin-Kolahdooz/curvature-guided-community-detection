import networkx as nx
import numpy as np

def prune_by_percentile(G: nx.Graph, key: str, pct: float):
    """
    Remove the lowest 'pct' percentile edges by attribute 'key'.
    Returns a new induced graph copy.
    """
    vals = []
    for u, v, data in G.edges(data=True):
        if key in data:
            vals.append(data[key])
    if not vals:
        raise ValueError(f"No edge attribute '{key}' found.")
    thresh = np.percentile(vals, pct)
    to_remove = [(u, v) for u, v, d in G.edges(data=True) if d.get(key, 0) <= thresh]
    H = G.copy()
    H.remove_edges_from(to_remove)
    # keep the largest connected component (as in the notebook’s plotting step)
    if not nx.is_empty(H):
        comps = list(nx.connected_components(H))
        H = H.subgraph(max(comps, key=len)).copy()
    return H, thresh

if __name__ == "__main__":
    G = nx.Graph()
    G.add_edges_from([(1, 2), (1, 3), (2, 3), (2, 4), (3, 4)])
    # Assign some dummy LRC values for testing
    for u, v in G.edges():
        G[u][v]["LRC"] = np.random.rand()
    H, thresh = prune_by_percentile(G, "LRC", 50)
    print(f"Threshold for pruning: {thresh}")
    print(f"Edges after pruning: {list(H.edges(data=True))}")