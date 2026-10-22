import networkx as nx

def compute_LRC(G):
    for u, v in G.edges():
        Nu = set(G.neighbors(u))
        Nv = set(G.neighbors(v))

        ni = len(Nu)
        nj = len(Nv)
        nij = len(Nu & Nv)

        if ni == 0 or nj == 0:
            lrc = -np.inf
        else:
            lrc = (
                2.0 / ni
                + 2.0 / nj
                - 2.0
                + 2.0 * nij / max(ni, nj)
                + nij / min(ni, nj)
            )

        G[u][v]["LRC"] = float(lrc)

    return G

if __name__ == "__main__":
    G = nx.Graph()
    G.add_edges_from([(1, 2), (1, 3), (2, 3), (2, 4), (3, 4)])
    compute_LRC(G)
    for u, v in G.edges():
        print(f"Edge ({u}, {v}) has LRC: {G[u][v]['LRC']}")