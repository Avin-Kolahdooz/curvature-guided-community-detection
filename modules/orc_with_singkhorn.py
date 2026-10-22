def compute_ORC(
    G,
    alpha=0.5,
    method="Sinkhorn",          # try Sinkhorn first
    sinkhorn_epsilon=1.0,
    sinkhorn_max_iter=25,
    verbose="ERROR",
    fallback_to_OT=True,        # auto-fallback to classic OT if needed
):
    """
    Compute Ollivier–Ricci curvature and ensure edges have 'ricciCurvature'.
    Works across GraphRicciCurvature versions by:
      - Returning orc.G (the instance graph) instead of the input G
      - Trying both compute_ricci_curvature() and compute_ricci_flow()
      - Normalizing any curvature-like attribute to 'ricciCurvature'
      - Optionally falling back to method='OT' if Sinkhorn path fails
    """
    from GraphRicciCurvature.OllivierRicci import OllivierRicci

    def _run_once(_G, _method):
        try:
            orc = OllivierRicci(
                _G, alpha=alpha, method=_method,
                sinkhorn_epsilon=sinkhorn_epsilon,
                sinkhorn_max_iter=sinkhorn_max_iter,
                verbose=verbose
            )
        except TypeError:
            orc = OllivierRicci(_G, alpha=alpha, method=_method, verbose=verbose)
            if hasattr(orc, "sinkhorn_epsilon"):
                orc.sinkhorn_epsilon = sinkhorn_epsilon
            if hasattr(orc, "sinkhorn_max_iter"):
                orc.sinkhorn_max_iter = sinkhorn_max_iter

        ok = False
        try:
            orc.compute_ricci_curvature()
            ok = True
        except Exception:
            pass
        try:
            orc.compute_ricci_flow()
            ok = True
        except Exception:
            pass

        # Always use the instance graph (some versions don’t mutate input G)
        H = getattr(orc, "G", _G)

        # Normalize any curvature-like key to 'ricciCurvature'
        candidates = set()
        for _, _, d in H.edges(data=True):
            for k in d.keys():
                if "ricci" in k.lower():
                    candidates.add(k)
        if "ricciCurvature" not in candidates and candidates:
            pick = sorted(candidates)[0]  # deterministic
            for u, v, d in H.edges(data=True):
                if "ricciCurvature" not in d and pick in d:
                    d["ricciCurvature"] = d[pick]

        has_attr = any("ricciCurvature" in d for _, _, d in H.edges(data=True))
        return H, ok and has_attr

    # Try requested method first
    H, success = _run_once(G, method)
    if success:
        return H

    # Fallback: classic OT is slower but reliable
    if fallback_to_OT and method != "OT":
        H2, success2 = _run_once(G, "OT")
        if success2:
            return H2

    # If still not present, raise with helpful context
    attrs = set()
    for _, _, d in H.edges(data=True):
        attrs |= set(d.keys())
    raise RuntimeError(
        "OllivierRicci did not produce a curvature attribute.\n"
        f"Method tried: {method} (fallback_to_OT={fallback_to_OT}).\n"
        f"Edge attributes seen (sample): {sorted(list(attrs))[:12]}"
    )

if __name__ == "__main__":
    G = nx.Graph()
    G.add_edges_from([(1, 2), (1, 3), (2, 3), (2, 4), (3, 4)])
    H = compute_ORC(G)
    for u, v in H.edges():
        print(f"Edge ({u}, {v}) has ricciCurvature: {H[u][v]['ricciCurvature']}")