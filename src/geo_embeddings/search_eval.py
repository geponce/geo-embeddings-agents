"""Stage 4: evaluate "find places like this one" against CDL classes.

For query nodes of each class (pure CDL pixels, 2024), take the top-k most
similar nodes and measure how many share the query's class. Nearby nodes in
the same field are trivially similar, so the honest score only counts matches
farther than 10 km. The naive score (no distance limit) is shown for contrast,
and the class's share of the county is the "random guess" baseline.

Run with:  python -m geo_embeddings.search_eval [--vs]
--vs also compares Databricks Vector Search results with exact search.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from geo_embeddings.classes import CLASS_NAMES
from geo_embeddings.search import exact_top_k, grid_embeddings, precision_beyond

GRID = Path("data/processed/grid_pinal_200m_2023_2024.parquet")
YEAR, K, MIN_DIST_M, N_QUERIES, SEED = 2024, 50, 10_000, 30, 42


def evaluate(g: pd.DataFrame, emb: np.ndarray) -> pd.DataFrame:
    """Per class: mean same-class share of top-K matches, beyond 10 km and with no limit."""
    rng = np.random.default_rng(SEED)
    prevalence = g[g.label_any >= 0].label_any.value_counts(normalize=True)
    rows = []
    for c, name in CLASS_NAMES.items():
        cand = np.flatnonzero((g.label_any.to_numpy() == c) & (g.pure.to_numpy() == 1))
        if len(cand) == 0:
            continue
        qs = rng.choice(cand, size=min(N_QUERIES, len(cand)), replace=False)
        far = [precision_beyond(g, emb, qi, K, MIN_DIST_M) for qi in qs]
        near = [precision_beyond(g, emb, qi, K, -1) for qi in qs]
        rows.append({"class_name": name, "n_queries": len(qs), "share_of_county": float(prevalence.get(c, 0)),
                     "precision_naive": float(np.mean(near)), "precision_beyond_10km": float(np.mean(far))})
    out = pd.DataFrame(rows)
    out["lift_beyond_10km"] = out.precision_beyond_10km / out.share_of_county
    return out.sort_values("precision_beyond_10km", ascending=False)


def plot_example(g: pd.DataFrame, emb: np.ndarray, qi: int, path: Path, k: int = 300) -> pd.Series:
    """Map of one query and its top-k matches beyond 10 km, over all nodes in grey."""
    q = g.iloc[qi]
    far = np.hypot(g.x.to_numpy() - q.x, g.y.to_numpy() - q.y) > MIN_DIST_M
    sim = emb @ emb[qi]
    sim[~far] = -np.inf
    idx = np.argpartition(-sim, k)[:k]
    m = g.iloc[idx]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(g.x / 1000, g.y / 1000, s=0.2, color="0.88", linewidths=0)
    sc = ax.scatter(m.x / 1000, m.y / 1000, s=6, c=sim[idx], cmap="viridis", linewidths=0)
    ax.scatter([q.x / 1000], [q.y / 1000], marker="*", s=250, color="red", edgecolor="k", zorder=5,
               label=f"query: {q.class_name} ({q.lon:.3f}, {q.lat:.3f})")
    ax.add_patch(plt.Circle((q.x / 1000, q.y / 1000), MIN_DIST_M / 1000, fill=False, ls="--", color="red"))
    comp = m.class_name.value_counts(normalize=True)
    ax.set_title(f"Top {k} places like this one, beyond 10 km ({comp.index[0]} {comp.iloc[0]:.0%} of matches)",
                 fontsize=10)
    fig.colorbar(sc, ax=ax, fraction=0.035, label="cosine similarity to query")
    ax.set_aspect("equal")
    ax.set_xlabel("km east (Albers)")
    ax.set_ylabel("km north (Albers)")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return comp


def compare_vector_search(g: pd.DataFrame, emb: np.ndarray, n: int = 20) -> dict:
    """Recall of Databricks Vector Search top-K against exact top-K, and query latency.

    Latency is measured from this laptop, so it includes the network round trip.
    """
    import time

    from geo_embeddings.search import vs_query, workspace_client

    w = workspace_client()
    rng = np.random.default_rng(SEED)
    recalls, secs = [], []
    vs_query(w, emb[0].tolist(), YEAR, k=K)  # warm-up call, not timed
    for qi in rng.choice(len(g), size=n, replace=False):
        idx, _ = exact_top_k(emb, emb[qi], K)
        exact_ids = set(g.node_id.iloc[idx])
        t0 = time.perf_counter()
        res = vs_query(w, emb[qi].tolist(), YEAR, k=K)
        secs.append(time.perf_counter() - t0)
        recalls.append(len(exact_ids & set(res.node_id)) / K)
    return {"n_queries": n, "k": K, "mean_recall": float(np.mean(recalls)), "min_recall": float(np.min(recalls)),
            "latency_ms_median": float(np.median(secs) * 1000), "latency_ms_p90": float(np.quantile(secs, 0.9) * 1000)}


def main() -> None:
    """Run the evaluation, save a results JSON and an example map."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vs", action="store_true", help="also check Databricks Vector Search vs exact search")
    args = parser.parse_args()

    g, emb = grid_embeddings(pd.read_parquet(GRID), YEAR)
    res = evaluate(g, emb)
    pd.set_option("display.width", 140)
    print(f"Top-{K} matches, {N_QUERIES} pure query nodes per class, {YEAR}")
    print(res.round(3).to_string(index=False))
    out = {"year": YEAR, "k": K, "min_dist_m": MIN_DIST_M, "per_class": res.to_dict(orient="records")}

    cotton = np.flatnonzero((g.label_any.to_numpy() == 3) & (g.pure.to_numpy() == 1))
    qi = int(np.random.default_rng(SEED).choice(cotton))
    comp = plot_example(g, emb, qi, Path("docs/figures/search_example_cotton.png"))
    print("\nexample cotton query, class mix of top 300 matches beyond 10 km:")
    print(comp.head(5).round(3).to_string())

    if args.vs:
        out["vector_search_vs_exact"] = compare_vector_search(g, emb)
        print("\nVector Search vs exact:", out["vector_search_vs_exact"])
    Path("docs/results/search_eval_2024.json").write_text(json.dumps(out, indent=2))
    print("\nsaved docs/results/search_eval_2024.json, docs/figures/search_example_cotton.png")


if __name__ == "__main__":
    main()
