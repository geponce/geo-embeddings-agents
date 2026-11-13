"""Stage 3, step 1: choose the spatial block size for cross-validation.

For each class and year, compute mean embedding similarity between same-class
pixel pairs as a function of distance. Similarity that stays above the far-
distance level means nearby pixels are "too alike" (same field, same farm).
The distance where the curve flattens is the minimum block size. Declines
at tens of km reflect regional differences (geology, elevation, vegetation
type), not field-level leakage, and no block size removes them.

Also counts how many blocks each class falls in at candidate block sizes, since
blocks that are too large leave too few independent units for k-fold CV.

Run with:  python -m geo_embeddings.block_size
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from geo_embeddings.embeddings import band_names
from geo_embeddings.spatial import block_ids, similarity_by_distance, to_local_km

SAMPLES = Path("data/processed/pinal_samples_2023_2024_n2000.parquet")
FIG_DIR = Path("docs/figures")
BINS_KM = [0, 0.1, 0.25, 0.5, 1, 2, 3, 5, 8, 12, 20, 35, 60, 100]
CANDIDATE_SIZES_KM = [1, 2, 5, 8, 10, 15, 20]


def similarity_curves(df: pd.DataFrame) -> pd.DataFrame:
    """Same-class similarity vs distance, averaged over years (pair-weighted)."""
    x, y = to_local_km(df.lon.to_numpy(), df.lat.to_numpy())
    df = df.assign(x=x, y=y)
    rows = []
    for (cls, year), g in df.groupby(["class_name", "year"]):
        out = similarity_by_distance(g[band_names()].to_numpy(), g.x.to_numpy(), g.y.to_numpy(), BINS_KM)
        rows.append(out.assign(class_name=cls, year=year))
    curves = pd.concat(rows)
    curves["w"] = curves.mean_sim * curves.n_pairs
    agg = curves.groupby(["class_name", "lo_km", "hi_km"], as_index=False)[["w", "n_pairs"]].sum()
    agg["mean_sim"] = agg.w / agg.n_pairs
    return agg.drop(columns="w")



def block_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Number of distinct blocks per class (years pooled) at each candidate size."""
    x, y = to_local_km(df.lon.to_numpy(), df.lat.to_numpy())
    out = {}
    for size in CANDIDATE_SIZES_KM:
        ids = block_ids(x, y, size)
        out[f"{size} km"] = pd.Series(ids).groupby(df.class_name.to_numpy()).nunique()
    return pd.DataFrame(out)


def plot_curves(curves: pd.DataFrame, path: Path) -> None:
    """Plot same-class similarity vs distance, one line per class."""
    fig, ax = plt.subplots(figsize=(8, 5))
    groups = list(curves[curves.n_pairs >= 50].groupby("class_name"))
    colors = plt.cm.tab20(np.arange(len(groups)) * 2 % 20 + np.arange(len(groups)) * 2 // 20)
    for (cls, c), color in zip(groups, colors):
        mid = np.sqrt(np.maximum(c.lo_km, 0.05) * c.hi_km)  # geometric bin center
        ax.plot(mid, c.mean_sim, marker="o", ms=3, color=color, label=cls)
    ax.axvline(10, color="0.3", ls="--", lw=1)
    ax.text(10.5, ax.get_ylim()[0] + 0.01, "10 km blocks", fontsize=8, color="0.3")
    ax.set_xscale("log")
    ax.set_xlabel("distance between pixel pairs (km, log scale)")
    ax.set_ylabel("mean cosine similarity (same class)")
    ax.set_title("Same-class embedding similarity vs distance, Pinal County 2023-2024")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)


def main() -> None:
    """Print correlation ranges and block counts, save the similarity figure."""
    df = pd.read_parquet(SAMPLES)
    curves = similarity_curves(df)
    pd.set_option("display.width", 140)
    print("Same-class similarity by distance (rows: class, cols: bin upper edge km)")
    print(curves.pivot(index="class_name", columns="hi_km", values="mean_sim").round(3).to_string())
    print("\nDistinct blocks per class at candidate sizes (2023+2024 pooled)")
    print(block_counts(df).to_string())
    fig = FIG_DIR / "similarity_vs_distance.png"
    plot_curves(curves, fig)
    print(f"\nsaved {fig}")


if __name__ == "__main__":
    main()
