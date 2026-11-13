"""Stage 4: embedding change score 2023 -> 2024, validated against CDL transitions.

Run with:  python -m geo_embeddings.change_analysis
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from geo_embeddings.change import NO_CHANGE, auc_by_type, pair_years, transition_type

GRID = Path("data/processed/grid_pinal_200m_2023_2024.parquet")
FIG_DIR = Path("docs/figures")
ORDER = [NO_CHANGE, "natural to natural", "farm <-> natural", "crop <-> fallow", "crop to crop",
         "to/from developed"]


def plot_maps(pairs: pd.DataFrame, path: Path) -> None:
    """Side by side: embedding change score, and CDL transition type, per node."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharex=True, sharey=True)
    x, y = pairs.x / 1000, pairs.y / 1000
    sc = axes[0].scatter(x, y, c=pairs.change, s=0.4, cmap="magma_r", vmin=0, vmax=0.5, linewidths=0)
    fig.colorbar(sc, ax=axes[0], fraction=0.035, label="change score, 1 - cosine (capped at 0.5)")
    axes[0].set_title("Embedding change 2023 to 2024", fontsize=10)
    colors = {NO_CHANGE: "#e0e0e0", "natural to natural": "#9ccc65", "farm <-> natural": "#8d6e63",
              "crop <-> fallow": "#ffb300", "crop to crop": "#1e88e5", "to/from developed": "#d81b60",
              "excluded": "#ffffff"}
    for t in ORDER:
        m = pairs.ttype == t
        axes[1].scatter(x[m], y[m], s=0.4, color=colors[t], linewidths=0, label=f"{t} ({m.sum():,})")
    axes[1].set_title("CDL grouped class transition 2023 to 2024", fontsize=10)
    axes[1].legend(markerscale=12, fontsize=7, loc="lower left")
    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlabel("km east (Albers)")
    axes[0].set_ylabel("km north (Albers)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def plot_distributions(pairs: pd.DataFrame, auc: pd.DataFrame, path: Path) -> None:
    """Change score distribution per transition type, with AUC against unchanged nodes."""
    a = auc.set_index("transition")
    data = [pairs.loc[pairs.ttype == t, "change"] for t in ORDER]
    labels = [f"{t}\nn={len(d):,}" + ("" if t == NO_CHANGE else f"\nAUC {a.loc[t, 'auc']:.2f}")
              for t, d in zip(ORDER, data)]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.boxplot(data, showfliers=False, whis=(5, 95))
    ax.set_xticks(range(1, len(ORDER) + 1), labels, fontsize=7)
    ax.set_ylabel("change score (1 - cosine), whiskers 5-95%")
    ax.set_title("Embedding change by CDL transition type, pure nodes, Pinal 2023 to 2024", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main() -> None:
    """Compute the change score, its AUC by transition type, and save figures and JSON."""
    grid = pd.read_parquet(GRID)
    pairs = pair_years(grid, 2023, 2024)
    pairs["ttype"] = [transition_type(a, b) for a, b in zip(pairs.label_1, pairs.label_2)]
    pd.set_option("display.width", 140)

    results = {}
    for name, subset in (("all_nodes", pairs), ("pure_both_years", pairs[pairs.pure_both])):
        auc = auc_by_type(subset)
        results[name] = auc.to_dict(orient="records")
        print(f"\n{name}: {len(subset):,} nodes")
        print(auc.round(3).to_string(index=False))

    pure = pairs[pairs.pure_both]
    stable = pure[pure.ttype == NO_CHANGE]
    hi = stable.change.quantile(0.99)
    print(f"\nstable pure nodes: 99th percentile change {hi:.3f}; top 1% by class:")
    top = stable[stable.change > hi]
    from geo_embeddings.classes import CLASS_NAMES
    print(top.label_1.map(CLASS_NAMES).value_counts().head(6).to_string())

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_maps(pairs, FIG_DIR / "change_map_2023_2024.png")
    plot_distributions(pure, auc_by_type(pure), FIG_DIR / "change_by_transition.png")
    out = Path("docs/results/change_2023_2024.json")
    out.write_text(json.dumps(results, indent=2))
    print(f"\nsaved {out}, docs/figures/change_map_2023_2024.png, docs/figures/change_by_transition.png")


if __name__ == "__main__":
    main()
