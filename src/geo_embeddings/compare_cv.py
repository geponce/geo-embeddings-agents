"""Stage 3, step 3: compare random-split and spatial-block CV results.

Reads the metrics JSON written by train_baseline for both CV schemes, prints
the per-class F1 gap next to how clustered each class is (distinct 1 km cells
per 1000 samples), and saves a bar chart.

Run with:  python -m geo_embeddings.compare_cv
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from geo_embeddings.spatial import block_ids, to_local_km

RESULTS = Path("docs/results")
SAMPLES = Path("data/processed/pinal_samples_2023_2024_n2000.parquet")


def load(tag: str) -> dict:
    """Load one results JSON from docs/results."""
    return json.loads((RESULTS / f"{tag}.json").read_text())


def clustering(df: pd.DataFrame) -> pd.Series:
    """Distinct 1 km cells per 1000 samples, by class. Lower means more clustered."""
    x, y = to_local_km(df.lon.to_numpy(), df.lat.to_numpy())
    cells = pd.Series(block_ids(x, y, 1.0), index=df.index)
    return cells.groupby(df.class_name).nunique() / df.class_name.value_counts() * 1000


def main() -> None:
    """Print the comparison table and save the F1 bar chart."""
    rnd, spa = load("rf_random"), load("rf_spatial10km")
    names = {int(k): v for k, v in spa["class_names"].items()}
    t = pd.DataFrame({
        "class_name": [names[int(c)] for c in spa["pooled"]["f1_per_class"]],
        "f1_random": list(rnd["pooled"]["f1_per_class"].values()),
        "f1_spatial": list(spa["pooled"]["f1_per_class"].values()),
    })
    t["gap"] = t.f1_random - t.f1_spatial
    t["cells_per_1000"] = t.class_name.map(clustering(pd.read_parquet(SAMPLES)))
    t = t.sort_values("gap", ascending=False)
    print(t.round(3).to_string(index=False))
    r = np.corrcoef(t.cells_per_1000, t.gap)[0, 1]
    print(f"\ncorrelation between clustering (cells per 1000) and F1 gap: r = {r:.2f}")
    for key in ("overall_accuracy", "kappa", "macro_f1"):
        print(f"{key:<17} random {rnd['pooled'][key]:.3f} | spatial {spa['pooled'][key]:.3f}")

    # Dot plot (not bars) so a non-zero axis does not exaggerate the gaps.
    fig, ax = plt.subplots(figsize=(7, 4.8))
    pos = np.arange(len(t))[::-1]
    ax.hlines(pos, t.f1_spatial, t.f1_random, color="0.75", lw=2, zorder=1)
    ax.scatter(t.f1_random, pos, color="#9e9e9e", zorder=2,
               label=f"random split (OA {rnd['pooled']['overall_accuracy']:.3f})")
    ax.scatter(t.f1_spatial, pos, color="#2b6cb0", zorder=2,
               label=f"10 km spatial blocks (OA {spa['pooled']['overall_accuracy']:.3f})")
    for p, row in zip(pos, t.itertuples()):
        ax.text(row.f1_random + 0.004, p, f"+{row.gap:.3f}", va="center", fontsize=7, color="0.4")
    ax.set_yticks(pos, t.class_name, fontsize=8)
    ax.set_xlim(0.78, 1.01)
    ax.set_xlabel("F1 (out-of-fold)")
    ax.set_title("Random splits overstate F1 most for clustered crop classes", fontsize=10)
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    path = Path("docs/figures/f1_random_vs_spatial.png")
    fig.savefig(path, dpi=150)
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
