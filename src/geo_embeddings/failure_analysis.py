"""Stage 3, step 4: where and why the spatial-CV baseline fails.

Uses the out-of-fold predictions from train_baseline (spatial 10 km blocks).

Run with:  python -m geo_embeddings.failure_analysis
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OOF = Path("data/processed/oof_rf_spatial10km.parquet")
FIG_DIR = Path("docs/figures")
CONF_BINS = [0, 50, 70, 90, 99, 100]
CONF_LABELS = ["<50", "50-69", "70-89", "90-98", "99-100"]


def add_error(df: pd.DataFrame) -> pd.DataFrame:
    """Add a boolean 'error' column (prediction differs from the CDL label)."""
    return df.assign(error=df.pred != df.label)


def error_by_confidence(df: pd.DataFrame) -> pd.DataFrame:
    """Error rate and row count by CDL confidence bin, for rows that have confidence.

    Bins are left-closed: <50, 50-69, 70-89, 90-98, 99-100.
    """
    d = df[df.cdl_conf >= 0]
    bins = pd.cut(d.cdl_conf, bins=[-1, 49, 69, 89, 98, 100], labels=CONF_LABELS)
    g = d.groupby(bins, observed=False).error
    return pd.DataFrame({"n": g.size(), "error_rate": g.mean()})


def confused_pairs(df: pd.DataFrame, top: int = 10) -> pd.DataFrame:
    """Most frequent (CDL class -> predicted class) errors, with share of the CDL class."""
    e = df[df.error]
    pairs = e.groupby(["class_name", "pred_name"]).size().rename("n").reset_index()
    pairs["share_of_class"] = pairs.n / pairs.class_name.map(df.class_name.value_counts())
    return pairs.sort_values("n", ascending=False).head(top)


def error_by_code(df: pd.DataFrame, class_name: str, min_n: int = 30) -> pd.DataFrame:
    """Error rate by original CDL code within one grouped class."""
    d = df[df.class_name == class_name]
    g = d.groupby("cdl_code").agg(n=("error", "size"), error_rate=("error", "mean"),
                                  top_wrong=("pred_name", lambda s: s[s != class_name].mode().iat[0]
                                             if (s != class_name).any() else ""))
    return g[g.n >= min_n].sort_values("error_rate", ascending=False)


def block_errors(df: pd.DataFrame) -> pd.DataFrame:
    """Per-block error rate, row count, fold and grid indices (from 'bx_by' ids)."""
    g = df.groupby("block").agg(n=("error", "size"), error_rate=("error", "mean"),
                                fold=("fold", "first"))
    idx = g.index.str.split("_", expand=True)
    g["bx"] = idx.get_level_values(0).astype(int)
    g["by"] = idx.get_level_values(1).astype(int)
    return g


def plot_block_map(blocks: pd.DataFrame, path: Path, block_km: float = 10, min_n: int = 50) -> None:
    """Map of 10 km blocks colored by error rate, with fold numbers.

    Blocks with fewer than min_n pixels are faded, since their error rate is noisy.
    """
    fig, ax = plt.subplots(figsize=(9, 6.5))
    norm = plt.Normalize(0, 0.4)
    cmap = plt.cm.magma_r
    for b in blocks.itertuples():
        ax.add_patch(plt.Rectangle((b.bx * block_km, b.by * block_km), block_km, block_km,
                                   facecolor=cmap(norm(b.error_rate)), edgecolor="white", lw=0.5,
                                   alpha=1.0 if b.n >= min_n else 0.3))
        ax.text((b.bx + 0.5) * block_km, (b.by + 0.5) * block_km, str(b.fold),
                ha="center", va="center", fontsize=6,
                color="white" if b.error_rate > 0.2 else "0.25")
    ax.set_xlim(blocks.bx.min() * block_km, (blocks.bx.max() + 1) * block_km)
    ax.set_ylim(blocks.by.min() * block_km, (blocks.by.max() + 1) * block_km)
    ax.set_aspect("equal")
    ax.set_xlabel("km east (local grid)")
    ax.set_ylabel("km north (local grid)")
    ax.set_title(f"Out-of-fold error rate per 10 km block (number = test fold, faded = under {min_n} pixels)",
                 fontsize=9)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=ax, fraction=0.03, label="error rate (capped at 0.4)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def plot_confidence(conf: pd.DataFrame, overall_2023: float, path: Path) -> None:
    """Error rate by CDL confidence bin, with the 2023 average for reference."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(conf.index.astype(str), conf.error_rate, color="#2b6cb0")
    for i, (n, e) in enumerate(zip(conf.n, conf.error_rate)):
        ax.text(i, e + 0.005, f"{e:.1%}\nn={n:,}", ha="center", va="bottom", fontsize=7)
    ax.axhline(overall_2023, color="0.4", ls="--", lw=1)
    ax.text(len(conf) - 0.5, overall_2023, f"2023 average {overall_2023:.1%}",
            ha="right", va="bottom", fontsize=7, color="0.3")
    ax.set_xlabel("CDL per-pixel confidence (%)")
    ax.set_ylabel("model error rate")
    ax.set_ylim(0, max(conf.error_rate.max() * 1.3, overall_2023 * 1.5))
    ax.set_title("Errors concentrate where the CDL label itself is uncertain (2023)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main() -> None:
    """Print the failure analysis and save two figures."""
    df = add_error(pd.read_parquet(OOF))
    pd.set_option("display.width", 140)
    print(f"overall error rate {df.error.mean():.3f} | by year:",
          df.groupby("year").error.mean().round(3).to_dict())

    d23 = df[df.year == 2023]
    conf = error_by_confidence(d23)
    print("\n1. Error rate by CDL confidence (2023)")
    print(conf.assign(error_rate=conf.error_rate.round(3)).to_string())
    print("\n   per class, error rate for CDL confidence <70 vs >=70 (2023)")
    low = d23.cdl_conf < 70
    per = pd.DataFrame({"n_low": d23[low].groupby("class_name").size(),
                        "err_low": d23[low].groupby("class_name").error.mean(),
                        "err_high": d23[~low].groupby("class_name").error.mean()}).round(3)
    print(per.sort_values("err_low", ascending=False).to_string())

    print("\n2. Most frequent confusions (CDL class -> predicted)")
    print(confused_pairs(df).round(3).to_string(index=False))
    print("\n   Other crops: error rate by original CDL code (n >= 30)")
    print(error_by_code(df, "Other crops").round(3).to_string())

    blocks = block_errors(df)
    print("\n3. Error rate by fold, and the worst blocks (n >= 50)")
    print(df.groupby("fold").agg(n=("error", "size"), error_rate=("error", "mean")).round(3).to_string())
    worst = blocks[blocks.n >= 50].sort_values("error_rate", ascending=False).head(8)
    for b in worst.itertuples():
        inb = df[df.block == b.Index]
        comp = inb.class_name.value_counts().head(3)
        print(f"   block {b.Index:>9} fold {b.fold} n={b.n:>4} error {b.error_rate:.2f}"
              f" | lon {inb.lon.mean():.2f} lat {inb.lat.mean():.2f}"
              f" | main classes: {', '.join(f'{k} {v}' for k, v in comp.items())}")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_block_map(blocks, FIG_DIR / "error_map_blocks.png")
    plot_confidence(conf, d23.error.mean(), FIG_DIR / "error_vs_cdl_confidence.png")
    print(f"\nsaved {FIG_DIR / 'error_map_blocks.png'}, {FIG_DIR / 'error_vs_cdl_confidence.png'}")


if __name__ == "__main__":
    main()
