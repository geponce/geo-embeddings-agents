"""Stage 4: wall-to-wall 200 m grid of embeddings and CDL labels for Pinal County.

Run with:  python -m geo_embeddings.build_grid [--step 200] [--years 2023 2024]

Writes one Parquet file with a row per (year, node). The same node_id appears
once per year, which is what the change score needs.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from geo_embeddings.build_samples import PINAL
from geo_embeddings.classes import CLASS_NAMES
from geo_embeddings.ee_auth import init_ee
from geo_embeddings.embeddings import band_names
from geo_embeddings.grid import build_grid
from geo_embeddings.sampling import county_geometry

OUT_DIR = Path("data/processed")


def main() -> None:
    """Build the grid for each year, check it and write Parquet."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", type=float, default=200, help="grid spacing in meters")
    parser.add_argument("--years", type=int, nargs="+", default=[2023, 2024])
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    init_ee()
    region = county_geometry(*PINAL)
    frames = []
    for year in args.years:
        t0 = time.time()
        print(f"grid {year} ...", flush=True)
        frames.append(build_grid(region, year, args.step, workers=args.workers))
        print(f"  {len(frames[-1]):,} nodes in {time.time() - t0:.0f} s", flush=True)
    df = pd.concat(frames, ignore_index=True)
    df["class_name"] = df.label_any.map(CLASS_NAMES).fillna("Excluded")
    cols = ["node_id", "year", "x", "y", "lon", "lat", "cdl_code", "label_any", "class_name",
            "pure", "cdl_conf"] + band_names()
    df = df[cols].astype({b: "float32" for b in band_names()})  # halves file size, no meaningful loss

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"grid_pinal_{args.step:g}m_{'_'.join(map(str, args.years))}.parquet"
    df.to_parquet(out, index=False)

    norms = np.linalg.norm(df[band_names()].to_numpy(), axis=1)
    per_year = df.groupby("year").node_id.nunique()
    common = set.intersection(*(set(g.node_id) for _, g in df.groupby("year")))
    print(f"\nwrote {out}: {len(df):,} rows | nodes per year {per_year.to_dict()} | in all years {len(common):,}")
    print(f"embedding L2 norm min {norms.min():.4f} median {np.median(norms):.4f} max {norms.max():.4f}")
    print(f"duplicate (year, node_id): {df.duplicated(['year', 'node_id']).sum()}")
    print("\nclass share by year (%):")
    print((pd.crosstab(df.class_name, df.year, normalize="columns") * 100).round(2).to_string())


if __name__ == "__main__":
    main()
