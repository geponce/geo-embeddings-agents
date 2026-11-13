"""Stage 2: sample embeddings and grouped CDL labels for Pinal County, write Parquet.

Run with:  python -m geo_embeddings.build_samples [--n 2000] [--years 2023 2024]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from geo_embeddings.ee_auth import init_ee
from geo_embeddings.embeddings import band_names
from geo_embeddings.sampling import county_geometry, sample_year, to_dataframe

PINAL = ("04", "021")  # Arizona, Pinal County (FIPS)
OUT_DIR = Path("data/processed")


def summarize(df: pd.DataFrame) -> None:
    """Print row counts per class and year and basic quality checks."""
    print(pd.crosstab(df["class_name"], df["year"], margins=True))
    norms = np.linalg.norm(df[band_names()].to_numpy(), axis=1)
    print(f"\nrows {len(df):,} | nulls {int(df.isna().sum().sum())}"
          f" | embedding L2 norm min {norms.min():.4f} median {np.median(norms):.4f}"
          f" max {norms.max():.4f}")
    dup = df.duplicated(subset=["year", "lon", "lat"]).sum()
    print(f"duplicate (year, lon, lat) rows: {dup}")


def main() -> None:
    """Sample each year, combine and write one Parquet file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=2000, help="pixels per class per year")
    parser.add_argument("--years", type=int, nargs="+", default=[2023, 2024])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    init_ee()
    region = county_geometry(*PINAL)
    frames = []
    for year in args.years:
        print(f"sampling {year} ...", flush=True)
        frames.append(to_dataframe(sample_year(region, year, args.n, args.seed)))
    df = pd.concat(frames, ignore_index=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    years = "_".join(str(y) for y in args.years)
    out = OUT_DIR / f"pinal_samples_{years}_n{args.n}.parquet"
    df.to_parquet(out, index=False)
    print(f"wrote {out}\n")
    summarize(df)


if __name__ == "__main__":
    main()
