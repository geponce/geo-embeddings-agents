"""Spatial helpers for leakage-free validation.

Nearby pixels share fields, soils and management, so their embeddings are
more alike than distant pixels of the same class. If train and test pixels sit
side by side, a model can look accurate by memorizing places. We measure how
far that effect reaches and group pixels into square blocks at least that
large, so whole blocks go to either train or test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

KM_PER_DEG_LAT = 110.574
KM_PER_DEG_LON_EQUATOR = 111.320


def to_local_km(lon: np.ndarray, lat: np.ndarray, lat0: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Project lon/lat to approximate kilometers (equirectangular around lat0).

    Accurate to well under 1% over a county-sized area, which is enough for
    distance binning and block assignment.
    """
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    if lat0 is None:
        lat0 = float(np.mean(lat))
    x = lon * KM_PER_DEG_LON_EQUATOR * np.cos(np.radians(lat0))
    y = lat * KM_PER_DEG_LAT
    return x, y


def block_ids(x_km: np.ndarray, y_km: np.ndarray, size_km: float) -> np.ndarray:
    """Return a string block id for each point on a square grid of size_km."""
    bx = np.floor(np.asarray(x_km) / size_km).astype(int)
    by = np.floor(np.asarray(y_km) / size_km).astype(int)
    return np.char.add(np.char.add(bx.astype(str), "_"), by.astype(str))


def similarity_by_distance(
    emb: np.ndarray,
    x_km: np.ndarray,
    y_km: np.ndarray,
    bins_km: list[float],
    chunk: int = 1000,
) -> pd.DataFrame:
    """Mean cosine similarity of all point pairs, binned by pair distance.

    emb rows should be unit length (the Satellite Embedding is), so the dot
    product is the cosine similarity. Returns one row per distance bin with
    columns lo_km, hi_km, n_pairs, mean_sim.
    """
    emb = np.asarray(emb, dtype=np.float32)
    xy = np.column_stack([x_km, y_km]).astype(np.float64)
    edges = np.asarray(bins_km, dtype=float)
    n_bins = len(edges) - 1
    sums = np.zeros(n_bins)
    counts = np.zeros(n_bins, dtype=np.int64)
    n = len(emb)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        # pairs (i, j) with i in this chunk and j > i, so each pair counts once
        d = np.sqrt(((xy[start:stop, None, :] - xy[None, start:, :]) ** 2).sum(-1))
        s = emb[start:stop] @ emb[start:].T
        upper = np.triu(np.ones((stop - start, n - start), dtype=bool), k=1)
        d, s = d[upper], s[upper]
        idx = np.digitize(d, edges) - 1
        ok = (idx >= 0) & (idx < n_bins)
        sums += np.bincount(idx[ok], weights=s[ok], minlength=n_bins)
        counts += np.bincount(idx[ok], minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = sums / counts
    return pd.DataFrame({"lo_km": edges[:-1], "hi_km": edges[1:],
                         "n_pairs": counts, "mean_sim": mean})
