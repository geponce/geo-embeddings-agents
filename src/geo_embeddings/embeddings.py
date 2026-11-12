"""Read Google Satellite Embedding vectors from Earth Engine.

Dataset: GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL. One image per tile per year,
64 bands A00-A63 at 10 m. Each pixel's 64 values form a unit-length vector.
"""

from __future__ import annotations

import math

import ee

COLLECTION_ID = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
N_BANDS = 64
SCALE_M = 10


def band_names() -> list[str]:
    """Return the 64 embedding band names, A00 through A63."""
    return [f"A{i:02d}" for i in range(N_BANDS)]


def annual_mosaic(year: int) -> ee.Image:
    """Return the embedding image for one calendar year, mosaicked across tiles."""
    return (
        ee.ImageCollection(COLLECTION_ID)
        .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
        .mosaic()
        .select(band_names())
    )


def read_pixel(lon: float, lat: float, year: int) -> list[float]:
    """Return the 64-value embedding at a point for one year.

    Raises ValueError if the point has no data (outside coverage or masked).
    """
    point = ee.Geometry.Point([lon, lat])
    values = (
        annual_mosaic(year)
        .reduceRegion(reducer=ee.Reducer.first(), geometry=point, scale=SCALE_M)
        .getInfo()
    )
    vector = [values.get(b) for b in band_names()]
    if any(v is None for v in vector):
        raise ValueError(f"No embedding at ({lon}, {lat}) for {year}")
    return vector


def l2_norm(vector: list[float]) -> float:
    """Return the Euclidean length of a vector."""
    return math.sqrt(sum(v * v for v in vector))
