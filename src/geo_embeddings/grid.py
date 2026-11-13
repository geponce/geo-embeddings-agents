"""Wall-to-wall grid of embeddings and CDL labels over a county, for search and change.

The grid is regular in CONUS Albers (EPSG:5070, meters), the CDL's own
projection. Each node takes the single 10 m embedding pixel at the node.
We force evaluation at 10 m, because asking Earth Engine for a coarse grid
otherwise reads from averaged pyramid levels, and an averaged embedding is
not the embedding of any real place (its length drops below 1).

Pixels come back as arrays from ee.data.computePixels, tile by tile, with no
export or Cloud Storage needed.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import ee
import numpy as np
import pandas as pd
from pyproj import Transformer

from geo_embeddings.embeddings import band_names, annual_mosaic
from geo_embeddings.sampling import cdl_layers

ALBERS = "EPSG:5070"
NODATA = -9999
LABEL_BANDS = ["cdl_code", "label_any", "pure", "cdl_conf"]


def snapped_bounds(region: ee.Geometry, step_m: float) -> tuple[float, float, float, float]:
    """Region bounds in Albers meters, snapped outward to multiples of step_m."""
    coords = np.array(region.bounds(1, ALBERS).coordinates().getInfo()[0])
    xmin, ymin = coords.min(axis=0)
    xmax, ymax = coords.max(axis=0)
    snap = lambda v, f: f(v / step_m) * step_m  # noqa: E731
    return snap(xmin, np.floor), snap(ymin, np.floor), snap(xmax, np.ceil), snap(ymax, np.ceil)


def tile_origins(bounds: tuple[float, float, float, float], step_m: float, tile_px: int) -> list[tuple[float, float, int, int]]:
    """Top-left corners and sizes (x0, y_top, width_px, height_px) of tiles covering bounds."""
    xmin, ymin, xmax, ymax = bounds
    nx, ny = int(round((xmax - xmin) / step_m)), int(round((ymax - ymin) / step_m))
    tiles = []
    for j in range(0, ny, tile_px):
        for i in range(0, nx, tile_px):
            tiles.append((xmin + i * step_m, ymax - j * step_m, min(tile_px, nx - i), min(tile_px, ny - j)))
    return tiles


def node_image(year: int, region: ee.Geometry) -> ee.Image:
    """Embedding (forced to 10 m) plus CDL layers and an 'inside' county flag."""
    emb = annual_mosaic(year).reproject(crs=ALBERS, scale=10)
    inside = ee.Image.constant(0).paint(region, 1).rename("inside").toInt()
    return (emb.addBands(cdl_layers(year)).addBands(inside)
            .unmask(NODATA, sameFootprint=False))


def fetch_tile(image: ee.Image, x0: float, y_top: float, w: int, h: int, step_m: float) -> pd.DataFrame:
    """Pull one tile of nodes as a DataFrame with Albers node-center coordinates."""
    arr = ee.data.computePixels({
        "expression": image,
        "fileFormat": "NUMPY_NDARRAY",
        "grid": {"dimensions": {"width": w, "height": h},
                 "affineTransform": {"scaleX": step_m, "shearX": 0, "translateX": x0,
                                     "shearY": 0, "scaleY": -step_m, "translateY": y_top},
                 "crsCode": ALBERS},
    })
    df = pd.DataFrame({name: arr[name].ravel() for name in arr.dtype.names})
    jj, ii = np.mgrid[0:h, 0:w]
    df["x"] = x0 + (ii.ravel() + 0.5) * step_m
    df["y"] = y_top - (jj.ravel() + 0.5) * step_m
    return df


def clean_nodes(df: pd.DataFrame) -> pd.DataFrame:
    """Keep nodes inside the region that have a full embedding; add lon/lat and node_id."""
    emb_ok = (df[band_names()] != NODATA).all(axis=1)
    df = df[(df["inside"] == 1) & emb_ok].drop(columns="inside").copy()
    lon, lat = Transformer.from_crs(ALBERS, "EPSG:4326", always_xy=True).transform(df.x.to_numpy(), df.y.to_numpy())
    df["lon"], df["lat"] = lon, lat
    df["node_id"] = (df.x.round().astype(np.int64).astype(str) + "_" + df.y.round().astype(np.int64).astype(str))
    return df


def build_grid(region: ee.Geometry, year: int, step_m: float = 200, tile_px: int = 64,
               workers: int = 8) -> pd.DataFrame:
    """All grid nodes for one year, fetched tile by tile in parallel."""
    image = node_image(year, region).select(band_names() + LABEL_BANDS + ["inside"])
    tiles = tile_origins(snapped_bounds(region, step_m), step_m, tile_px)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        frames = list(pool.map(lambda t: fetch_tile(image, *t, step_m), tiles))
    df = clean_nodes(pd.concat(frames, ignore_index=True))
    df.insert(0, "year", year)
    return df.astype({"year": "int16", "cdl_code": "int16", "label_any": "int8", "pure": "int8",
                      "cdl_conf": "int16"})
