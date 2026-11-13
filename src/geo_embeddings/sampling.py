"""Stratified sampling of satellite embeddings with grouped CDL labels in Earth Engine.

Label quality matters more than sample count, so we only sample "pure" pixels.
The CDL is 30 m and embeddings are 10 m, so a 10 m pixel near a field edge can
carry the wrong label. A pixel is pure when its 3x3 neighborhood on the CDL
30 m grid (a 90 m block) has a single grouped class.
"""

from __future__ import annotations

import ee
import pandas as pd

from geo_embeddings.classes import CLASS_NAMES, EXCLUDED_ID, remap_lists
from geo_embeddings.embeddings import SCALE_M, annual_mosaic, band_names

CDL_COLLECTION = "USDA/NASS/CDL"
COUNTIES = "TIGER/2018/Counties"
CDL_SCALE_M = 30
NO_CONFIDENCE = -1  # value of cdl_conf for years without a CDL confidence band


def county_geometry(statefp: str, countyfp: str) -> ee.Geometry:
    """Return a county boundary from TIGER 2018 by state and county FIPS codes."""
    fc = ee.FeatureCollection(COUNTIES).filter(
        ee.Filter.And(ee.Filter.eq("STATEFP", statefp), ee.Filter.eq("COUNTYFP", countyfp))
    )
    return fc.geometry()


def cdl_layers(year: int) -> ee.Image:
    """Return unmasked CDL layers for one year on the CDL 30 m grid.

    Bands: cdl_code (raw CDL code), label_any (grouped class id, or EXCLUDED_ID),
    pure (1 where the 3x3 CDL neighborhood has one grouped class) and cdl_conf
    (CDL confidence, or NO_CONFIDENCE for years without that band).
    """
    cdl = ee.Image(f"{CDL_COLLECTION}/{year}")
    code = cdl.select("cropland")
    from_codes, to_ids = remap_lists()
    grouped = code.remap(from_codes, to_ids).unmask(EXCLUDED_ID)
    n_distinct = grouped.reduceNeighborhood(
        reducer=ee.Reducer.countDistinct(), kernel=ee.Kernel.square(1, "pixels")
    ).reproject(code.projection())
    conf = ee.Image(
        ee.Algorithms.If(
            cdl.bandNames().contains("confidence"),
            cdl.select("confidence"),
            ee.Image.constant(NO_CONFIDENCE),
        )
    )
    return ee.Image.cat(
        code.rename("cdl_code").toInt(),
        grouped.rename("label_any").toInt(),
        n_distinct.eq(1).rename("pure").toInt(),
        conf.rename("cdl_conf").toInt(),
    )


def label_image(year: int) -> ee.Image:
    """Return bands label, cdl_code and cdl_conf for one year.

    label is the grouped class id, masked where the class is excluded or the
    3x3 CDL neighborhood is mixed. cdl_conf is NO_CONFIDENCE when the CDL
    year has no confidence band (only some years include it).
    """
    layers = cdl_layers(year)
    grouped = layers.select("label_any")
    label = grouped.updateMask(layers.select("pure")).updateMask(grouped.neq(EXCLUDED_ID))
    return ee.Image.cat(label.rename("label"), layers.select(["cdl_code", "cdl_conf"]))


def sample_year(
    region: ee.Geometry, year: int, n_per_class: int, seed: int = 42
) -> ee.FeatureCollection:
    """Return a stratified sample of pure pixels for one year.

    Two steps keep this fast. First the stratified sample runs on the labels
    alone at the CDL's native 30 m grid, which picks the points. Then the 64
    embedding bands are read only at those points (the 10 m pixel at the
    center of each 30 m CDL pixel). Sampling all 64 bands at 10 m over a
    whole county times out.

    Each feature has the 64 embedding bands, label, cdl_code, cdl_conf,
    longitude, latitude (CDL pixel center) and year.
    """
    labels = label_image(year)
    points = labels.addBands(ee.Image.pixelLonLat()).stratifiedSample(
        numPoints=n_per_class,
        classBand="label",
        region=region,
        scale=CDL_SCALE_M,
        projection=labels.select("cdl_code").projection(),
        seed=seed,
        geometries=True,
        tileScale=16,
    )
    return annual_mosaic(year).sampleRegions(
        collection=points, scale=SCALE_M, geometries=False
    ).map(lambda f: f.set("year", year))


def to_dataframe(fc: ee.FeatureCollection) -> pd.DataFrame:
    """Download a FeatureCollection as a tidy DataFrame with a fixed column order."""
    df = ee.data.computeFeatures({"expression": fc, "fileFormat": "PANDAS_DATAFRAME"})
    df = df.rename(columns={"longitude": "lon", "latitude": "lat"})
    df["class_name"] = df["label"].map(CLASS_NAMES)
    meta = ["year", "label", "class_name", "cdl_code", "cdl_conf", "lon", "lat"]
    return df[meta + band_names()].astype({"year": "int16", "label": "int8",
                                           "cdl_code": "int16", "cdl_conf": "int16"})
