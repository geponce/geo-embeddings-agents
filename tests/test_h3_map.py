"""Tests for H3 map helpers that do not need a workspace."""

import pandas as pd

from geo_embeddings.h3_map import hex_polygons


def test_hex_polygons_parse_wkt() -> None:
    w = "POLYGON ((-111 33, -110.9 33, -110.9 33.1, -111 33.1, -111 33))"
    rings = hex_polygons(pd.Series([w]))
    assert rings[0].shape == (5, 2)
    assert rings[0][0].tolist() == [-111.0, 33.0]
