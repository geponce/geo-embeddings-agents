"""Tests for pure helpers in geo_embeddings.embeddings (no Earth Engine calls)."""

import math

from geo_embeddings.embeddings import N_BANDS, band_names, l2_norm


def test_band_names_cover_a00_to_a63() -> None:
    names = band_names()
    assert len(names) == N_BANDS
    assert names[0] == "A00"
    assert names[-1] == "A63"
    assert len(set(names)) == N_BANDS


def test_l2_norm() -> None:
    assert l2_norm([3.0, 4.0]) == 5.0
    assert math.isclose(l2_norm([1 / 8] * 64), 1.0)
