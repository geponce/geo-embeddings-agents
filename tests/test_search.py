"""Tests for exact similarity search helpers."""

import numpy as np
import pandas as pd

from geo_embeddings.search import exact_top_k, nearest_node, precision_beyond


def test_exact_top_k_orders_by_similarity() -> None:
    emb = np.eye(4, dtype=np.float32)
    emb[1] = [0.8, 0.6, 0, 0]
    idx, sim = exact_top_k(emb, emb[0], k=2)
    assert idx.tolist() == [0, 1]
    assert np.isclose(sim[1], 0.8)


def test_precision_beyond_ignores_nearby_matches() -> None:
    # query at x=0; an identical same-class node 100 m away is excluded by min_dist
    g = pd.DataFrame({"x": [0, 100, 50_000, 60_000], "y": [0, 0, 0, 0], "label_any": [3, 3, 3, 5]})
    emb = np.array([[1, 0], [1, 0], [0.9, 0.436], [0.95, 0.312]], dtype=np.float32)
    assert precision_beyond(g, emb, qi=0, k=1, min_dist_m=10_000) == 0.0   # nearest far match is class 5
    assert precision_beyond(g, emb, qi=0, k=2, min_dist_m=10_000) == 0.5


def test_nearest_node() -> None:
    g = pd.DataFrame({"year": [2024, 2024, 2023], "lon": [-111.0, -111.5, -111.0], "lat": [33.0, 33.0, 33.0]})
    assert nearest_node(g, -111.45, 33.0, 2024).lon == -111.5
