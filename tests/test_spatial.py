"""Tests for spatial helpers."""

import numpy as np

from geo_embeddings.spatial import block_ids, similarity_by_distance, to_local_km


def test_to_local_km_one_degree_lat() -> None:
    x, y = to_local_km(np.array([0.0, 0.0]), np.array([33.0, 34.0]), lat0=33.5)
    assert abs((y[1] - y[0]) - 110.574) < 1e-6
    assert x[0] == x[1] == 0.0


def test_to_local_km_lon_shrinks_with_latitude() -> None:
    x, _ = to_local_km(np.array([0.0, 1.0]), np.array([60.0, 60.0]), lat0=60.0)
    assert abs((x[1] - x[0]) - 111.320 * 0.5) < 1e-3


def test_block_ids_group_nearby_points() -> None:
    ids = block_ids(np.array([0.1, 0.9, 1.1, -0.1]), np.array([0.5, 0.5, 0.5, 0.5]), 1.0)
    assert ids[0] == ids[1]
    assert ids[1] != ids[2]
    assert ids[0] != ids[3]  # negative coordinates floor to a different block


def test_similarity_by_distance_counts_each_pair_once() -> None:
    emb = np.eye(3, dtype=np.float32)[[0, 0, 1]]      # points 0,1 identical, 2 orthogonal
    x = np.array([0.0, 1.0, 10.0]); y = np.zeros(3)
    out = similarity_by_distance(emb, x, y, bins_km=[0, 5, 20], chunk=2)
    assert out.n_pairs.tolist() == [1, 2]
    assert out.mean_sim.tolist() == [1.0, 0.0]
