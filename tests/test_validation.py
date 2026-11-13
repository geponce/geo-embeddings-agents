"""Tests for fold assignment and metrics."""

import numpy as np
from sklearn.dummy import DummyClassifier

from geo_embeddings.validation import block_bootstrap, cross_validate, evaluate, random_folds, spatial_folds


def _grid_points(n_side: int = 20, step_deg: float = 0.05, seed: int = 0):
    """Points on a lon/lat grid around Pinal with two classes."""
    rng = np.random.default_rng(seed)
    lon, lat = np.meshgrid(-111.8 + np.arange(n_side) * step_deg, 32.6 + np.arange(n_side) * step_deg)
    lon, lat = lon.ravel(), lat.ravel()
    y = rng.integers(0, 2, size=lon.size)
    return lon, lat, y


def test_spatial_folds_keep_blocks_whole() -> None:
    lon, lat, y = _grid_points()
    folds, blocks = spatial_folds(lon, lat, y, block_km=10, n_splits=5)
    assert set(folds) == {0, 1, 2, 3, 4}
    for b in np.unique(blocks):
        assert len(set(folds[blocks == b])) == 1


def test_random_folds_cover_all_rows() -> None:
    _, _, y = _grid_points()
    folds = random_folds(y, n_splits=4)
    assert (folds >= 0).all()
    assert np.bincount(folds).min() > 0


def test_evaluate_perfect_and_confused() -> None:
    m = evaluate(np.array([0, 0, 1, 1]), np.array([0, 0, 1, 1]), labels=[0, 1])
    assert m["overall_accuracy"] == 1.0 and m["kappa"] == 1.0
    m = evaluate(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]), labels=[0, 1])
    assert m["overall_accuracy"] == 0.75
    assert m["confusion_matrix"] == [[1, 1], [0, 2]]


def test_cross_validate_predicts_every_row() -> None:
    lon, lat, y = _grid_points()
    X = np.column_stack([lon, lat])
    folds, _ = spatial_folds(lon, lat, y, block_km=10, n_splits=3)
    oof, per_fold = cross_validate(lambda: DummyClassifier(strategy="most_frequent"), X, y, folds, [0, 1])
    assert (oof >= 0).all()
    assert len(per_fold) == 3


def test_block_bootstrap_matches_point_metrics_and_brackets_them() -> None:
    rng = np.random.default_rng(1)
    y = rng.integers(0, 3, 600)
    pred = np.where(rng.random(600) < 0.8, y, rng.integers(0, 3, 600))
    blocks = np.repeat(np.arange(60).astype(str), 10)
    out = block_bootstrap(y, pred, blocks, labels=[0, 1, 2], n_boot=300)
    m = evaluate(y, pred, [0, 1, 2])
    assert abs(out["overall_accuracy"]["value"] - m["overall_accuracy"]) < 1e-12
    assert abs(out["kappa"]["value"] - m["kappa"]) < 1e-12
    lo, hi = out["overall_accuracy"]["ci"]
    assert lo < m["overall_accuracy"] < hi
    for c in (0, 1, 2):
        assert abs(out["f1_per_class"][c]["value"] - m["f1_per_class"][c]) < 1e-12
        assert out["f1_per_class"][c]["n_blocks"] == len(np.unique(blocks[y == c]))


def test_block_bootstrap_wider_when_class_in_few_blocks() -> None:
    rng = np.random.default_rng(2)
    blocks = np.repeat(np.arange(50).astype(str), 20)
    y = np.zeros(1000, dtype=int)
    y[:60] = 1                                    # class 1 only in the first 3 blocks
    pred = np.where(rng.random(1000) < 0.85, y, 1 - y)
    out = block_bootstrap(y, pred, blocks, labels=[0, 1], n_boot=500)
    w0 = np.subtract(*out["f1_per_class"][0]["ci"][::-1])
    w1 = np.subtract(*out["f1_per_class"][1]["ci"][::-1])
    assert out["f1_per_class"][1]["n_blocks"] == 3
    assert w1 > 3 * w0
