"""Cross-validation and accuracy metrics for land-cover classifiers.

Folds are built from spatial blocks so that whole blocks go to either train or
test. Blocks come from location only, so the same place in different years
always stays in one fold.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from geo_embeddings.spatial import block_ids, to_local_km


def spatial_folds(
    lon: np.ndarray, lat: np.ndarray, y: np.ndarray, block_km: float, n_splits: int = 5, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Assign each row a fold so that every block_km square block sits in one fold.

    Class proportions are balanced across folds as far as the blocks allow.
    Returns (fold index per row, block id per row).
    """
    x_km, y_km = to_local_km(lon, lat)
    blocks = block_ids(x_km, y_km, block_km)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = np.full(len(y), -1, dtype=int)
    for k, (_, test_idx) in enumerate(cv.split(np.zeros(len(y)), y, groups=blocks)):
        folds[test_idx] = k
    return folds, blocks


def random_folds(y: np.ndarray, n_splits: int = 5, seed: int = 42) -> np.ndarray:
    """Assign folds at random (stratified by class), ignoring location."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = np.full(len(y), -1, dtype=int)
    for k, (_, test_idx) in enumerate(cv.split(np.zeros(len(y)), y)):
        folds[test_idx] = k
    return folds


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, labels: list[int]) -> dict[str, Any]:
    """Overall accuracy, Cohen's Kappa, macro F1, per-class F1 and confusion matrix.

    Confusion matrix rows are true classes, columns are predicted, in `labels` order.
    """
    per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {
        "overall_accuracy": float(accuracy_score(y_true, y_pred)),
        "kappa": float(cohen_kappa_score(y_true, y_pred, labels=labels)),
        "macro_f1": float(np.mean(per_class)),
        "f1_per_class": {int(c): float(f) for c, f in zip(labels, per_class)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def _metrics_from_confusion(cm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Overall accuracy, Kappa and per-class F1 from a stack of confusion matrices.

    cm has shape (..., C, C) with rows = true and columns = predicted. Returns
    arrays shaped (...), (...) and (..., C). Undefined values (empty) are NaN.
    """
    cm = cm.astype(float)
    n = cm.sum(axis=(-2, -1))
    diag = np.diagonal(cm, axis1=-2, axis2=-1)
    true_tot, pred_tot = cm.sum(axis=-1), cm.sum(axis=-2)
    with np.errstate(invalid="ignore", divide="ignore"):
        oa = diag.sum(-1) / n
        pe = (true_tot * pred_tot).sum(-1) / n**2
        kappa = (oa - pe) / (1 - pe)
        f1 = 2 * diag / (true_tot + pred_tot)
    return oa, kappa, f1


def block_bootstrap(
    y_true: np.ndarray, y_pred: np.ndarray, blocks: np.ndarray, labels: list[int],
    n_boot: int = 1000, seed: int = 42, ci: float = 0.95,
) -> dict[str, Any]:
    """Confidence intervals for OA, Kappa and per-class F1 by resampling whole blocks.

    Pixels in the same block are not independent, so we resample blocks (with
    replacement) rather than pixels. A class found in only a few blocks gets a
    wide interval, which is the honest answer when few places carry the class.
    Returns point estimates, lower and upper bounds, and the number of blocks
    that contain each class.
    """
    labels = list(labels)
    idx = {c: i for i, c in enumerate(labels)}
    ub, b = np.unique(blocks, return_inverse=True)
    C = len(labels)
    t = np.array([idx[v] for v in y_true])
    p = np.array([idx[v] for v in y_pred])
    per_block = np.zeros((len(ub), C, C))
    np.add.at(per_block, (b, t, p), 1)

    rng = np.random.default_rng(seed)
    weights = rng.multinomial(len(ub), np.full(len(ub), 1 / len(ub)), size=n_boot)  # (n_boot, n_blocks)
    boot_cm = np.einsum("kb,bij->kij", weights, per_block)
    oa, kappa, f1 = _metrics_from_confusion(boot_cm)
    oa0, kappa0, f10 = _metrics_from_confusion(per_block.sum(0))

    a = (1 - ci) / 2 * 100
    q = lambda x: (float(np.nanpercentile(x, a)), float(np.nanpercentile(x, 100 - a)))  # noqa: E731
    blocks_per_class = (per_block.sum(-1) > 0).sum(0)
    return {
        "n_boot": n_boot, "ci": ci, "n_blocks": int(len(ub)),
        "overall_accuracy": {"value": float(oa0), "ci": q(oa)},
        "kappa": {"value": float(kappa0), "ci": q(kappa)},
        "f1_per_class": {int(c): {"value": float(f10[i]), "ci": q(f1[:, i]),
                                  "n_blocks": int(blocks_per_class[i])}
                         for i, c in enumerate(labels)},
    }


def cross_validate(
    make_model: Callable[[], Any], X: np.ndarray, y: np.ndarray, folds: np.ndarray, labels: list[int]
) -> tuple[np.ndarray, pd.DataFrame]:
    """Fit one model per fold and predict the held-out fold.

    Returns out-of-fold predictions for every row and a table of per-fold metrics.
    """
    oof = np.full(len(y), -1, dtype=int)
    rows = []
    for k in np.unique(folds):
        test = folds == k
        model = make_model().fit(X[~test], y[~test])
        oof[test] = model.predict(X[test])
        m = evaluate(y[test], oof[test], labels)
        rows.append({"fold": int(k), "n_test": int(test.sum()),
                     "overall_accuracy": m["overall_accuracy"], "kappa": m["kappa"],
                     "macro_f1": m["macro_f1"]})
    return oof, pd.DataFrame(rows)
