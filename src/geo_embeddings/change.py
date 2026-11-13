"""Year-to-year change score from embeddings, validated against CDL label changes.

change = 1 - cosine similarity between a node's embeddings in two years
(0 means identical, larger means more different). Embeddings are unit length,
so the cosine similarity is the dot product.

A CDL label change is not always a real change. Crop rotations and fallowing
are usually real. Flips between natural classes (shrub, grass, barren, woody)
are often CDL relabeling. So we validate the score separately by transition type.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from geo_embeddings.classes import EXCLUDED_ID
from geo_embeddings.embeddings import band_names

CROP = {2, 3, 4, 5, 6, 11, 12}   # Alfalfa, Cotton, Corn+Sorghum, Small grains, Other crops, Double crop, Orchards
FALLOW = 7
NATURAL = {0, 1, 9, 10}          # Shrubland, Grass/Pasture, Barren, Woody
DEVELOPED = 8
NO_CHANGE = "no change"


def pair_years(df: pd.DataFrame, y1: int, y2: int) -> pd.DataFrame:
    """One row per node present in both years, with labels, purity and the change score."""
    keep = ["node_id", "x", "y", "lon", "lat", "label_any", "pure", "cdl_conf"]
    a = df[df.year == y1].set_index("node_id")
    b = df[df.year == y2].set_index("node_id")
    common = a.index.intersection(b.index)
    a, b = a.loc[common], b.loc[common]
    sim = np.einsum("ij,ij->i", a[band_names()].to_numpy(np.float64), b[band_names()].to_numpy(np.float64))
    out = a[[c for c in keep if c not in ("node_id", "label_any", "pure", "cdl_conf")]].copy()
    out["label_1"], out["label_2"] = a.label_any.to_numpy(), b.label_any.to_numpy()
    out["pure_both"] = ((a.pure == 1) & (b.pure == 1)).to_numpy()
    out["conf_1"] = a.cdl_conf.to_numpy()
    out["change"] = 1 - sim
    return out.reset_index()


def transition_type(l1: int, l2: int) -> str:
    """Name the kind of CDL transition between two grouped class ids."""
    if EXCLUDED_ID in (l1, l2):
        return "excluded"
    if l1 == l2:
        return NO_CHANGE
    if DEVELOPED in (l1, l2):
        return "to/from developed"
    if l1 in CROP and l2 in CROP:
        return "crop to crop"
    if {l1, l2} & {FALLOW} and {l1, l2} & CROP:
        return "crop <-> fallow"
    if l1 in NATURAL and l2 in NATURAL:
        return "natural to natural"
    return "farm <-> natural"


def auc_by_type(pairs: pd.DataFrame) -> pd.DataFrame:
    """AUC of the change score for each transition type against unchanged nodes.

    AUC is the chance that a random changed node scores higher than a random
    unchanged node (0.5 is no better than chance).
    """
    t = pairs.assign(ttype=[transition_type(a, b) for a, b in zip(pairs.label_1, pairs.label_2)])
    t = t[t.ttype != "excluded"]
    stable = t[t.ttype == NO_CHANGE].change.to_numpy()
    rows = [{"transition": NO_CHANGE, "n": len(stable), "median_change": float(np.median(stable)), "auc": np.nan}]
    changed = t[t.ttype != NO_CHANGE]
    for name, g in changed.groupby("ttype"):
        y = np.r_[np.zeros(len(stable)), np.ones(len(g))]
        rows.append({"transition": name, "n": len(g), "median_change": float(g.change.median()),
                     "auc": float(roc_auc_score(y, np.r_[stable, g.change.to_numpy()]))})
    y = np.r_[np.zeros(len(stable)), np.ones(len(changed))]
    rows.append({"transition": "any change", "n": len(changed), "median_change": float(changed.change.median()),
                 "auc": float(roc_auc_score(y, np.r_[stable, changed.change.to_numpy()]))})
    return pd.DataFrame(rows)
