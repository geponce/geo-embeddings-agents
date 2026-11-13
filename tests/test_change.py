"""Tests for the change score and transition types."""

import numpy as np
import pandas as pd

from geo_embeddings.change import NO_CHANGE, auc_by_type, pair_years, transition_type
from geo_embeddings.embeddings import band_names


def test_transition_type() -> None:
    assert transition_type(3, 3) == NO_CHANGE
    assert transition_type(3, 5) == "crop to crop"          # cotton -> small grains
    assert transition_type(7, 2) == "crop <-> fallow"       # fallow -> alfalfa
    assert transition_type(0, 1) == "natural to natural"    # shrub -> grass
    assert transition_type(0, 8) == "to/from developed"
    assert transition_type(1, 3) == "farm <-> natural"      # grass -> cotton
    assert transition_type(7, 0) == "farm <-> natural"      # fallow -> shrub
    assert transition_type(-1, 3) == "excluded"


def _grid_two_years() -> pd.DataFrame:
    e = np.zeros((3, 64)); e[:, 0] = 1.0
    e2 = e.copy(); e2[2] = 0; e2[2, 1] = 1.0                 # node c turns orthogonal
    rows = []
    for year, emb in ((2023, e), (2024, e2)):
        d = pd.DataFrame(emb, columns=band_names())
        d["node_id"] = ["a", "b", "c"]; d["year"] = year
        d["x"] = d["y"] = 0.0; d["lon"] = d["lat"] = 0.0
        d["label_any"] = [0, 3, 3] if year == 2023 else [0, 3, 5]
        d["pure"] = 1; d["cdl_conf"] = -1
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


def test_pair_years_change_score() -> None:
    p = pair_years(_grid_two_years(), 2023, 2024).set_index("node_id")
    assert np.isclose(p.loc["a", "change"], 0.0)
    assert np.isclose(p.loc["c", "change"], 1.0)
    assert p.loc["c", "label_2"] == 5 and p.pure_both.all()


def test_auc_by_type_perfect_separation() -> None:
    pairs = pd.DataFrame({"label_1": [0, 0, 3, 3], "label_2": [0, 0, 5, 5], "change": [0.0, 0.1, 0.5, 0.6]})
    out = auc_by_type(pairs).set_index("transition")
    assert out.loc["crop to crop", "auc"] == 1.0
    assert out.loc["any change", "n"] == 2
