"""Tests for failure analysis helpers."""

import pandas as pd

from geo_embeddings.failure_analysis import add_error, block_errors, confused_pairs, error_by_confidence


def _toy() -> pd.DataFrame:
    return add_error(pd.DataFrame({
        "label":      [0, 0, 1, 1, 1, 0],
        "pred":       [0, 1, 1, 0, 1, 0],
        "class_name": ["A", "A", "B", "B", "B", "A"],
        "pred_name":  ["A", "B", "B", "A", "B", "A"],
        "cdl_conf":   [100, 40, 95, 60, -1, 99],
        "block":      ["0_0", "0_0", "1_0", "1_0", "-1_2", "-1_2"],
        "fold":       [0, 0, 1, 1, 2, 2],
    }))


def test_error_by_confidence_skips_missing_and_bins() -> None:
    out = error_by_confidence(_toy())
    assert out.n.sum() == 5                       # the -1 row is excluded
    assert out.loc["<50", "error_rate"] == 1.0    # conf 40 was wrong
    assert out.loc["99-100", "error_rate"] == 0.0


def test_confused_pairs_share() -> None:
    out = confused_pairs(_toy())
    row = out[(out.class_name == "A") & (out.pred_name == "B")].iloc[0]
    assert row.n == 1 and abs(row.share_of_class - 1 / 3) < 1e-9


def test_block_errors_parses_negative_ids() -> None:
    out = block_errors(_toy())
    assert out.loc["-1_2", "bx"] == -1 and out.loc["-1_2", "by"] == 2
    assert out.loc["0_0", "error_rate"] == 0.5
