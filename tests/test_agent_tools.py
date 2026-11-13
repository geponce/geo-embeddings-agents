"""Tests for agent tool helpers that need no network."""

import numpy as np
import pandas as pd
import pytest

from geo_embeddings import agent_tools as T


def test_in_study_area() -> None:
    assert T.in_study_area(-111.5, 33.0)
    assert not T.in_study_area(-100.0, 40.0)


def test_haversine_one_degree_latitude() -> None:
    assert abs(T.haversine_km(-111.0, 32.0, -111.0, 33.0) - 111.2) < 0.5


def test_df_to_markdown_handles_missing_and_long_cells() -> None:
    df = pd.DataFrame({"a": [1.0, np.nan], "b": ["x" * 200, None]})
    md = T.df_to_markdown(df, max_cell=20)
    assert md.startswith("| a | b |")
    assert "..." in md and "nan" not in md


def test_class_mix() -> None:
    assert T.class_mix(pd.Series(["Cotton"] * 3 + ["Alfalfa"])) == "Cotton 75%, Alfalfa 25%"


def test_validation_evidence_reads_stored_results() -> None:
    out = T.validation_evidence("classification", "Cotton")
    assert "Cotton" in out and "95%" in out
    assert "crop to crop" in T.validation_evidence("change")
    with pytest.raises(ValueError):
        T.validation_evidence("weather")


def test_run_sql_guard_blocks_before_any_query() -> None:
    called = []
    with pytest.raises(ValueError):
        T.run_sql("DROP TABLE workspace.geo_embeddings.h3_pinal_r7", run_query=lambda s: called.append(s))
    assert called == []
    out = T.run_sql("SELECT 1 AS x FROM workspace.geo_embeddings.h3_pinal_r7",
                    run_query=lambda s: pd.DataFrame({"x": ["1"]}))
    assert "1 rows" in out
