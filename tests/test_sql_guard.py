"""Tests for the agent SQL guard."""

import pytest

from geo_embeddings.sql_guard import check_select

T = "workspace.geo_embeddings.h3_pinal_r7"


def test_allows_select_and_wraps_with_limit() -> None:
    out = check_select(f"SELECT h3_hex, share_changed FROM {T} ORDER BY share_changed DESC;")
    assert out.startswith("SELECT * FROM (SELECT h3_hex")
    assert out.endswith("LIMIT 200")


def test_allows_cte() -> None:
    check_select(f"WITH t AS (SELECT * FROM {T}) SELECT count(*) FROM t")


@pytest.mark.parametrize("sql", [
    f"DROP TABLE {T}",
    f"SELECT 1 FROM {T}; DROP TABLE {T}",
    f"DELETE FROM {T}",
    "SELECT * FROM system.access.audit",
    "SELECT * FROM workspace.default.other",
    f"SELECT * FROM read_files('/Volumes/workspace/geo_embeddings/raw/x.parquet')",
    f"SELECT http_request('https://example.com') FROM {T}",
    f"SELECT ai_query('m', 'x') FROM {T}",
    "SELECT 1",
    f"SELECT * FROM `workspace`.`geo_embeddings`.`h3_pinal_r7`",
])
def test_rejects_unsafe(sql: str) -> None:
    with pytest.raises(ValueError):
        check_select(sql)


def test_keywords_inside_strings_are_ignored() -> None:
    check_select(f"SELECT * FROM {T} WHERE main_class_2024 = 'Developed; drop'")
