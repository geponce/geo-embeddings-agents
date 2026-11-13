"""Run SQL on the Databricks SQL warehouse through the Databricks SDK."""

from __future__ import annotations

import pandas as pd


def run_query(sql: str, w=None) -> pd.DataFrame:
    """Run one SQL statement on the first SQL warehouse and return all rows as strings.

    Uses the CLI profile in DATABRICKS_CONFIG_PROFILE (default geo-embeddings).
    """
    if w is None:
        from geo_embeddings.search import workspace_client

        w = workspace_client()
    wh = next(iter(w.warehouses.list())).id
    r = w.statement_execution.execute_statement(warehouse_id=wh, statement=sql, wait_timeout="50s")
    if r.status.state.value != "SUCCEEDED":
        raise RuntimeError(f"query {r.status.state.value}: {r.status.error}")
    cols = [c.name for c in r.manifest.schema.columns]
    rows = list(r.result.data_array or [])
    chunk = r.result.next_chunk_index
    while chunk is not None:  # large results come back in chunks
        part = w.statement_execution.get_statement_result_chunk_n(r.statement_id, chunk)
        rows += part.data_array or []
        chunk = part.next_chunk_index
    return pd.DataFrame(rows, columns=cols)
