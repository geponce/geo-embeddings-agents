"""Find places like this one": similarity search over the 200 m embedding grid.

Two paths that should agree:
- exact search in numpy over the local grid (used for evaluation), and
- Databricks Vector Search (Delta Sync index on the grid table, used by apps
  and agents). It ranks by L2 distance. Embeddings are unit length, so L2
  order equals cosine order.

Nearby nodes in the same field are trivially similar, so evaluation counts
only matches farther than a minimum distance from the query.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from geo_embeddings.embeddings import band_names

ENDPOINT = "geo-embeddings-vs"
SOURCE_TABLE = "workspace.geo_embeddings.grid_pinal_200m"
INDEX = "workspace.geo_embeddings.grid_pinal_200m_index"
SYNC_COLUMNS = ["id", "node_id", "year", "lon", "lat", "x", "y", "label_any", "class_name", "pure"]


# ---------- exact search (local) ----------

def nearest_node(grid: pd.DataFrame, lon: float, lat: float, year: int) -> pd.Series:
    """Grid row closest to (lon, lat) in a given year (planar approximation is fine at 200 m)."""
    g = grid[grid.year == year]
    d2 = ((g.lon - lon) * np.cos(np.radians(lat))) ** 2 + (g.lat - lat) ** 2
    return g.loc[d2.idxmin()]


def exact_top_k(emb: np.ndarray, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Indices and cosine similarities of the k most similar rows (descending)."""
    sim = emb @ query
    idx = np.argpartition(-sim, k)[:k]
    idx = idx[np.argsort(-sim[idx])]
    return idx, sim[idx]


def precision_beyond(grid_year: pd.DataFrame, emb: np.ndarray, qi: int, k: int,
                     min_dist_m: float) -> float:
    """Share of the top-k matches farther than min_dist_m from the query that share its class."""
    q = grid_year.iloc[qi]
    far = np.hypot(grid_year.x.to_numpy() - q.x, grid_year.y.to_numpy() - q.y) > min_dist_m
    sim = emb @ emb[qi]
    sim[~far] = -np.inf
    idx = np.argpartition(-sim, k)[:k]
    return float((grid_year.label_any.to_numpy()[idx] == q.label_any).mean())


# ---------- Databricks Vector Search ----------

def workspace_client(profile: str | None = None):
    """WorkspaceClient for the project's CLI profile (OAuth login from the CLI)."""
    import os

    from databricks.sdk import WorkspaceClient

    return WorkspaceClient(profile=profile or os.environ.get("DATABRICKS_CONFIG_PROFILE", "geo-embeddings"))


def ensure_endpoint(w, name: str = ENDPOINT, timeout_min: int = 30) -> str:
    """Create the Standard endpoint if it does not exist and wait until it is ONLINE."""
    from databricks.sdk.service.vectorsearch import EndpointType

    existing = {e.name for e in w.vector_search_endpoints.list_endpoints()}
    if name not in existing:
        w.vector_search_endpoints.create_endpoint(name=name, endpoint_type=EndpointType.STANDARD)
    deadline = time.monotonic() + timeout_min * 60
    while time.monotonic() < deadline:
        state = w.vector_search_endpoints.get_endpoint(name).endpoint_status.state.value
        if state == "ONLINE":
            return state
        if state in ("OFFLINE", "STOPPED"):
            raise RuntimeError(f"endpoint {name} is {state}")
        time.sleep(30)
    return state


def ensure_index(w, endpoint: str = ENDPOINT, index: str = INDEX, source: str = SOURCE_TABLE) -> None:
    """Create a TRIGGERED Delta Sync index with self-managed 64-d vectors, if missing."""
    from databricks.sdk.service.vectorsearch import (
        DeltaSyncVectorIndexSpecRequest, EmbeddingVectorColumn, PipelineType, VectorIndexType)

    existing = {i.name for i in w.vector_search_indexes.list_indexes(endpoint_name=endpoint)}
    if index in existing:
        return
    w.vector_search_indexes.create_index(
        name=index, endpoint_name=endpoint, primary_key="id",
        index_type=VectorIndexType.DELTA_SYNC,
        delta_sync_index_spec=DeltaSyncVectorIndexSpecRequest(
            source_table=source, pipeline_type=PipelineType.TRIGGERED,
            embedding_vector_columns=[EmbeddingVectorColumn(name="embedding", embedding_dimension=64)],
            columns_to_sync=SYNC_COLUMNS))


def index_ready(w, index: str = INDEX) -> tuple[bool, str, int | None]:
    """(ready, status message, indexed row count) for the index."""
    s = w.vector_search_indexes.get_index(index).status
    return bool(s.ready), str(s.message), s.indexed_row_count


def vs_query(w, query_vector: list[float], year: int, k: int = 50,
             columns: list[str] | None = None, index: str = INDEX) -> pd.DataFrame:
    """Top-k nodes from the Vector Search index for one year, with the returned score."""
    cols = columns or ["id", "node_id", "year", "lon", "lat", "x", "y", "label_any", "class_name"]
    r = w.vector_search_indexes.query_index(
        index_name=index, columns=cols, query_vector=[float(v) for v in query_vector],
        num_results=k, filters_json=json.dumps({"year": year}))
    names = [c.name for c in r.manifest.columns]
    return pd.DataFrame(r.result.data_array or [], columns=names)


def grid_embeddings(grid: pd.DataFrame, year: int) -> tuple[pd.DataFrame, np.ndarray]:
    """Rows and embedding matrix (float32) for one year."""
    g = grid[grid.year == year].reset_index(drop=True)
    return g, g[band_names()].to_numpy(np.float32)
