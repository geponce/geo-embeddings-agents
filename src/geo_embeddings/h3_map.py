"""Stage 4: map the H3 hexagon table built in Databricks SQL.

Reads workspace.geo_embeddings.h3_pinal_r7 through the SQL warehouse and draws
two hexagon maps side by side: the share of grid nodes whose embedding changed
from 2023 to 2024, and the share whose CDL class changed.

Run with:  python -m geo_embeddings.h3_map
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PolyCollection
from shapely import wkt

from geo_embeddings.search import workspace_client

TABLE = "workspace.geo_embeddings.h3_pinal_r7"
FIG = Path("docs/figures/h3_change_r7.png")


def run_query(sql: str) -> pd.DataFrame:
    """Run one SQL statement on the first SQL warehouse and return a DataFrame."""
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


def hex_polygons(wkts: pd.Series) -> list[np.ndarray]:
    """Exterior rings (lon, lat) of the hexagon WKT polygons."""
    return [np.asarray(wkt.loads(s).exterior.coords) for s in wkts]


def main() -> None:
    """Query the hexagon table and save the two-panel map."""
    df = run_query(f"SELECT h3_hex, boundary_wkt, n_nodes, change_mean, share_changed, "
                   f"share_cdl_changed, share_cropland_2024 FROM {TABLE}")
    for c in ("n_nodes", "change_mean", "share_changed", "share_cdl_changed", "share_cropland_2024"):
        df[c] = df[c].astype(float)
    print(f"{len(df):,} hexagons | corr(share_changed, share_cdl_changed) = "
          f"{df.share_changed.corr(df.share_cdl_changed):.3f}")
    print("top 5 hexagons by share of embedding change:")
    print(df.nlargest(5, "share_changed")[["h3_hex", "n_nodes", "share_changed", "share_cdl_changed",
                                           "share_cropland_2024"]].round(3).to_string(index=False))

    polys = hex_polygons(df.boundary_wkt)
    lat0 = np.mean([p[:, 1].mean() for p in polys])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharex=True, sharey=True)
    for ax, col, title in (
        (axes[0], "share_changed", "Embeddings: share of nodes changed (change > 0.14)"),
        (axes[1], "share_cdl_changed", "CDL: share of nodes whose class changed"),
    ):
        pc = PolyCollection(polys, array=df[col].to_numpy(), cmap="magma_r", edgecolor="white",
                            linewidths=0.15, clim=(0, 0.6))
        ax.add_collection(pc)
        ax.autoscale_view()
        ax.set_aspect(1 / np.cos(np.radians(lat0)))
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("longitude")
        ax.set_xticks(np.arange(-112.0, -110.25, 0.5))
        fig.colorbar(pc, ax=ax, fraction=0.035, label="share of nodes in hexagon (capped at 0.6)")
    axes[0].set_ylabel("latitude")
    fig.suptitle("Pinal County 2023 to 2024, H3 resolution 7 hexagons, computed in Databricks SQL",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG, dpi=150)
    print(f"saved {FIG}")


if __name__ == "__main__":
    main()
