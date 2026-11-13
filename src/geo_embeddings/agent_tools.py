"""Read-only functions the Claude agents call as tools.

Each function returns plain text (markdown) that an agent can read, plus the
path of any map it saved. Nothing here writes to Databricks or Earth Engine.
Maps go to data/outputs/ (gitignored).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from geo_embeddings.classes import CLASS_NAMES

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "docs" / "results"
OUTPUTS = REPO / "data" / "outputs"
GRID_FILE = REPO / "data" / "processed" / "grid_pinal_200m_2023_2024.parquet"
GRID_TABLE = "workspace.geo_embeddings.grid_pinal_200m"
H3_TABLE = "workspace.geo_embeddings.h3_pinal_r7"
PINAL_BBOX = (-112.21, 32.50, -110.44, 33.47)  # lon_min, lat_min, lon_max, lat_max
YEARS_IN_TABLES = (2023, 2024)
CHANGE_THRESHOLD = 0.14

DATA_GUIDE = f"""Study area: Pinal County, Arizona (lon {PINAL_BBOX[0]} to {PINAL_BBOX[2]}, lat {PINAL_BBOX[1]} to {PINAL_BBOX[3]}).
Years in the Databricks tables: 2023 and 2024. Earth Engine tools can reach 2017 to 2025.

Tables (Databricks Unity Catalog, read-only):
- {GRID_TABLE}: 200 m grid, one row per (node_id, year). Columns: id, node_id, year, x, y (Albers m), lon, lat,
  cdl_code, label_any (class id, -1 excluded), class_name, pure (1 if the 90 m CDL block is one class), cdl_conf
  (2023 only, else -1), embedding (64 floats, unit length). 348,004 nodes per year.
- {H3_TABLE}: H3 resolution 7 hexagons (about 5.2 km2). Columns: h3, h3_hex, boundary_wkt, n_nodes, change_mean,
  change_p90, share_changed (embedding change > {CHANGE_THRESHOLD}), share_cdl_changed, share_cropland_2024,
  main_class_2024. 2,455 rows.
- workspace.geo_embeddings.samples_pinal: 52,000 training pixels (2,000 per class per year, pure pixels only).

Classes (CDL grouped): {", ".join(f"{k} {v}" for k, v in CLASS_NAMES.items())}.
Change score = 1 - cosine similarity of a place's 2023 and 2024 embeddings (0 = same, about 0.4 = typical crop change).
"""


# ---------- helpers ----------

def in_study_area(lon: float, lat: float) -> bool:
    """True if the point falls inside the Pinal County bounding box."""
    return PINAL_BBOX[0] <= lon <= PINAL_BBOX[2] and PINAL_BBOX[1] <= lat <= PINAL_BBOX[3]


def haversine_km(lon1, lat1, lon2, lat2):
    """Great-circle distance in km (works on scalars or numpy arrays)."""
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def df_to_markdown(df: pd.DataFrame, max_cell: int = 80) -> str:
    """Small markdown table, truncating long cells (such as embedding arrays)."""
    if df.empty:
        return "(no rows)"
    def cell(v) -> str:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return ""
        s = str(v)
        return s if len(s) <= max_cell else s[: max_cell - 3] + "..."

    d = df.map(cell)
    head = "| " + " | ".join(d.columns) + " |\n|" + "---|" * len(d.columns) + "\n"
    return head + "\n".join("| " + " | ".join(r) + " |" for r in d.itertuples(index=False))


def class_mix(labels: pd.Series) -> str:
    """'Cotton 82%, Alfalfa 9%, ...' for the top classes in a label series."""
    s = labels.value_counts(normalize=True).head(5)
    return ", ".join(f"{k} {v:.0%}" for k, v in s.items())


def _backdrop(ax) -> None:
    """Light grey county outline from every 7th grid node, if the local grid file exists."""
    if GRID_FILE.exists():
        g = pd.read_parquet(GRID_FILE, columns=["year", "lon", "lat"])
        g = g[g.year == 2024].iloc[::7]
        ax.scatter(g.lon, g.lat, s=0.3, color="0.85", linewidths=0)


def _save(fig, name: str) -> str:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    path = OUTPUTS / name
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return str(path.relative_to(REPO))


# ---------- Databricks tools ----------

def run_sql(sql: str, run_query=None) -> str:
    """Run a guarded read-only SELECT and return a markdown table (at most 200 rows)."""
    from geo_embeddings.sql_guard import check_select

    safe = check_select(sql)
    if run_query is None:
        from geo_embeddings.dbsql import run_query
    df = run_query(safe)
    note = " (limit reached, refine the query)" if len(df) >= 200 else ""
    return f"{len(df)} rows{note}\n\n{df_to_markdown(df)}"


def _nearest_node_sql(lon: float, lat: float, year: int, extra: str = "") -> str:
    k = math.cos(math.radians(lat))
    return (f"SELECT node_id, year, lon, lat, class_name, label_any, pure, cdl_conf{extra} FROM {GRID_TABLE} "
            f"WHERE year = {int(year)} AND lon BETWEEN {lon - 0.01} AND {lon + 0.01} "
            f"AND lat BETWEEN {lat - 0.01} AND {lat + 0.01} "
            f"ORDER BY pow((lon - ({lon})) * {k}, 2) + pow(lat - ({lat}), 2) LIMIT 1")


def place_profile(lon: float, lat: float) -> str:
    """CDL class in 2023 and 2024, embedding change and H3 hexagon context for the nearest grid node."""
    from geo_embeddings.dbsql import run_query

    if not in_study_area(lon, lat):
        raise ValueError(f"({lon}, {lat}) is outside Pinal County, the study area {PINAL_BBOX}.")
    rows = {y: run_query(_nearest_node_sql(lon, lat, y, ", embedding")) for y in YEARS_IN_TABLES}
    if any(r.empty for r in rows.values()):
        raise ValueError("No grid node within about 1 km of that point.")
    a, b = rows[2023].iloc[0], rows[2024].iloc[0]
    ea = np.asarray(json.loads(a.embedding), dtype=float)  # SQL returns array elements as strings
    eb = np.asarray(json.loads(b.embedding), dtype=float)
    change = 1 - float(ea @ eb)
    hexrow = run_query(f"SELECT h3_hex, n_nodes, round(change_mean,3) AS change_mean, round(share_changed,3) AS "
                       f"share_changed, round(share_cdl_changed,3) AS share_cdl_changed, "
                       f"round(share_cropland_2024,3) AS share_cropland_2024, main_class_2024 FROM {H3_TABLE} "
                       f"WHERE h3 = h3_longlatash3({float(a.lon)}, {float(a.lat)}, 7)")
    dist = float(haversine_km(lon, lat, float(a.lon), float(a.lat)))
    verdict = "changed" if change > CHANGE_THRESHOLD else "stable"
    lines = [
        f"Nearest 200 m grid node {a.node_id} at ({float(a.lon):.5f}, {float(a.lat):.5f}), {dist * 1000:.0f} m from the query.",
        f"CDL class 2023: {a.class_name} (pure block: {a.pure == '1'}, CDL confidence {a.cdl_conf}).",
        f"CDL class 2024: {b.class_name} (pure block: {b.pure == '1'}).",
        f"Embedding change 2023 to 2024: {change:.3f} ({verdict}; threshold {CHANGE_THRESHOLD}, crop changes are typically about 0.4).",
    ]
    if not hexrow.empty:
        lines.append("H3 hexagon around it:\n" + df_to_markdown(hexrow))
    return "\n".join(lines)


def find_similar(lon: float, lat: float, year: int = 2024, k: int = 50, min_distance_km: float = 10.0) -> tuple[str, str]:
    """Most similar places to the nearest grid node, beyond a minimum distance, via Vector Search.

    Returns a text summary and the path of a saved map.
    """
    from geo_embeddings.dbsql import run_query
    from geo_embeddings.search import vs_query, workspace_client

    if not in_study_area(lon, lat):
        raise ValueError(f"({lon}, {lat}) is outside Pinal County, the study area {PINAL_BBOX}.")
    if year not in YEARS_IN_TABLES:
        raise ValueError("The search index covers 2023 and 2024 only.")
    q = run_query(_nearest_node_sql(lon, lat, year, ", embedding"))
    if q.empty:
        raise ValueError("No grid node within about 1 km of that point.")
    q = q.iloc[0]
    emb = [float(v) for v in json.loads(q.embedding)]
    res = vs_query(workspace_client(), emb, year, k=max(400, 4 * k))
    res[["lon", "lat"]] = res[["lon", "lat"]].astype(float)
    score_col = [c for c in res.columns if c not in ("id", "node_id", "year", "lon", "lat", "x", "y", "label_any", "class_name")][-1]
    res["dist_km"] = haversine_km(float(q.lon), float(q.lat), res.lon, res.lat)
    far = res[res.dist_km > min_distance_km].head(k)
    if far.empty:
        return (f"All {len(res)} nearest matches lie within {min_distance_km} km of the query, so there are no "
                "distinct places like it elsewhere in the county.", "")
    same = (far.class_name == q.class_name).mean()

    fig, ax = plt.subplots(figsize=(7, 5))
    _backdrop(ax)
    ax.scatter(far.lon, far.lat, s=10, c="#2b6cb0", label=f"top {len(far)} matches beyond {min_distance_km:g} km")
    ax.scatter([float(q.lon)], [float(q.lat)], marker="*", s=220, color="red", edgecolor="k", label=f"query ({q.class_name})")
    ax.set_aspect(1 / math.cos(math.radians(float(q.lat))))
    ax.set_title(f"Places like ({float(q.lon):.3f}, {float(q.lat):.3f}) in {year}", fontsize=10)
    ax.legend(fontsize=8, loc="lower left")
    path = _save(fig, f"similar_{float(q.lon):.3f}_{float(q.lat):.3f}_{year}.png")

    top = far.head(10)[["lon", "lat", "class_name", "dist_km"]].round({"lon": 4, "lat": 4, "dist_km": 1})
    text = (f"Query node {q.node_id}, CDL {year} class {q.class_name}.\n"
            f"Top {len(far)} matches beyond {min_distance_km:g} km (Databricks Vector Search, 64-d embeddings).\n"
            f"Same CDL class as the query: {same:.0%}. Class mix of matches: {class_mix(far.class_name)}.\n"
            f"Index score of match 1 and match {len(far)}: {float(far[score_col].iloc[0]):.4f}, {float(far[score_col].iloc[-1]):.4f}.\n"
            f"Ten closest-in-embedding matches:\n{df_to_markdown(top)}\nMap saved: {path}")
    return text, path


def change_hotspots(top_n: int = 10, max_cropland_share: float | None = None, min_share_changed: float = 0.3) -> tuple[str, str]:
    """H3 hexagons with the largest share of embedding change 2023 to 2024, with a map."""
    from geo_embeddings.dbsql import run_query

    crop = f"AND share_cropland_2024 <= {float(max_cropland_share)}" if max_cropland_share is not None else ""
    df = run_query(f"SELECT h3_hex, h3_centeraswkt(h3) AS center, boundary_wkt, n_nodes, round(share_changed,3) AS "
                   f"share_changed, round(share_cdl_changed,3) AS share_cdl_changed, round(share_cropland_2024,3) AS "
                   f"share_cropland_2024, main_class_2024 FROM {H3_TABLE} WHERE share_changed >= {float(min_share_changed)} "
                   f"{crop} ORDER BY share_changed DESC LIMIT {int(top_n)}")
    if df.empty:
        return "No hexagons match those filters.", ""
    from shapely import wkt

    centers = [wkt.loads(c) for c in df.center]
    df["lon"], df["lat"] = [round(c.x, 4) for c in centers], [round(c.y, 4) for c in centers]
    fig, ax = plt.subplots(figsize=(7, 5))
    _backdrop(ax)
    for b in df.boundary_wkt:
        xy = np.asarray(wkt.loads(b).exterior.coords)
        ax.fill(xy[:, 0], xy[:, 1], color="#c0392b", alpha=0.75)
    ax.set_aspect(1 / math.cos(math.radians(33.0)))
    ax.set_title(f"Top {len(df)} change hotspots 2023 to 2024 (H3 resolution 7)", fontsize=10)
    path = _save(fig, f"hotspots_top{top_n}_crop{max_cropland_share}.png")
    show = df[["h3_hex", "lon", "lat", "n_nodes", "share_changed", "share_cdl_changed", "share_cropland_2024", "main_class_2024"]]
    return (f"share_changed = share of 200 m nodes with embedding change > {CHANGE_THRESHOLD}; "
            f"share_cdl_changed = share whose CDL class changed.\n{df_to_markdown(show)}\nMap saved: {path}"), path


# ---------- Earth Engine tools (public catalog only) ----------

def _ee():
    import ee

    from geo_embeddings.ee_auth import init_ee
    init_ee()
    return ee


def cdl_history(lon: float, lat: float, start_year: int = 2017, end_year: int = 2025) -> str:
    """CDL class at a point for each year (live from Earth Engine), with 2023 confidence when present."""
    from geo_embeddings.classes import cdl_to_class

    ee = _ee()
    pt = ee.Geometry.Point([lon, lat])
    rows = []
    for y in range(int(start_year), int(end_year) + 1):
        img = ee.Image(f"USDA/NASS/CDL/{y}")
        v = img.reduceRegion(ee.Reducer.first(), pt, 30).getInfo()
        meta = img.get("cropland_class_values").getInfo().split(","), img.get("cropland_class_names").getInfo().split(",")
        names = dict(zip(map(int, meta[0]), meta[1]))
        code = v.get("cropland")
        grouped = CLASS_NAMES.get(cdl_to_class(int(code)), "Excluded") if code is not None else "no data"
        rows.append({"year": y, "cdl_code": code, "cdl_name": names.get(int(code)) if code is not None else None,
                     "grouped_class": grouped, "confidence": v.get("confidence")})
    return "CDL at the 30 m pixel containing the point (USDA/NASS/CDL):\n" + df_to_markdown(pd.DataFrame(rows))


def embedding_change(lon: float, lat: float, year_a: int, year_b: int) -> str:
    """1 - cosine similarity of the 10 m embedding at a point between two years (live from Earth Engine)."""
    from geo_embeddings.embeddings import read_pixel

    _ee()
    a, b = np.array(read_pixel(lon, lat, int(year_a))), np.array(read_pixel(lon, lat, int(year_b)))
    change = 1 - float(a @ b)
    verdict = "changed" if change > CHANGE_THRESHOLD else "stable"
    return (f"Embedding change at ({lon}, {lat}) from {year_a} to {year_b}: {change:.3f} ({verdict}; threshold "
            f"{CHANGE_THRESHOLD}, which is about the 99th percentile of change where CDL saw no change in 2023 to 2024).")


def burned_area(lon: float, lat: float, radius_km: float = 2.0, start_year: int = 2020, end_year: int = 2025) -> str:
    """Share of a circle mapped as burned by MODIS MCD64A1 each year, and the median burn day."""
    ee = _ee()
    region = ee.Geometry.Point([lon, lat]).buffer(float(radius_km) * 1000)
    rows = []
    for y in range(int(start_year), int(end_year) + 1):
        img = ee.ImageCollection("MODIS/061/MCD64A1").filterDate(f"{y}-01-01", f"{y + 1}-01-01").select("BurnDate").max()
        share = img.gt(0).unmask(0).reduceRegion(ee.Reducer.mean(), region, 500).get("BurnDate").getInfo()
        day = img.reduceRegion(ee.Reducer.median(), region, 500).get("BurnDate").getInfo()
        rows.append({"year": y, "burned_share": round(share or 0, 3), "median_burn_day_of_year": day})
    return (f"MODIS burned area (MCD64A1, 500 m) within {radius_km} km of ({lon}, {lat}):\n"
            + df_to_markdown(pd.DataFrame(rows)))


# ---------- validation evidence (stored results) ----------

def validation_evidence(topic: str, class_name: str | None = None) -> str:
    """Stored validation numbers for 'classification', 'search' or 'change', optionally for one class."""
    topic = topic.lower().strip()
    if topic == "classification":
        r = json.loads((RESULTS / "rf_spatial10km.json").read_text())
        b = r["block_bootstrap"]
        names = {int(k): v for k, v in r["class_names"].items()}
        rows = [{"class": names[int(c)], "f1": round(f["value"], 3), "ci_95": f"{f['ci'][0]:.3f} to {f['ci'][1]:.3f}",
                 "blocks": f["n_blocks"]} for c, f in b["f1_per_class"].items()]
        df = pd.DataFrame(rows)
        if class_name:
            df = df[df["class"].str.lower() == class_name.lower()]
        oa = b["overall_accuracy"]
        return (f"Random Forest on embeddings, 5-fold CV on 10 km spatial blocks, agreement with CDL on pure pixels. "
                f"Overall accuracy {oa['value']:.3f} (95% {oa['ci'][0]:.3f} to {oa['ci'][1]:.3f}). A random split "
                f"would report about 0.955. Model error is 25 to 30% where 2023 CDL confidence is under 50 and 3 to 4% "
                f"at 99 to 100.\n{df_to_markdown(df)}")
    if topic == "search":
        r = json.loads((RESULTS / "search_eval_2024.json").read_text())
        df = pd.DataFrame(r["per_class"])[["class_name", "precision_beyond_10km", "precision_naive", "share_of_county"]].round(3)
        if class_name:
            df = df[df.class_name.str.lower() == class_name.lower()]
        vs = r.get("vector_search_vs_exact", {})
        return (f"Share of top {r['k']} matches beyond {r['min_dist_m'] / 1000:g} km with the query's CDL class "
                f"(30 pure queries per class, {r['year']}). Vector Search agrees with exact search on answer quality "
                f"(top-50 overlap {vs.get('mean_recall', float('nan')):.2f}, swaps are near-ties).\n{df_to_markdown(df)}")
    if topic == "change":
        r = json.loads((RESULTS / "change_2023_2024.json").read_text())
        df = pd.DataFrame(r["pure_both_years"]).round(3)
        return ("Change score AUC against unchanged nodes, by CDL transition type (pure nodes both years). "
                "Near 0.5 means the score does not separate that transition from stable land. Natural-to-natural CDL "
                "flips are often relabeling, not land change.\n" + df_to_markdown(df))
    raise ValueError("topic must be 'classification', 'search' or 'change'.")
