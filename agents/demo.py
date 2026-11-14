"""Scripted walkthroughs of the agent tools, with no language model involved.

Run with:  python -m agents.demo

Each walkthrough runs the same read-only tools the agent team uses, in a fixed order a
planner could choose, against the real Databricks tables and public Earth Engine data.
It costs no Claude tokens. The output is raw tool evidence, not an agent's answer, and
each file says so. Results go to docs/examples/, with maps in docs/examples/maps/.
"""

from __future__ import annotations

import shutil
from datetime import date
from pathlib import Path
from typing import Callable

from shapely import wkt

from geo_embeddings import agent_tools as T
from geo_embeddings.dbsql import run_query

REPO = Path(__file__).resolve().parents[1]
EXAMPLES = REPO / "docs" / "examples"
MAPS = EXAMPLES / "maps"

NOTE = ("> **Scripted snapshot, no model involved.** These are the outputs of the agent team's read-only tools, "
        "run in a fixed order by `agents/demo.py` on {day}. A live run (`python -m agents.run`) lets Claude "
        "choose the tools and write the answer, with a validator agent checking it.")


def _text(out: str | tuple[str, str]) -> str:
    """Tool text, with any saved map copied into docs/examples/maps and linked."""
    if isinstance(out, str):
        return out
    text, path = out
    if path:
        MAPS.mkdir(parents=True, exist_ok=True)
        dest = MAPS / Path(path).name
        shutil.copy(REPO / path, dest)
        text = text.replace(path, f"maps/{dest.name}") + f"\n\n![map](maps/{dest.name})"
    return text


def _write(slug: str, question: str, steps: list[tuple[str, str, Callable[[], str | tuple[str, str]]]]) -> Path:
    """Run each (agent, call, fn) step and write the evidence as markdown."""
    parts = [f"# {question}", NOTE.format(day=date.today().isoformat())]
    for i, (agent, call, fn) in enumerate(steps, 1):
        print(f"[{slug}] {i}. {agent}: {call}", flush=True)
        try:
            body = _text(fn())
        except Exception as e:  # keep going and record the failure honestly
            body = f"Tool error: {e}"
        parts.append(f"## {i}. {agent} · `{call}`\n\n{body}")
    EXAMPLES.mkdir(parents=True, exist_ok=True)
    path = EXAMPLES / f"{slug}.md"
    path.write_text("\n\n".join(parts) + "\n")
    return path


def top_nonfarm_hotspot() -> tuple[float, float]:
    """Center of the non-farm H3 hexagon with the largest share of changed nodes."""
    df = run_query(f"SELECT h3_centeraswkt(h3) AS c FROM {T.H3_TABLE} WHERE share_cropland_2024 <= 0.05 "
                   f"AND share_changed >= 0.3 ORDER BY share_changed DESC LIMIT 1")
    p = wkt.loads(df.c.iloc[0])
    return round(p.x, 4), round(p.y, 4)


def main() -> None:
    """Run the three showcase walkthroughs."""
    lon, lat = -111.855, 33.082
    print(_write("cotton_similarity", f"Where else in Pinal County looks like the cotton field at {lon}, {lat}?", [
        ("databricks-analyst", f"place_profile({lon}, {lat})", lambda: T.place_profile(lon, lat)),
        ("databricks-analyst", f"find_similar({lon}, {lat}, 2024, k=50, min_distance_km=10)",
         lambda: T.find_similar(lon, lat, 2024, 50, 10.0)),
        ("validator", "validation_evidence('search', 'Cotton')", lambda: T.validation_evidence("search", "Cotton")),
        ("validator", "validation_evidence('classification', 'Cotton')",
         lambda: T.validation_evidence("classification", "Cotton")),
    ]))

    hx, hy = top_nonfarm_hotspot()
    print(_write("nonfarm_change", "Where did land outside farms change most from 2023 to 2024, and why?", [
        ("databricks-analyst", "change_hotspots(top_n=10, max_cropland_share=0.05)",
         lambda: T.change_hotspots(10, 0.05, 0.3)),
        ("earth-engine-data", f"burned_area({hx}, {hy}, radius_km=2, 2020-2025)",
         lambda: T.burned_area(hx, hy, 2.0, 2020, 2025)),
        ("earth-engine-data", f"cdl_history({hx}, {hy}, 2020-2025)", lambda: T.cdl_history(hx, hy, 2020, 2025)),
        ("earth-engine-data", f"embedding_change({hx}, {hy}, 2023, 2024)",
         lambda: T.embedding_change(hx, hy, 2023, 2024)),
        ("validator", "validation_evidence('change')", lambda: T.validation_evidence("change")),
    ]))

    print(_write("orchard_reliability", "How reliable is the orchard classification?", [
        ("validator", "validation_evidence('classification', 'Orchards')",
         lambda: T.validation_evidence("classification", "Orchards")),
        ("validator", "validation_evidence('search', 'Orchards')",
         lambda: T.validation_evidence("search", "Orchards")),
        ("databricks-analyst", "run_sql(orchard grid nodes per year)", lambda: T.run_sql(
            "SELECT year, count(*) AS n_nodes, sum(pure) AS n_pure, "
            "round(avg(CASE WHEN cdl_conf >= 0 THEN cdl_conf END), 1) AS mean_cdl_conf "
            f"FROM {T.GRID_TABLE} WHERE class_name = 'Orchards' GROUP BY year ORDER BY year")),
    ]))


if __name__ == "__main__":
    main()
