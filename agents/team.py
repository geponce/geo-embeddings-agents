"""The agent team: read-only tools on an in-process MCP server, three specialists and a planner.

planner (main agent)      splits the question, delegates, sends its draft to the validator
earth-engine-data         live per-place data from public Earth Engine datasets, 2017 to 2025
databricks-analyst        Unity Catalog tables, Vector Search, H3 hotspots, place profiles
validator                 checks every claim against stored validation evidence and independent data

All tools are read-only. Built-in tools (Bash, file edits, web) are removed, so the
agents can only use the tools defined here. Project files such as CLAUDE.md are not
loaded into the agents' context.
"""

from __future__ import annotations

import asyncio
from typing import Any

from claude_agent_sdk import AgentDefinition, ClaudeAgentOptions, create_sdk_mcp_server, tool
from mcp.types import ToolAnnotations

from geo_embeddings import agent_tools as T

MODEL = "claude-opus-5-5"
SERVER = "geo"
READ_ONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False)


def _ok(text: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}]}


def _err(e: Exception) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": f"Tool error: {e}"}], "is_error": True}


async def _call(fn, *args, **kwargs) -> dict[str, Any]:
    """Run a blocking tool function in a thread and wrap its text (or error) for the agent."""
    try:
        out = await asyncio.to_thread(fn, *args, **kwargs)
        return _ok(out[0] if isinstance(out, tuple) else out)
    except Exception as e:  # report the reason so the agent can adjust, rather than crash the loop
        return _err(e)


def _num(desc: str) -> dict[str, Any]:
    return {"type": "number", "description": desc}


LONLAT = {"lon": _num("Longitude, WGS84"), "lat": _num("Latitude, WGS84")}


# ---------- tools ----------

@tool("describe_data", "Describe the study area, tables, classes, years and what the change score means.",
      {"type": "object", "properties": {}}, annotations=READ_ONLY)
async def describe_data(args: dict[str, Any]) -> dict[str, Any]:
    return _ok(T.DATA_GUIDE)


@tool("run_sql", "Run one read-only SELECT on the project's Databricks tables (full three-part names). "
      "Returns at most 200 rows. Do not select the embedding column unless needed.",
      {"type": "object", "properties": {"sql": {"type": "string"}}, "required": ["sql"]}, annotations=READ_ONLY)
async def run_sql(args: dict[str, Any]) -> dict[str, Any]:
    return await _call(T.run_sql, args["sql"])


@tool("place_profile", "CDL class in 2023 and 2024, embedding change score and H3 hexagon context at a point.",
      {"type": "object", "properties": LONLAT, "required": ["lon", "lat"]}, annotations=READ_ONLY)
async def place_profile(args: dict[str, Any]) -> dict[str, Any]:
    return await _call(T.place_profile, float(args["lon"]), float(args["lat"]))


@tool("find_similar", "Find places like this one: the most similar 200 m grid nodes by embedding, farther than "
      "min_distance_km from the query, via Databricks Vector Search. Saves a map.",
      {"type": "object", "properties": {
          **LONLAT,
          "year": {"type": "integer", "enum": [2023, 2024]},
          "k": {"type": "integer", "minimum": 5, "maximum": 200, "description": "Number of matches, default 50"},
          "min_distance_km": _num("Ignore matches closer than this, default 10 (avoids the same field)")},
       "required": ["lon", "lat"]}, annotations=READ_ONLY)
async def find_similar(args: dict[str, Any]) -> dict[str, Any]:
    return await _call(T.find_similar, float(args["lon"]), float(args["lat"]), int(args.get("year", 2024)),
                       int(args.get("k", 50)), float(args.get("min_distance_km", 10.0)))


@tool("change_hotspots", "H3 hexagons with the most embedding change from 2023 to 2024. Optionally exclude farmland "
      "with max_cropland_share (for example 0.05). Saves a map.",
      {"type": "object", "properties": {
          "top_n": {"type": "integer", "minimum": 1, "maximum": 50},
          "max_cropland_share": _num("Keep hexagons with at most this share of cropland in 2024"),
          "min_share_changed": _num("Keep hexagons with at least this share of changed nodes, default 0.3")}},
      annotations=READ_ONLY)
async def change_hotspots(args: dict[str, Any]) -> dict[str, Any]:
    return await _call(T.change_hotspots, int(args.get("top_n", 10)), args.get("max_cropland_share"),
                       float(args.get("min_share_changed", 0.3)))


@tool("cdl_history", "USDA CDL class at a point for each year from start_year to end_year (2017 to 2025), live "
      "from Earth Engine, with CDL confidence where available.",
      {"type": "object", "properties": {**LONLAT, "start_year": {"type": "integer", "minimum": 2017, "maximum": 2025},
                                        "end_year": {"type": "integer", "minimum": 2017, "maximum": 2025}},
       "required": ["lon", "lat"]}, annotations=READ_ONLY)
async def cdl_history(args: dict[str, Any]) -> dict[str, Any]:
    return await _call(T.cdl_history, float(args["lon"]), float(args["lat"]),
                       int(args.get("start_year", 2017)), int(args.get("end_year", 2025)))


@tool("embedding_change", "Change score (1 - cosine) of the 10 m embedding at a point between any two years "
      "2017 to 2025, live from Earth Engine.",
      {"type": "object", "properties": {**LONLAT, "year_a": {"type": "integer", "minimum": 2017, "maximum": 2025},
                                        "year_b": {"type": "integer", "minimum": 2017, "maximum": 2025}},
       "required": ["lon", "lat", "year_a", "year_b"]}, annotations=READ_ONLY)
async def embedding_change(args: dict[str, Any]) -> dict[str, Any]:
    return await _call(T.embedding_change, float(args["lon"]), float(args["lat"]), int(args["year_a"]), int(args["year_b"]))


@tool("burned_area", "Share of a circle mapped as burned by NASA MODIS (MCD64A1) in each year, with the median "
      "burn day of year. Independent evidence for fire-related change.",
      {"type": "object", "properties": {**LONLAT, "radius_km": _num("Default 2"),
                                        "start_year": {"type": "integer", "minimum": 2001, "maximum": 2025},
                                        "end_year": {"type": "integer", "minimum": 2001, "maximum": 2025}},
       "required": ["lon", "lat"]}, annotations=READ_ONLY)
async def burned_area(args: dict[str, Any]) -> dict[str, Any]:
    return await _call(T.burned_area, float(args["lon"]), float(args["lat"]), float(args.get("radius_km", 2.0)),
                       int(args.get("start_year", 2020)), int(args.get("end_year", 2025)))


@tool("validation_evidence", "Stored validation results: topic 'classification' (per-class F1 with 95% intervals), "
      "'search' (match precision by class) or 'change' (AUC by transition type). Optional class_name filter.",
      {"type": "object", "properties": {"topic": {"type": "string", "enum": ["classification", "search", "change"]},
                                        "class_name": {"type": "string"}},
       "required": ["topic"]}, annotations=READ_ONLY)
async def validation_evidence(args: dict[str, Any]) -> dict[str, Any]:
    return await _call(T.validation_evidence, args["topic"], args.get("class_name"))


ALL_TOOLS = [describe_data, run_sql, place_profile, find_similar, change_hotspots, cdl_history,
             embedding_change, burned_area, validation_evidence]


def q(name: str) -> str:
    """Fully qualified MCP tool name."""
    return f"mcp__{SERVER}__{name}"


# ---------- agents ----------

EE_AGENT = AgentDefinition(
    description="Live per-place data from public Earth Engine datasets for any year 2017 to 2025: CDL label history, "
                "embedding change between two years, MODIS burned area. Use for questions outside 2023 and 2024 "
                "or for independent evidence about a specific place.",
    prompt=f"""You fetch facts about specific places from public Earth Engine datasets. Report the numbers the tools
return, name the dataset each came from, and say when a tool returned no data. Do not interpret beyond what the
numbers support, and do not guess values you did not fetch.

{T.DATA_GUIDE}""",
    tools=[q("cdl_history"), q("embedding_change"), q("burned_area"), q("describe_data")],
    model="inherit", effort="medium",
)

ANALYST = AgentDefinition(
    description="Analysis on the Databricks lakehouse for 2023 and 2024: read-only SQL on the grid, H3 and sample "
                "tables, place profiles, 'find places like this one' with Vector Search, change hotspots. Saves maps.",
    prompt=f"""You answer quantitative questions from the project's Databricks tables. Prefer the purpose-built tools
(place_profile, find_similar, change_hotspots) and use run_sql for anything else. Report exact numbers, the table or
tool they came from, and the paths of any maps saved. Keep tables short.

{T.DATA_GUIDE}""",
    tools=[q("run_sql"), q("place_profile"), q("find_similar"), q("change_hotspots"), q("describe_data")],
    model="inherit", effort="medium",
)

VALIDATOR = AgentDefinition(
    description="Checks a draft answer before it is returned. Give it the question, the draft and the evidence. "
                "It labels each claim and states how sure we can be, with numbers.",
    prompt=f"""You are the validation agent. For each claim in the draft you receive, decide whether the evidence
supports it, using validation_evidence for the relevant topic and class, and independent data (cdl_history,
burned_area, run_sql) where a claim concerns a specific place.

Label each claim Supported, Weak or Unsupported, with one line of reason that cites numbers: per-class F1 and its
95% interval, search precision for that class, change AUC for that kind of transition, CDL confidence, or burned
area. Point out when the reference labels themselves are the weak link (CDL is not ground truth; natural-class
flips between years are often relabeling). Flag any number in the draft that no tool returned. End with an overall
confidence of high, medium or low and the main reason.

{T.DATA_GUIDE}""",
    tools=[q("validation_evidence"), q("cdl_history"), q("burned_area"), q("run_sql"), q("describe_data")],
    model="inherit", effort="high",
)

PLANNER_PROMPT = """You answer questions about land cover and land change in Pinal County, Arizona, by leading a team
of specialist agents. Your only data tool is describe_data. Everything else goes through the agents:

- databricks-analyst for the 2023 and 2024 tables, similarity search, hotspots and place profiles
- earth-engine-data for live per-place data in any year 2017 to 2025 and for burned area
- validator to check your draft before you answer

Delegate independent data requests in parallel and wait for their reports. Then write a draft and send the validator
the question, the draft and the evidence you received, and wait for its verdict. Revise the draft to match it.
Never state a number that no agent returned.

Reply in markdown with these parts: a direct answer in two or three sentences; an evidence table (claim, value,
source); how sure we are, using the validator's labels and numbers; maps (the saved file paths); caveats. Keep it
under 350 words."""


def build_options(max_budget_usd: float = 3.0, outputs_dir: str | None = None) -> ClaudeAgentOptions:
    """ClaudeAgentOptions for the team: our tools only, spend and subagent caps, no project files loaded."""
    server = create_sdk_mcp_server(name=SERVER, version="1.0.0", tools=ALL_TOOLS)
    return ClaudeAgentOptions(
        model=MODEL,
        system_prompt=PLANNER_PROMPT,
        mcp_servers={SERVER: server},
        tools=["Agent"],                              # remove every other built-in tool
        allowed_tools=["Agent", q("describe_data"), *[q(t.name) for t in ALL_TOOLS]],
        agents={"earth-engine-data": EE_AGENT, "databricks-analyst": ANALYST, "validator": VALIDATOR},
        setting_sources=[],                           # do not load CLAUDE.md or local settings
        cwd=outputs_dir,
        effort="high",
        max_budget_usd=max_budget_usd,
        env={"CLAUDE_CODE_MAX_SUBAGENT_SPAWN_DEPTH": "1", "CLAUDE_CODE_MAX_CONCURRENT_SUBAGENTS": "4"},
    )
