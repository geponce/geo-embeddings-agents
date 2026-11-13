"""Guard for SQL written by an agent: one read-only SELECT on the project's tables.

This is a defense layer on top of a read-only tool, not a full SQL parser. It
rejects anything that is not a single SELECT (or WITH ... SELECT), any write or
admin keyword, functions that reach files, the network or other models, and any
three-part table name outside the project's schema. The query is then wrapped
in an outer SELECT with a row limit.
"""

from __future__ import annotations

import re

ALLOWED_TABLES = frozenset({
    "workspace.geo_embeddings.samples_pinal",
    "workspace.geo_embeddings.grid_pinal_200m",
    "workspace.geo_embeddings.h3_pinal_r7",
})
MAX_ROWS = 200

_FORBIDDEN_WORDS = (
    "insert", "update", "delete", "merge", "drop", "create", "alter", "truncate", "grant",
    "revoke", "copy", "optimize", "vacuum", "call", "set", "use", "refresh", "msck", "repair",
    "put", "remove", "restore", "clone", "comment", "analyze", "cache", "uncache", "load",
)
_FORBIDDEN_FUNCS = ("read_files", "http_request", "remote_query", "secret", "ai_", "vector_search",
                    "read_kafka", "read_kinesis", "read_pubsub", "read_pulsar", "cloud_files")


def _strip_strings_and_comments(sql: str) -> str:
    """Blank out string literals and comments so keyword checks ignore their contents."""
    sql = re.sub(r"--[^\n]*", " ", sql)
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.S)
    sql = re.sub(r"'(?:[^'\\]|\\.)*'", "''", sql)
    return re.sub(r'"(?:[^"\\]|\\.)*"', '""', sql)


def check_select(sql: str, max_rows: int = MAX_ROWS) -> str:
    """Return a row-limited version of a safe SELECT, or raise ValueError saying why not."""
    body = sql.strip().rstrip(";").strip()
    bare = _strip_strings_and_comments(body).lower()
    if ";" in bare:
        raise ValueError("Only one statement is allowed.")
    if not re.match(r"^\s*(select|with)\b", bare):
        raise ValueError("Only SELECT queries (optionally starting with WITH) are allowed.")
    for word in _FORBIDDEN_WORDS:
        if re.search(rf"\b{word}\b", bare):
            raise ValueError(f"The keyword '{word.upper()}' is not allowed in a read-only query.")
    for fn in _FORBIDDEN_FUNCS:
        if re.search(rf"\b{re.escape(fn)}", bare):
            raise ValueError(f"The function '{fn}' is not allowed.")
    if re.search(r"`|/volumes|\bsystem\.|\bsamples\.", bare):
        raise ValueError("Backticks, file paths, system tables and other catalogs are not allowed.")
    for name in re.findall(r"\b[a-z_][a-z0-9_]*\.[a-z_][a-z0-9_]*\.[a-z_][a-z0-9_]*\b", bare):
        if name not in ALLOWED_TABLES:
            raise ValueError(f"Table '{name}' is not allowed. Use: {', '.join(sorted(ALLOWED_TABLES))}.")
    if not any(t in bare for t in ALLOWED_TABLES):
        raise ValueError(f"Query one of: {', '.join(sorted(ALLOWED_TABLES))} (full three-part names).")
    return f"SELECT * FROM ({body}) AS agent_query LIMIT {int(max_rows)}"
