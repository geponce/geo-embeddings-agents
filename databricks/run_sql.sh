#!/usr/bin/env bash
# Run each statement of a SQL file on the default SQL warehouse.
# Usage: databricks/run_sql.sh <sql file>   (from the repo root)
# Uses the CLI profile in DATABRICKS_CONFIG_PROFILE (default: geo-embeddings).
set -euo pipefail
PROFILE="${DATABRICKS_CONFIG_PROFILE:-geo-embeddings}"
python3 - "$PROFILE" "$1" <<'PY'
import subprocess, sys
profile, path = sys.argv[1], sys.argv[2]
sql = "\n".join(l for l in open(path) if not l.lstrip().startswith("--"))
for stmt in (s.strip() for s in sql.split(";")):
    if stmt:
        subprocess.run(["databricks", "experimental", "aitools", "tools", "query",
                        stmt, "--profile", profile], check=True, stdout=subprocess.DEVNULL)
print(f"ran {path}")
PY
