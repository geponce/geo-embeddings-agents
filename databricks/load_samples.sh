#!/usr/bin/env bash
# Stage 2: upload the sample Parquet to a Unity Catalog volume and create a Delta table.
# Usage: databricks/load_samples.sh   (from the repo root)
# Uses the CLI profile in DATABRICKS_CONFIG_PROFILE (default: geo-embeddings).
set -euo pipefail

PROFILE="${DATABRICKS_CONFIG_PROFILE:-geo-embeddings}"
CATALOG=workspace
SCHEMA=geo_embeddings
VOLUME=raw
FILE=data/processed/pinal_samples_2023_2024_n2000.parquet

db() { databricks "$@" --profile "$PROFILE"; }

# Schema and volume (ignore "already exists").
db schemas create --json "{\"name\": \"$SCHEMA\", \"catalog_name\": \"$CATALOG\",
  \"comment\": \"geo-embeddings-agents project tables\"}" >/dev/null 2>&1 || true
db volumes create --json "{\"catalog_name\": \"$CATALOG\", \"schema_name\": \"$SCHEMA\",
  \"name\": \"$VOLUME\", \"volume_type\": \"MANAGED\",
  \"comment\": \"Raw files exported from Earth Engine\"}" >/dev/null 2>&1 || true

db fs cp --overwrite "$FILE" "dbfs:/Volumes/$CATALOG/$SCHEMA/$VOLUME/$(basename "$FILE")"

# Run each SQL statement on the default SQL warehouse.
python3 - "$PROFILE" databricks/sql/create_samples_table.sql <<'EOF'
import subprocess, sys
profile, path = sys.argv[1], sys.argv[2]
sql = "\n".join(l for l in open(path) if not l.lstrip().startswith("--"))
for stmt in (s.strip() for s in sql.split(";")):
    if stmt:
        subprocess.run(["databricks", "experimental", "aitools", "tools", "query",
                        stmt, "--profile", profile], check=True, stdout=subprocess.DEVNULL)
print("table created")
EOF
