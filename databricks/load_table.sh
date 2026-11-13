#!/usr/bin/env bash
# Upload a local Parquet file to the project's Unity Catalog volume and run a SQL file.
# Usage: databricks/load_table.sh <parquet file> <sql file>   (from the repo root)
# Uses the CLI profile in DATABRICKS_CONFIG_PROFILE (default: geo-embeddings).
set -euo pipefail

FILE="$1"
SQL="$2"
PROFILE="${DATABRICKS_CONFIG_PROFILE:-geo-embeddings}"
CATALOG=workspace
SCHEMA=geo_embeddings
VOLUME=raw

db() { databricks "$@" --profile "$PROFILE"; }

# Schema and volume (ignore "already exists").
db schemas create --json "{\"name\": \"$SCHEMA\", \"catalog_name\": \"$CATALOG\",
  \"comment\": \"geo-embeddings-agents project tables\"}" >/dev/null 2>&1 || true
db volumes create --json "{\"catalog_name\": \"$CATALOG\", \"schema_name\": \"$SCHEMA\",
  \"name\": \"$VOLUME\", \"volume_type\": \"MANAGED\",
  \"comment\": \"Raw files exported from Earth Engine\"}" >/dev/null 2>&1 || true

db fs cp --overwrite "$FILE" "dbfs:/Volumes/$CATALOG/$SCHEMA/$VOLUME/$(basename "$FILE")"

"$(dirname "$0")/run_sql.sh" "$SQL"
