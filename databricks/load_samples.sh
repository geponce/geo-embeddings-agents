#!/usr/bin/env bash
# Stage 2: create workspace.geo_embeddings.samples_pinal from the sample Parquet.
# Usage: databricks/load_samples.sh   (from the repo root)
set -euo pipefail
"$(dirname "$0")/load_table.sh" data/processed/pinal_samples_2023_2024_n2000.parquet \
  databricks/sql/create_samples_table.sql
