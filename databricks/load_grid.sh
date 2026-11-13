#!/usr/bin/env bash
# Stage 4: create workspace.geo_embeddings.grid_pinal_200m from the grid Parquet.
# Usage: databricks/load_grid.sh   (from the repo root)
set -euo pipefail
"$(dirname "$0")/load_table.sh" data/processed/grid_pinal_200m_2023_2024.parquet \
  databricks/sql/create_grid_table.sql
