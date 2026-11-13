-- Stage 4: wall-to-wall 200 m grid of embeddings with CDL labels, Pinal County AZ, 2023 and 2024.
-- Source: Parquet written by `python -m geo_embeddings.build_grid`, uploaded to a UC volume.
-- Change Data Feed is required for a Delta Sync Vector Search index.
CREATE OR REPLACE TABLE workspace.geo_embeddings.grid_pinal_200m
TBLPROPERTIES (delta.enableChangeDataFeed = true)
COMMENT 'Wall-to-wall 200 m grid (CONUS Albers EPSG:5070) over Pinal County AZ, 2023 and 2024. Each row is the 10 m Google Satellite Embedding pixel at a grid node plus USDA CDL labels. Embeddings: The AlphaEarth Foundations Satellite Embedding dataset is produced by Google and Google DeepMind (CC-BY 4.0). Labels: USDA NASS Cropland Data Layer.'
AS
SELECT
  concat(cast(year AS STRING), '_', node_id) AS id,
  node_id, year, x, y, lon, lat, cdl_code, label_any, class_name, pure, cdl_conf,
  cast(array(A00, A01, A02, A03, A04, A05, A06, A07, A08, A09, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23, A24, A25, A26, A27, A28, A29, A30, A31, A32, A33, A34, A35, A36, A37, A38, A39, A40, A41, A42, A43, A44, A45, A46, A47, A48, A49, A50, A51, A52, A53, A54, A55, A56, A57, A58, A59, A60, A61, A62, A63) AS ARRAY<FLOAT>) AS embedding
FROM read_files(
  '/Volumes/workspace/geo_embeddings/raw/grid_pinal_200m_2023_2024.parquet',
  format => 'parquet'
);

ALTER TABLE workspace.geo_embeddings.grid_pinal_200m ALTER COLUMN id COMMENT 'Primary key, year_nodeid';
ALTER TABLE workspace.geo_embeddings.grid_pinal_200m ALTER COLUMN node_id COMMENT 'Grid node, Albers x_y of the node center in meters (same across years)';
ALTER TABLE workspace.geo_embeddings.grid_pinal_200m ALTER COLUMN label_any COMMENT 'Grouped class id 0-12 (scheme v2), or -1 for excluded CDL codes';
ALTER TABLE workspace.geo_embeddings.grid_pinal_200m ALTER COLUMN pure COMMENT '1 if the 3x3 CDL block (90 m) around the node is one grouped class';
ALTER TABLE workspace.geo_embeddings.grid_pinal_200m ALTER COLUMN embedding COMMENT '64-d unit-length Satellite Embedding at the node (10 m pixel)';
