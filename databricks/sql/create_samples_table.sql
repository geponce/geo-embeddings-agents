-- Stage 2: Delta table of embedding samples with grouped CDL labels, Pinal County AZ.
-- Source: Parquet written by `python -m geo_embeddings.build_samples`, uploaded to a UC volume.
CREATE OR REPLACE TABLE workspace.geo_embeddings.samples_pinal
COMMENT 'Stratified sample of Google Satellite Embeddings (64-d, 10 m) with grouped USDA CDL labels. Pinal County AZ, 2023-2024, 2000 pure pixels per class per year. Embeddings: The AlphaEarth Foundations Satellite Embedding dataset is produced by Google and Google DeepMind (CC-BY 4.0). Labels: USDA NASS Cropland Data Layer.'
AS
SELECT
  monotonically_increasing_id() AS sample_id,
  * EXCEPT (_rescued_data)
FROM read_files(
  '/Volumes/workspace/geo_embeddings/raw/pinal_samples_2023_2024_n2000.parquet',
  format => 'parquet'
);

ALTER TABLE workspace.geo_embeddings.samples_pinal ALTER COLUMN label COMMENT 'Grouped class id 0-10 (see src/geo_embeddings/classes.py)';
ALTER TABLE workspace.geo_embeddings.samples_pinal ALTER COLUMN cdl_code COMMENT 'Original USDA CDL cropland code';
ALTER TABLE workspace.geo_embeddings.samples_pinal ALTER COLUMN cdl_conf COMMENT 'CDL per-pixel confidence 0-100, or -1 when the CDL year has no confidence band (2024)';
ALTER TABLE workspace.geo_embeddings.samples_pinal ALTER COLUMN lon COMMENT 'Longitude of the CDL 30 m pixel center (WGS84)';
ALTER TABLE workspace.geo_embeddings.samples_pinal ALTER COLUMN lat COMMENT 'Latitude of the CDL 30 m pixel center (WGS84)';
