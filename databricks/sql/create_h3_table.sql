-- Stage 4: roll the 200 m grid up to H3 hexagons (resolution 7, about 5.2 km2 each) in Databricks SQL.
-- Per hexagon: embedding change 2023 -> 2024, how much CDL says changed, and the 2024 class mix.
-- change = 1 - cosine similarity of a node's 2023 and 2024 embeddings (unit vectors, so 1 - dot product).
-- A node counts as "changed" when change > 0.14, about the 99th percentile of change among
-- pure nodes whose CDL class did not change (see docs/plan.md, 2026-11-13).
CREATE OR REPLACE TABLE workspace.geo_embeddings.h3_pinal_r7
COMMENT 'Pinal County AZ 200 m embedding grid aggregated to H3 resolution 7 hexagons. Embedding change 2023 to 2024 and CDL class mix per hexagon. Embeddings: The AlphaEarth Foundations Satellite Embedding dataset is produced by Google and Google DeepMind (CC-BY 4.0). Labels: USDA NASS Cropland Data Layer.'
AS
WITH pairs AS (
  SELECT
    a.node_id, a.lon, a.lat,
    a.label_any AS label_2023, b.label_any AS label_2024, b.class_name AS class_2024,
    1 - aggregate(zip_with(a.embedding, b.embedding, (x, y) -> x * y), 0D, (acc, v) -> acc + v) AS change
  FROM workspace.geo_embeddings.grid_pinal_200m a
  JOIN workspace.geo_embeddings.grid_pinal_200m b
    ON a.node_id = b.node_id AND a.year = 2023 AND b.year = 2024
),
cells AS (
  SELECT h3_longlatash3(lon, lat, 7) AS h3, * FROM pairs
)
SELECT
  h3,
  h3_h3tostring(h3) AS h3_hex,
  h3_boundaryaswkt(h3) AS boundary_wkt,
  count(*) AS n_nodes,
  avg(change) AS change_mean,
  percentile_approx(change, 0.9) AS change_p90,
  avg(CASE WHEN change > 0.14 THEN 1.0 ELSE 0.0 END) AS share_changed,
  avg(CASE WHEN label_2023 >= 0 AND label_2024 >= 0 AND label_2023 <> label_2024 THEN 1.0 ELSE 0.0 END) AS share_cdl_changed,
  avg(CASE WHEN label_2024 IN (2, 3, 4, 5, 6, 11, 12) THEN 1.0 ELSE 0.0 END) AS share_cropland_2024,
  mode(class_2024) AS main_class_2024
FROM cells
GROUP BY h3;

ALTER TABLE workspace.geo_embeddings.h3_pinal_r7 ALTER COLUMN share_changed COMMENT 'Share of grid nodes with embedding change > 0.14';
ALTER TABLE workspace.geo_embeddings.h3_pinal_r7 ALTER COLUMN share_cdl_changed COMMENT 'Share of grid nodes whose grouped CDL class differs between 2023 and 2024';
