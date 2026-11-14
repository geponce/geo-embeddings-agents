# geo-embeddings-agents

**Find places like this one.** An end-to-end workflow that turns Google's Satellite Embedding dataset in Earth Engine into validated land-cover maps, similarity search and change detection on Databricks, driven by a small team of Claude agents.

> Work in progress. This README grows as each stage is completed.

![Architecture of geo-embeddings-agents](docs/architecture.svg)

Read [how it works](docs/overview.md) for the datasets, the pipeline, why Databricks and H3, and what the validation shows.

## Why
Satellite embeddings compress a year of observations for every 10 m pixel into 64 numbers. They make it easy to ask "where else looks like this?" But an embedding is only useful if we know when to trust it. This project puts validation at the center.

## Planned workflow
1. Sample embeddings and USDA Cropland Data Layer labels in Earth Engine (Python API)
2. Load them into Delta tables on Databricks
3. Train and validate a land-cover classifier, tracked in MLflow
4. Build a vector index for similarity search and a simple change score
5. Let Claude agents answer plain-language questions with maps, tables and stated uncertainty

## Stack
Google Earth Engine, Python, Databricks (Unity Catalog, MLflow, Vector Search), Claude Agent SDK, Claude Code.

## Setup
These steps reproduce the current state of the project.

1. Create the conda environment from the repo root. This also installs the project package in editable mode.
   ```
   conda env create -f environment.yml
   conda activate geo-embeddings
   ```
2. Authenticate Earth Engine with Google Application Default Credentials. The standard `earthengine authenticate` flow may be blocked for personal accounts.
   ```
   gcloud auth application-default login --scopes=https://www.googleapis.com/auth/earthengine,https://www.googleapis.com/auth/cloud-platform
   gcloud auth application-default set-quota-project <your-project>
   ```
3. Copy `.env.example` to `.env` and set `EE_PROJECT` to your Earth Engine Cloud project.
4. Install the [Databricks CLI](https://docs.databricks.com/dev-tools/cli/install.html) and sign in to your workspace.
   ```
   databricks auth login --host <your-workspace-url> --profile geo-embeddings
   ```
5. Check that everything works.
   ```
   python -m geo_embeddings.smoke                     # reads one embedding pixel
   databricks catalogs list --profile geo-embeddings  # lists Unity Catalog catalogs
   pytest
   ```
6. Build the Stage 2 sample (about 10 minutes) and load it into Databricks as a Delta table.
   ```
   python -m geo_embeddings.build_samples        # writes data/processed/*.parquet
   databricks/load_samples.sh                    # creates workspace.geo_embeddings.samples_pinal
   ```
7. Train and validate the classifier (Stage 3). Add `--mlflow` to log the run to Databricks.
   ```
   python -m geo_embeddings.train_baseline       # 10 km spatial block CV
   python -m geo_embeddings.train_baseline --cv random
   python -m geo_embeddings.failure_analysis
   ```
8. Build the wall-to-wall grid, change score, H3 table and search index (Stage 4).
   ```
   python -m geo_embeddings.build_grid           # about 5 minutes, writes the 200 m grid
   databricks/load_grid.sh                       # creates workspace.geo_embeddings.grid_pinal_200m
   databricks/run_sql.sh databricks/sql/create_h3_table.sql
   python -m geo_embeddings.change_analysis
   python -m geo_embeddings.h3_map
   python -c "from geo_embeddings.search import *; w = workspace_client(); ensure_endpoint(w); ensure_index(w)"
   python -m geo_embeddings.search_eval --vs     # after the index finishes syncing
   ```

## Data so far
52,000 labeled pixels from Pinal County, Arizona. For 2023 and 2024 there are 2,000 pixels for each of 13 land-cover classes grouped from the Cropland Data Layer. Each pixel has its 64 embedding values, its original CDL code and location. Only pixels inside a uniform 90 m block of one class are kept, which avoids label errors at field edges.

## Results so far (Stage 3)
A Random Forest on the 64 embedding values, validated with 5-fold cross-validation on 10 km spatial blocks so that nearby pixels never sit on both sides of the split. Intervals are 95% block-bootstrap intervals.

| | Spatial blocks (honest) | Random split |
|---|---|---|
| Overall accuracy | 0.891 (0.868 to 0.913) | 0.955 |
| Kappa | 0.882 (0.857 to 0.904) | 0.951 |

- A random split hides about 59% of the errors. The gap is largest for spatially clustered crop classes such as Double crop, Corn+Sorghum and Orchards.
- Errors follow the reference label quality. Where the 2023 CDL confidence is under 50%, the model disagrees with CDL 30% of the time. At 99 to 100% confidence it disagrees 4% of the time.
- Cotton and Developed are the most reliable classes (F1 about 0.97). Double crop is the weakest (F1 0.78) and is mostly confused with its own component crops. Orchards scores F1 0.89, but the interval runs from 0.72 to 0.97 because one farm block holds about a quarter of its samples.
- These numbers measure agreement with CDL on a class-balanced sample of pure pixels. They are not the accuracy of a wall-to-wall map.

## Results so far (Stage 4)
- **Change.** One minus the cosine similarity of a place's 2023 and 2024 embeddings is a change score. It separates crop rotations from stable land with AUC 0.98 and crop to fallow with AUC 0.97. For flips between natural CDL classes it is near chance (0.56), which suggests many of those flips are relabeling.
- **A burn scar the labels miss.** The strongest change outside farmland sits inside an area NASA MODIS maps as burned in mid 2024. The CDL still calls it shrubland.
- **Find places like this one.** A Databricks Vector Search index over 696,008 grid vectors returns places of the same CDL class for 93% of matches more than 10 km from a cotton query. It agrees with exact search on answer quality.
- **H3.** The grid rolls up to 2,455 H3 hexagons in Databricks SQL, where the change score is computed as array math and matches Python exactly.

Figures are in `docs/figures/`, metrics in `docs/results/`, and every run is logged to MLflow on Databricks. To reproduce, run `python -m geo_embeddings.train_baseline` (add `--cv random` for the comparison and `--mlflow` to log the run).

## Agents (Stage 5)
A planner agent leads three specialists built on the Claude Agent SDK. An Earth Engine agent fetches live per-place data, a Databricks agent runs read-only SQL, Vector Search and H3 hotspots, and a validator checks each claim against the stored validation results and independent data. Every tool is read-only and spend is capped per question.

[docs/examples/](docs/examples/) shows three worked questions as tool walkthroughs. They run the agents' tools in a fixed order, with no language model and no token cost, so every number there comes straight from Databricks or Earth Engine. Run them with `python -m agents.demo`. With an Anthropic API key in `.env`, `python -m agents.run "question"` runs the full team.

## Data sources
- **Google Satellite Embedding** (`GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`), licensed CC-BY 4.0. The AlphaEarth Foundations Satellite Embedding [dataset](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL) is produced by Google and Google DeepMind.
- **USDA NASS Cropland Data Layer** (`USDA/NASS/CDL`). Public domain. Thanks to USDA NASS for the CDL.
- **US Census TIGER 2018 counties** (`TIGER/2018/Counties`) for the Pinal County boundary.

## Author
Guillermo Ponce, PhD. Remote sensing data scientist, Tucson, Arizona.
