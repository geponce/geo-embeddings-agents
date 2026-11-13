# geo-embeddings-agents

**Find places like this one.** An end-to-end workflow that turns Google's Satellite Embedding dataset in Earth Engine into validated land-cover maps, similarity search and change detection on Databricks, driven by a small team of Claude agents.

> Work in progress. This README grows as each stage is completed.

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

## Data so far
52,000 labeled pixels from Pinal County, Arizona. For 2023 and 2024 there are 2,000 pixels for each of 13 land-cover classes grouped from the Cropland Data Layer. Each pixel has its 64 embedding values, its original CDL code and location. Only pixels inside a uniform 90 m block of one class are kept, which avoids label errors at field edges.

## Data sources
- **Google Satellite Embedding** (`GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`), licensed CC-BY 4.0. The AlphaEarth Foundations Satellite Embedding [dataset](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL) is produced by Google and Google DeepMind.
- **USDA NASS Cropland Data Layer** (`USDA/NASS/CDL`). Public domain. Thanks to USDA NASS for the CDL.
- **US Census TIGER 2018 counties** (`TIGER/2018/Counties`) for the Pinal County boundary.

## Author
Guillermo Ponce, PhD. Remote sensing data scientist, Tucson, Arizona.
