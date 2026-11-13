# Plan and decisions log

## Stage checklist
- [x] 1. Setup (done 2026-11-12)
  - [x] Create conda env `geo-embeddings` from `environment.yml`
  - [x] Earth Engine auth via gcloud ADC, `init_ee()` works (2026-11-12)
  - [x] Create Databricks Free Edition workspace (2026-11-12)
  - [x] Install Databricks agent skills in Claude Code, run `/databricks:setup` and `/databricks:doctor` (2026-11-12)
  - [x] Smoke test: read one embedding pixel in Python (`python -m geo_embeddings.smoke`), list catalogs in Databricks (2026-11-12)
  - [ ] Create GitHub repo (public) when there is something worth showing (deferred)
- [x] 2. Data (done 2026-11-12)
  - [x] Verify datasets and terms (embeddings CC-BY 4.0, CDL public domain, TIGER 2018 counties)
  - [x] Group CDL codes into 11 classes (`classes.py`)
  - [x] Stratified sample of pure pixels, 2023 and 2024, 2000 per class per year (`python -m geo_embeddings.build_samples`)
  - [x] Parquet in `data/processed/`, Delta table `workspace.geo_embeddings.samples_pinal` (`databricks/load_samples.sh`)
- [ ] 3. Model and validation (MLflow, spatially separated test set, per-class metrics, failure cases)
- [ ] 4. Similarity search and change score (Vector Search, maps)
- [ ] 5. Agents (planner, EE data, Databricks analysis, validation)
- [ ] 6. Polish (README, diagram, results, limitations, demo)

## Decisions
- 2026-11-12: Project created. Start region Pinal County, Arizona (farmland and rangeland). Agents with Claude Agent SDK and Claude Opus 5.5. Databricks Free Edition.
- 2026-11-12: Databricks CLI v1.26.0 via Homebrew. Auth by OAuth browser login (U2M) into a CLI profile, token kept in the macOS keychain. The workspace has catalogs `system`, `samples` and `workspace`. Project tables will go in `workspace`.
- 2026-11-12: Package installed in editable mode (`pip install -e .`) so imports work without `PYTHONPATH`. `init_ee()` loads `.env` with python-dotenv.
- 2026-11-12: Smoke test pixel (-111.70, 32.80), 2024, returned 64 values with L2 norm 0.9991. Values repeat across bands, which suggests reduced storage precision. Fine for similarity, note it in the docs.

- 2026-11-12: Stage 2 years 2023 and 2024. 2023 is the only recent CDL year with a per-pixel confidence band, so we can test whether model errors concentrate where CDL is unsure. CDL 2025 shows Grass/Pasture dropping from 5.8% to 1.1% of Pinal, which looks like a labeling change rather than land change.
- 2026-11-12: 11 grouped classes (Shrubland, Grass/Pasture, Alfalfa, Cotton, Corn+Sorghum, Small grains, Other crops, Fallow/Idle, Developed, Barren, Woody). Water and herbaceous wetland excluded (tiny area).
- 2026-11-12: Only "pure" pixels are sampled, where the 3x3 CDL block (90 m) is one grouped class. This removes field-edge label errors from the 30 m to 10 m mismatch.
- 2026-11-12: Two-step sampling. Stratified sample on labels at the CDL 30 m grid, then read embeddings at those points. Sampling all 64 bands at 10 m over the county timed out (over 5 min) even at 50 points per class. The two-step version took 30 s for 550 points and 9 min for 44,000.
- 2026-11-12: Removed databricks-connect from the env. Its latest release for Python 3.11 pins numpy<2, which broke pyarrow and scikit-learn. Databricks code runs in the workspace instead.
- 2026-11-12: Sample QA. Labels match CDL codes 100%. Crop classes are spatially clustered (Corn+Sorghum 2000 points fall in about 300 distinct 1 km cells, Shrubland in about 1800). A random train/test split would leak. In 2023, 25% of Fallow/Idle and 21% of Small grains labels have CDL confidence under 70%.

## Open questions
- Spatial block size for Stage 3 cross-validation (1 km cells look too small for clustered crop classes)
- Whether to drop or down-weight low-confidence CDL labels in training
