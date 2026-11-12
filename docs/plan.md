# Plan and decisions log

## Stage checklist
- [x] 1. Setup (done 2026-11-12)
  - [x] Create conda env `geo-embeddings` from `environment.yml`
  - [x] Earth Engine auth via gcloud ADC, `init_ee()` works (2026-11-12)
  - [x] Create Databricks Free Edition workspace (2026-11-12)
  - [x] Install Databricks agent skills in Claude Code, run `/databricks:setup` and `/databricks:doctor` (2026-11-12)
  - [x] Smoke test: read one embedding pixel in Python (`python -m geo_embeddings.smoke`), list catalogs in Databricks (2026-11-12)
  - [ ] Create GitHub repo (public) when there is something worth showing (deferred)
- [ ] 2. Data (embeddings + CDL labels, one county, 1-2 years, Parquet, Delta tables)
- [ ] 3. Model and validation (MLflow, spatially separated test set, per-class metrics, failure cases)
- [ ] 4. Similarity search and change score (Vector Search, maps)
- [ ] 5. Agents (planner, EE data, Databricks analysis, validation)
- [ ] 6. Polish (README, diagram, results, limitations, demo)

## Decisions
- 2026-11-12: Project created. Start region Pinal County, Arizona (farmland and rangeland). Agents with Claude Agent SDK and Claude Opus 5.5. Databricks Free Edition.
- 2026-11-12: Databricks CLI v1.26.0 via Homebrew. Auth by OAuth browser login (U2M) into a CLI profile, token kept in the macOS keychain. The workspace has catalogs `system`, `samples` and `workspace`. Project tables will go in `workspace`.
- 2026-11-12: Package installed in editable mode (`pip install -e .`) so imports work without `PYTHONPATH`. `init_ee()` loads `.env` with python-dotenv.
- 2026-11-12: Smoke test pixel (-111.70, 32.80), 2024, returned 64 values with L2 norm 0.9991. Values repeat across bands, which suggests reduced storage precision. Fine for similarity, note it in the docs.

## Open questions
- Which years to start with (embeddings available 2017-2025)
- Which CDL classes to keep, and how to group rare ones
- Sample size per class that fits Free Edition limits
