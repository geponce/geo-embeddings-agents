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
  - [x] Choose spatial block size from same-class similarity vs distance (`python -m geo_embeddings.block_size`, `docs/figures/similarity_vs_distance.png`)
  - [x] Random Forest baseline with spatial block CV, overall accuracy, per-class F1, Kappa, confusion matrix (`python -m geo_embeddings.train_baseline`)
  - [x] Same model with a random split, to show how much a random split overstates accuracy (`--cv random`, `python -m geo_embeddings.compare_cv`)
  - [x] Failure analysis (errors vs CDL confidence, confused class pairs) (`python -m geo_embeddings.failure_analysis`)
  - [ ] Log runs to MLflow on Databricks
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

- 2026-11-12: Spatial CV blocks of 10 km. Same-class embedding similarity falls steeply from about 0.98 within 100 m to a near-flat level by 3-5 km for Alfalfa, Cotton, Small grains and Fallow, and by about 10-12 km for Corn+Sorghum. That is field and farm scale autocorrelation, the leakage to block. Further drops at 10-100 km (Woody, Barren, Grass/Pasture, Shrubland, Other crops) reflect regional differences in vegetation and geology, which no block size removes. At 10 km every class still falls in at least 37 blocks (Corn+Sorghum is the smallest), enough for 5-fold CV. At 15 km Corn+Sorghum drops to 21 blocks. An automatic "range" rule was tried and dropped because the regional declines distort it.

- 2026-11-12: RF baseline (300 trees, sqrt features), 5-fold StratifiedGroupKFold on 10 km blocks (177 blocks). Pooled out-of-fold OA 0.913, Kappa 0.904, macro F1 0.913. Across folds OA 0.913 ± 0.019 (worst fold 0.879). Best F1 Cotton 0.97, Developed 0.97. Worst Other crops 0.81, Fallow/Idle 0.86, Grass/Pasture 0.86. Main confusions are Shrubland with Grass/Pasture (6% and 5%), Other crops spread across all crop classes, and Corn+Sorghum to Other crops (6%).
- 2026-11-12: Caveats on the baseline. (1) It measures agreement with CDL, not ground truth, and CDL has its own errors. (2) The sample is class-balanced and pure-pixel only, so this is not the accuracy of a wall-to-wall map. Shrubland covers about 80% of Pinal, and field edges and mixed pixels were excluded. (3) Clustering makes folds uneven for some classes (Corn+Sorghum ranges from 284 to 1494 test rows per fold).

- 2026-11-12: Random split vs 10 km spatial blocks, same RF and seed. OA 0.951 vs 0.913, Kappa 0.946 vs 0.904. The error rate rises from 4.9% to 8.7%, so a random split hides about 44% of the errors. Fold-to-fold spread is also understated (±0.003 vs ±0.019). The largest F1 gaps are the clustered crop classes, Other crops +0.12, Small grains +0.05, Corn+Sorghum +0.05. Clustering alone does not explain the gap across all classes (r = -0.23 with distinct 1 km cells per 1000 samples). Barren and Woody are as clustered but gap only +0.02, likely because they are easy to separate from every other class anyway. The gap is largest where classes are both clustered and hard to tell apart, so the model can lean on "where" instead of "what".

- 2026-11-12: Failure analysis of the spatial-CV RF. (1) Errors track CDL confidence (2023). Error rate 25.1% where CDL confidence is under 50, 19.6% at 50-69, 14.8% at 70-89, 9.9% at 90-98 and 3.4% at 99-100. This fits label noise in the reference, but low-confidence pixels may also be truly ambiguous places, so it does not prove the CDL label is wrong. (2) "Other crops" is not one thing. Pecans (4.8% error), misc vegetables (0.5%) and potatoes (7.5%) are learned well. Double crops are mostly predicted as Corn+Sorghum (Dbl Crop Oats/Corn 57% error, Triticale/Corn 56%), citrus and oranges as Woody or Shrubland, olives and sod as Alfalfa. Many of these "errors" are a class design problem, not a model failure. (3) The worst blocks are scattered, not one region. Fold 3 (12.1% error) is worst mainly because it holds one 663-pixel mixed farm block near Maricopa (lon -111.91, lat 33.07) with 35% error. Without that block fold 3 is at 10.2%. Other weak blocks are Grass/Pasture in the northeast mountains (about lon -111.3, lat 33.44) and Woody along the San Pedro side (lon -110.95, lat 32.78). (4) 2023 error 9.5%, 2024 7.9%.

## Open questions
- Regroup "Other crops"? Option: double crops to their own class, tree crops (citrus, oranges, olives, pecans) to an Orchards class. Only needs a relabel from the stored cdl_code, no new Earth Engine pull.
- Area-weighted accuracy for a real map (weight per-class accuracy by class area, or validate on a random rather than stratified sample)
- Whether to drop or down-weight low-confidence CDL labels in training
