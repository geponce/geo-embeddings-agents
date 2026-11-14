# How reliable is the orchard classification?

> **Scripted snapshot, no model involved.** These are the outputs of the agent team's read-only tools, run in a fixed order by `agents/demo.py` on 2026-11-13. A live run (`python -m agents.run`) lets Claude choose the tools and write the answer, with a validator agent checking it.

## 1. validator · `validation_evidence('classification', 'Orchards')`

Random Forest on embeddings, 5-fold CV on 10 km spatial blocks, agreement with CDL on pure pixels. Overall accuracy 0.891 (95% 0.868 to 0.913). A random split would report about 0.955. Model error is 25 to 30% where 2023 CDL confidence is under 50 and 3 to 4% at 99 to 100.
| class | f1 | ci_95 | blocks |
|---|---|---|---|
| Orchards | 0.887 | 0.722 to 0.971 | 40 |

## 2. validator · `validation_evidence('search', 'Orchards')`

Share of top 50 matches beyond 10 km with the query's CDL class (30 pure queries per class, 2024). Vector Search agrees with exact search on answer quality (top-50 overlap 0.84, swaps are near-ties).
| class_name | precision_beyond_10km | precision_naive | share_of_county |
|---|---|---|---|
| Orchards | 0.467 | 0.586 | 0.001 |

## 3. databricks-analyst · `run_sql(orchard grid nodes per year)`

2 rows

| year | n_nodes | n_pure | mean_cdl_conf |
|---|---|---|---|
| 2023 | 368 | 139 | 87.6 |
| 2024 | 313 | 204 |  |
