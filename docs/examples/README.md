# Agent tool walkthroughs

These three files show what the agent team sees when it works on a question. Each one runs the team's read-only tools in a fixed order against the real Databricks tables and public Earth Engine data. No language model is involved, so they cost nothing to run and every number comes straight from a tool. A live run (`python -m agents.run "question"`) instead lets Claude pick the tools, write the answer and have the validator agent check it. To regenerate these files, run `python -m agents.demo`.

| Question | File | What the evidence shows |
|---|---|---|
| Where else looks like this cotton field? | [cotton_similarity.md](cotton_similarity.md) | 86% of the 50 matches beyond 10 km are CDL cotton. The query node was fallow in 2023, so its change score is high (0.46), as expected for a rotation. |
| Where did land outside farms change most, and why? | [nonfarm_change.md](nonfarm_change.md) | The top hexagon has 93% of its nodes changed. MODIS maps all of a 2 km circle around its center as burned in 2024 (day of year 194). CDL calls it shrubland every year from 2020 to 2025. The single 10 m pixel at the hexagon center scores 0.124, just under the 0.14 threshold, which is a reminder to read change per area, not per pixel. |
| How reliable is the orchard class? | [orchard_reliability.md](orchard_reliability.md) | F1 is 0.887 but the 95% interval runs from 0.722 to 0.971. Orchards cover about 0.1% of the county. Search for orchards is weak (0.467 of matches beyond 10 km), compared with 0.586 when matches inside 10 km are allowed. |

The summaries in the third column were written by hand from the tool outputs. Maps are in [maps/](maps/).
