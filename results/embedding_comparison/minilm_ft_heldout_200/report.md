# Encoder embedding model comparison

- Run: `minilm_ft_heldout_200`
- All models use the same 1297 passages, queries, gold labels, and cosine top-k.
- Benchmark: 60 evaluation cases, 60 scored.
- Metrics: Hit@1/3/5, MRR@5, nDCG@5, and query latency.
- Latency measures query embedding plus top-5 cosine retrieval; model loading,
  document embedding, and cache/index construction are excluded.
- Each aggregate includes a 95% percentile bootstrap interval over scored queries.
- This small benchmark is descriptive; intervals do not establish superiority.

## Overall results

| Model | Hit@1 | Hit@3 (95% CI) | Hit@5 | MRR@5 (95% CI) | nDCG@5 | Mean latency ms (95% CI) |
|---|---:|---:|---:|---:|---:|---:|
| current | 0.250 | 0.417 [0.300, 0.550] | 0.483 | 0.333 [0.233, 0.442] | 0.356 | 55.01 [52.88, 57.34] |
| fine_tuned | 0.317 | 0.483 [0.367, 0.617] | 0.517 | 0.389 [0.282, 0.501] | 0.405 | 52.17 [49.35, 55.41] |

## Model configuration

| Model | Model identifier | Revision | Query prefix | Document prefix | Dimension |
|---|---|---|---|---|---:|
| current | sentence-transformers/all-MiniLM-L6-v2 | main | `` | `` | 384 |
| fine_tuned | experiments/models/minilm_benchmark_200_grouped_70_30 | local | `` | `` | 384 |

## Hit@3 changes versus current model

- **fine_tuned:** current-only Hit@3 cases: river-charger-port-paraphrased, nexus-reverse-beep; fine_tuned-only Hit@3 cases: tvs-regen-paraphrased, nexus-modes-paraphrased, uv-f77-bluetooth-paraphrased, uv-x47-tyre-paraphrased, honda-battery-temp-paraphrased, qc1-standby-indicator.

## Preliminary conclusion

Use the aggregate table and per-query CSV to identify observed differences. No model is declared statistically superior from these results alone; the paired per-query outcomes and uncertainty intervals should be reviewed before drawing broader conclusions.

Detailed query-level results are in `per_query_results.csv`; aggregate results and confidence intervals are in `aggregate_results.csv`.
