# Encoder embedding model comparison

- Run: `01`
- Objective: compare embedding models while holding the existing 891 chunks,
  chunk IDs, cosine retrieval, top-k, queries, and gold labels fixed.
- Chunking is unchanged: FAQs remain single chunks; PDFs use 600-word chunks
  with 100-word overlap.
- Benchmark: 40 evaluation cases; 39 have gold labels and are scored.
- Metrics: Hit@1/3/5, MRR@5, nDCG@5, and query latency.
- Latency measures query embedding plus top-5 cosine retrieval; model loading,
  document embedding, and cache/index construction are excluded.
- Each aggregate includes a 95% percentile bootstrap interval over scored queries.
- This small benchmark is descriptive; intervals do not establish superiority.

## Overall results

| Model | Hit@1 | Hit@3 (95% CI) | Hit@5 | MRR@5 (95% CI) | nDCG@5 | Mean latency ms (95% CI) |
|---|---:|---:|---:|---:|---:|---:|
| current | 0.615 | 0.718 [0.564, 0.846] | 0.744 | 0.659 [0.517, 0.795] | 0.636 | 54.05 [51.32, 56.70] |
| bge-base | 0.641 | 0.718 [0.564, 0.846] | 0.795 | 0.691 [0.552, 0.819] | 0.640 | 98.53 [95.55, 101.63] |
| e5-base | 0.615 | 0.718 [0.564, 0.846] | 0.769 | 0.665 [0.528, 0.797] | 0.609 | 96.74 [91.78, 101.83] |

## Model configuration

| Model | Model identifier | Revision | Query prefix | Document prefix | Dimension |
|---|---|---|---|---|---:|
| current | sentence-transformers/all-MiniLM-L6-v2 | main | `` | `` | 384 |
| bge-base | BAAI/bge-base-en-v1.5 | main | `` | `` | 768 |
| e5-base | intfloat/e5-base-v2 | main | `query: ` | `passage: ` | 768 |

## Hit@3 changes versus current model

- **bge-base:** current-only Hit@3 cases: motor-kill-switch, high-voltage-service, charger-unresponsive; bge-base-only Hit@3 cases: 450x-vin-location, pillion-inflation, motor-off-control.
- **e5-base:** current-only Hit@3 cases: eight70-battery-degradation, high-voltage-service, charger-unresponsive; e5-base-only Hit@3 cases: combined-braking-system, solo-rider-inflation, find-grid-charger.

## Preliminary conclusion

Use the aggregate table and per-query CSV to identify observed differences. No model is declared statistically superior from these results alone; the paired per-query outcomes and uncertainty intervals should be reviewed before drawing broader conclusions.

Detailed query-level results are in `per_query_results.csv`; aggregate results and confidence intervals are in `aggregate_results.csv`.
