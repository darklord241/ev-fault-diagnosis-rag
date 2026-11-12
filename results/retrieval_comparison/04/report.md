# Retrieval-method comparison

- Embedding is fixed to `sentence-transformers/all-MiniLM-L6-v2`.
- Corpus: 891 unchanged chunks; benchmark: 40 queries, 39 scored and one null-gold.
- R1 and R2 both use production dense cosine without vehicle filtering; their rankings should be identical.
- BM25: `rank_bm25.BM25Okapi`, `k1=1.2`, `b=0.75`.
- RRF: `1 / (60 + rank)` per input ranking; no weighted score fusion.
- R5/R6 rerank the top 20 candidates and evaluate the full reranked 20-item list so Recall@20 is not truncated; final serving cutoff is 5.
- Latency is mean per-query retrieval latency over the configured repeats; index/model setup time is excluded.

## Aggregate results

| Method | Status | Hit@1 | Hit@3 | Hit@5 | Recall@5 | Recall@10 | Recall@20 | MRR@5 | nDCG@10 | Mean latency (ms) | Median (ms) | P95 (ms) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R1 | complete | 0.615 | 0.718 | 0.744 | 0.568 | 0.793 | 0.931 | 0.659 | 0.694 | 36.39 | 36.84 | 42.75 |
| R2 | complete | 0.615 | 0.718 | 0.744 | 0.568 | 0.793 | 0.931 | 0.659 | 0.694 | 34.11 | 33.68 | 41.26 |
| R3 | complete | 0.590 | 0.667 | 0.769 | 0.596 | 0.793 | 0.856 | 0.641 | 0.679 | 2.92 | 2.95 | 3.54 |
| R4 | complete | 0.641 | 0.744 | 0.872 | 0.619 | 0.883 | 0.978 | 0.718 | 0.750 | 36.18 | 34.77 | 47.12 |
| R5 | complete | 0.821 | 0.872 | 0.923 | 0.725 | 0.904 | 0.931 | 0.850 | 0.855 | 2895.10 | 3035.04 | 3726.04 |
| R6 | complete | 0.795 | 0.897 | 0.949 | 0.763 | 0.964 | 0.978 | 0.850 | 0.888 | 2827.99 | 2786.52 | 3761.06 |

## Method comparisons

- R1/R2 equivalence: identical per-query rankings and metrics.
- BM25 vs R1: Hit@5 0.769 vs 0.744; Recall@20 0.856 vs 0.931; nDCG@10 0.679 vs 0.694.
- RRF vs R1: Hit@5 0.872 vs 0.744; Recall@20 0.978 vs 0.931; nDCG@10 0.750 vs 0.694.
- RRF vs BM25: Hit@5 0.872 vs 0.769; Recall@20 0.978 vs 0.856; nDCG@10 0.750 vs 0.679.
- dense reranking vs R1: Hit@5 0.923 vs 0.744; nDCG@10 0.855 vs 0.694.
- RRF reranking vs R4: Hit@5 0.949 vs 0.872; nDCG@10 0.888 vs 0.750.
- Best completed method by nDCG@10 (recall@20 tie-break): R6 (0.888).
- R1 missed gold evidence within top 5 for: 450x-pillion-tyre-pressure, 450x-underseat-storage, 450x-vin-location, combined-braking-system, solo-rider-inflation, pillion-inflation, motor-off-control, left-lever-both-wheels, find-grid-charger, roadside-assistance-breakdown.
- R2 missed gold evidence within top 5 for: 450x-pillion-tyre-pressure, 450x-underseat-storage, 450x-vin-location, combined-braking-system, solo-rider-inflation, pillion-inflation, motor-off-control, left-lever-both-wheels, find-grid-charger, roadside-assistance-breakdown.
- R3 missed gold evidence within top 5 for: charger-no-light-prcd, 450x-pillion-tyre-pressure, charger-unresponsive, solo-rider-inflation, storage-luggage-limit, motor-off-control, app-sync-network, warning-caution-labels, battery-warranty-plain-language.
- R4 missed gold evidence within top 5 for: solo-rider-inflation, pillion-inflation, storage-luggage-limit, motor-off-control, roadside-assistance-breakdown.
- R5 missed gold evidence within top 5 for: solo-rider-inflation, motor-off-control, roadside-assistance-breakdown.
- R6 missed gold evidence within top 5 for: solo-rider-inflation, motor-off-control.

Per-query metrics are in `per_query_results.csv`; ranked chunk evidence and text are in `retrieved_evidence.csv`.
