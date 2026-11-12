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
| R1 | complete | 0.615 | 0.718 | 0.744 | 0.568 | 0.793 | 0.931 | 0.659 | 0.694 | 32.02 | 30.44 | 47.82 |
| R2 | complete | 0.615 | 0.718 | 0.744 | 0.568 | 0.793 | 0.931 | 0.659 | 0.694 | 28.27 | 26.40 | 43.90 |
| R3 | complete | 0.590 | 0.667 | 0.769 | 0.596 | 0.793 | 0.856 | 0.641 | 0.679 | 1.47 | 1.43 | 1.87 |
| R4 | complete | 0.641 | 0.744 | 0.846 | 0.632 | 0.845 | 0.961 | 0.711 | 0.738 | 30.45 | 28.17 | 46.18 |
| R5 | blocked_reranker_unavailable | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| R6 | blocked_reranker_unavailable | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

## Method comparisons

- R1/R2 equivalence: identical per-query rankings and metrics.
- BM25 vs R1: Hit@5 0.769 vs 0.744; Recall@20 0.856 vs 0.931; nDCG@10 0.679 vs 0.694.
- RRF vs R1: Hit@5 0.846 vs 0.744; Recall@20 0.961 vs 0.931; nDCG@10 0.738 vs 0.694.
- RRF vs BM25: Hit@5 0.846 vs 0.769; Recall@20 0.961 vs 0.856; nDCG@10 0.738 vs 0.679.
- Best completed method by nDCG@10 (recall@20 tie-break): R4 (0.738).
- R1 missed gold evidence within top 5 for: 450x-pillion-tyre-pressure, 450x-underseat-storage, 450x-vin-location, combined-braking-system, solo-rider-inflation, pillion-inflation, motor-off-control, left-lever-both-wheels, find-grid-charger, roadside-assistance-breakdown.
- R2 missed gold evidence within top 5 for: 450x-pillion-tyre-pressure, 450x-underseat-storage, 450x-vin-location, combined-braking-system, solo-rider-inflation, pillion-inflation, motor-off-control, left-lever-both-wheels, find-grid-charger, roadside-assistance-breakdown.
- R3 missed gold evidence within top 5 for: charger-no-light-prcd, 450x-pillion-tyre-pressure, charger-unresponsive, solo-rider-inflation, storage-luggage-limit, motor-off-control, app-sync-network, warning-caution-labels, battery-warranty-plain-language.
- R4 missed gold evidence within top 5 for: solo-rider-inflation, pillion-inflation, storage-luggage-limit, motor-off-control, app-sync-network, roadside-assistance-breakdown.

## Reranker availability

R5 and R6 were not run: `OSError: We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.
Check your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'.`
No substitute model was used. Re-run with `--allow-model-download` when Hugging Face access is available, or after the specified model is cached.

Per-query metrics are in `per_query_results.csv`; ranked chunk evidence and text are in `retrieved_evidence.csv`.
