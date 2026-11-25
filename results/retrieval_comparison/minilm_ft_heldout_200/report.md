# Retrieval-method comparison

- Embedding is fixed to `C:\Users\Abhay\Desktop\Projects\ev-fault-diagnosis-rag\experiments\models\minilm_benchmark_200_grouped_70_30`.
- Corpus: 1297 passages. Benchmark: 60 queries, 60 scored.
- R1 and R2 both use production dense cosine without vehicle filtering; their rankings should be identical.
- BM25: `rank_bm25.BM25Okapi`, `k1=1.2`, `b=0.75`.
- RRF: `1 / (60 + rank)` per input ranking; no weighted score fusion.
- R5/R6 rerank the top 20 candidates and evaluate the full reranked 20-item list so Recall@20 is not truncated; final serving cutoff is 5.
- Latency is mean per-query retrieval latency over the configured repeats; index/model setup time is excluded.

## Aggregate results

| Method | Status | Hit@1 | Hit@3 | Hit@5 | Recall@5 | Recall@10 | Recall@20 | MRR@5 | nDCG@10 | Mean latency (ms) | Median (ms) | P95 (ms) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R1 | complete | 0.317 | 0.483 | 0.517 | 0.483 | 0.618 | 0.717 | 0.389 | 0.454 | 39.32 | 39.56 | 50.28 |
| R2 | complete | 0.317 | 0.483 | 0.517 | 0.483 | 0.618 | 0.717 | 0.389 | 0.454 | 39.09 | 38.42 | 48.09 |
| R3 | complete | 0.383 | 0.500 | 0.533 | 0.493 | 0.550 | 0.628 | 0.441 | 0.462 | 4.53 | 4.42 | 6.19 |
| R4 | complete | 0.333 | 0.467 | 0.517 | 0.492 | 0.689 | 0.800 | 0.397 | 0.477 | 44.67 | 43.65 | 53.57 |
| R5 | complete | 0.400 | 0.583 | 0.667 | 0.618 | 0.663 | 0.717 | 0.500 | 0.530 | 5066.20 | 5004.80 | 6371.91 |
| R6 | complete | 0.483 | 0.683 | 0.733 | 0.672 | 0.747 | 0.800 | 0.572 | 0.609 | 5279.34 | 5280.09 | 6353.75 |

## Method comparisons

- R1/R2 equivalence: identical per-query rankings and metrics.
- BM25 vs R1: Hit@5 0.533 vs 0.517; Recall@20 0.628 vs 0.717; nDCG@10 0.462 vs 0.454.
- RRF vs R1: Hit@5 0.517 vs 0.517; Recall@20 0.800 vs 0.717; nDCG@10 0.477 vs 0.454.
- RRF vs BM25: Hit@5 0.517 vs 0.533; Recall@20 0.800 vs 0.628; nDCG@10 0.477 vs 0.462.
- dense reranking vs R1: Hit@5 0.667 vs 0.517; nDCG@10 0.530 vs 0.454.
- RRF reranking vs R4: Hit@5 0.733 vs 0.517; nDCG@10 0.609 vs 0.477.
- Best completed method by nDCG@10 (recall@20 tie-break): R6 (0.609).
- R1 missed gold evidence within top 5 for: left-lever-both-wheels, battery-warranty-plain-language, river-charger-port-paraphrased, river-storage-paraphrased, tvs-regen-exact_term, nexus-modes-exact_term, nexus-battery-saver-exact_term, nexus-battery-saver-paraphrased, vida-ports-exact_term, vida-ports-paraphrased, vida-charge-safety-exact_term, vida-charge-safety-paraphrased, simple-service-exact_term, simple-service-paraphrased, simple-battery-health-exact_term, simple-battery-health-paraphrased, simple-battery-warranty-exact_term, uv-f77-bluetooth-exact_term, uv-x47-fuse-exact_term, uv-x47-fuse-paraphrased, honda-motor-cutoff-exact_term, ultra-rate-control, ultra-obd-location, vida-v2-sos-cancel, vida-v2-boost-while-cruising, x47-ballistic-unavailable, x47-hover-mode-charge, x47-motor-arm-error, x47-hill-hold-countdown.
- R2 missed gold evidence within top 5 for: left-lever-both-wheels, battery-warranty-plain-language, river-charger-port-paraphrased, river-storage-paraphrased, tvs-regen-exact_term, nexus-modes-exact_term, nexus-battery-saver-exact_term, nexus-battery-saver-paraphrased, vida-ports-exact_term, vida-ports-paraphrased, vida-charge-safety-exact_term, vida-charge-safety-paraphrased, simple-service-exact_term, simple-service-paraphrased, simple-battery-health-exact_term, simple-battery-health-paraphrased, simple-battery-warranty-exact_term, uv-f77-bluetooth-exact_term, uv-x47-fuse-exact_term, uv-x47-fuse-paraphrased, honda-motor-cutoff-exact_term, ultra-rate-control, ultra-obd-location, vida-v2-sos-cancel, vida-v2-boost-while-cruising, x47-ballistic-unavailable, x47-hover-mode-charge, x47-motor-arm-error, x47-hill-hold-countdown.
- R3 missed gold evidence within top 5 for: battery-warranty-plain-language, river-charger-port-paraphrased, river-storage-paraphrased, tvs-regen-exact_term, nexus-modes-exact_term, nexus-modes-paraphrased, nexus-battery-saver-paraphrased, vida-tyre-exact_term, vida-tyre-paraphrased, vida-ports-exact_term, vida-ports-paraphrased, simple-service-exact_term, simple-service-paraphrased, simple-battery-health-exact_term, simple-battery-health-paraphrased, simple-battery-warranty-exact_term, uv-f77-bluetooth-exact_term, uv-f77-bluetooth-paraphrased, honda-battery-temp-paraphrased, honda-motor-cutoff-paraphrased, nexus-reverse-beep, activa-stale-pairing, ultra-resume-charge, ultra-obd-location, vida-v2-cruise-after-brake, vida-v2-boost-while-cruising, x47-hover-mode-charge, x47-hill-hold-countdown.
- R4 missed gold evidence within top 5 for: left-lever-both-wheels, battery-warranty-plain-language, river-storage-paraphrased, tvs-regen-exact_term, nexus-modes-exact_term, nexus-battery-saver-paraphrased, vida-tyre-exact_term, vida-tyre-paraphrased, vida-ports-exact_term, vida-ports-paraphrased, vida-charge-safety-exact_term, vida-charge-safety-paraphrased, simple-service-exact_term, simple-service-paraphrased, simple-battery-health-exact_term, simple-battery-health-paraphrased, uv-f77-bluetooth-exact_term, uv-f77-bluetooth-paraphrased, uv-x47-fuse-exact_term, honda-battery-temp-paraphrased, honda-motor-cutoff-paraphrased, nexus-reverse-beep, ultra-obd-location, vida-v2-sos-cancel, vida-v2-boost-while-cruising, x47-ballistic-unavailable, x47-hover-mode-charge, x47-motor-arm-error, x47-hill-hold-countdown.
- R5 missed gold evidence within top 5 for: left-lever-both-wheels, river-storage-paraphrased, vida-tyre-paraphrased, vida-ports-exact_term, vida-ports-paraphrased, simple-battery-health-exact_term, simple-battery-health-paraphrased, uv-f77-bluetooth-exact_term, uv-x47-fuse-exact_term, honda-battery-temp-paraphrased, honda-motor-cutoff-paraphrased, nexus-reverse-beep, ultra-resume-charge, ultra-obd-location, vida-v2-sos-cancel, vida-v2-boost-while-cruising, x47-ballistic-unavailable, x47-hover-mode-charge, x47-motor-arm-error, x47-hill-hold-countdown.
- R6 missed gold evidence within top 5 for: river-storage-paraphrased, nexus-battery-saver-paraphrased, vida-tyre-paraphrased, vida-ports-exact_term, vida-ports-paraphrased, simple-battery-health-exact_term, simple-battery-health-paraphrased, uv-f77-bluetooth-exact_term, honda-battery-temp-paraphrased, honda-motor-cutoff-paraphrased, nexus-reverse-beep, ultra-resume-charge, ultra-obd-location, vida-v2-boost-while-cruising, x47-hover-mode-charge, x47-hill-hold-countdown.

Per-query metrics are in `per_query_results.csv`; ranked chunk evidence and text are in `retrieved_evidence.csv`.
