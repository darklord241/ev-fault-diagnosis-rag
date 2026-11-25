# Baseline retrieval results

- Run: `minilm_ft_heldout_200`
- Embedding model: `C:\Users\Abhay\Desktop\Projects\ev-fault-diagnosis-rag\experiments\models\minilm_benchmark_200_grouped_70_30`
- Retrieval: dense cosine similarity, top-5
- Corpus: 1297 passages.
- Benchmark: 60 evaluation cases, 60 scored.
- Metrics: Hit@1, Hit@3, Hit@5, MRR@5, and nDCG@5.
- This run establishes the reference point for subsequent experiments.

| Metric | Score | 95% bootstrap CI |
|---|---:|---:|
| Hit@1 | 0.317 | [0.200, 0.433] |
| Hit@3 | 0.483 | [0.367, 0.617] |
| Hit@5 | 0.517 | [0.400, 0.650] |
| MRR@5 | 0.389 | [0.282, 0.501] |
| nDCG@5 | 0.405 | [0.299, 0.517] |

Per-query retrievals and scores are in `per_query_results.csv`; aggregate metrics are in `aggregate_results.csv`.
