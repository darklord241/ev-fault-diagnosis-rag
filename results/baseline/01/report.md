# Baseline retrieval results

- Run: `01`
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Retrieval: dense cosine similarity, top-5
- Corpus: the current 891 chunks
- Benchmark: 40 evaluation cases; 39 have gold labels and are scored.
- Metrics: Hit@1, Hit@3, Hit@5, MRR@5, and nDCG@5.
- This run establishes the reference point for subsequent experiments.

| Metric | Score | 95% bootstrap CI |
|---|---:|---:|
| Hit@1 | 0.615 | [0.462, 0.769] |
| Hit@3 | 0.718 | [0.564, 0.846] |
| Hit@5 | 0.744 | [0.590, 0.872] |
| MRR@5 | 0.659 | [0.517, 0.795] |
| nDCG@5 | 0.636 | [0.496, 0.767] |

Per-query retrievals and scores are in `per_query_results.csv`; aggregate metrics are in `aggregate_results.csv`.
