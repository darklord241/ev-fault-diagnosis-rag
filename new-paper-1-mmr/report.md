# Maximal Marginal Relevance for Diversity-Based Reranking

## Paper

**Maximal Marginal Relevance for Diversity-Based Reranking**

Jaime Carbonell and Jade Goldstein (SIGIR 1998)

Paper: https://doi.org/10.1145/290941.291025

## Why it is related

MMR balances query relevance against similarity to already selected passages. It is designed to reduce repeated or near-duplicate evidence while preserving useful coverage.

## What changed in this project

Applied conservative MMR to the dense top-100 candidates with lambda=0.95. The implementation uses the existing chunk vectors and greedily selects a non-redundant top 20.

Implementation: `dev-notes/model-test/five_new_retrieval_experiments.py`

## Evaluation

Same 20 questions, 105 gold chunk IDs, 891 chunks, and MiniLM baseline were used. Metrics: Hit@1/5/10/20, Recall@5/10, NDCG@10, MRR@20, and latency.

## Results

| Metric | Baseline | Method | Change |
|---|---:|---:|---:|
| Hit@1 | 80.0% | 80.0% | +0.0 pp |
| Hit@5 | 85.0% | 80.0% | -5.0 pp |
| Hit@10 | 95.0% | 100.0% | +5.0 pp |
| Hit@20 | 100.0% | 100.0% | +0.0 pp |
| Recall@5 | 64.7% | 60.9% | -3.8 pp |
| NDCG@10 | 84.0% | 82.4% | -1.5 pp |
| MRR@20 | 82.9% | 82.9% | +0.0 pp |
| Mean latency | 26.9 ms | 181.1 ms | — |

## Inference

MMR can improve rank diversity, but exact gold-label recall did not improve here. This is a useful negative result: the benchmark labels many duplicate manual chunks as relevant, while diversity intentionally removes some duplicates.

## Files

- `paper.pdf` or `paper.md` — paper/source record
- `implementation.py` — implementation harness
- `metrics.json` — per-query and aggregate results
- `report.md` — detailed report
