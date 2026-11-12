# Uncertainty-Aware Hybrid Retrieval for Long-Document RAG

## Paper

**Uncertainty-Aware Hybrid Retrieval for Long-Document RAG**

Hoin Jung and Xiaoqian Wang (EMNLP 2026)

Paper: https://doi.org/10.18653/2026.emnlp-main.2104

## Why it is related

The paper argues that fine chunks can locate evidence while broader neighboring context improves coherence. Its parent-promotion variant uses fine-grained hits to return broader local context.

## What changed in this project

Added a same-document adjacent-page support score. A chunk receives a small bonus when nearby pages in the same manual are semantically compatible, promoting locally coherent evidence without changing the source text.

Implementation: `dev-notes/model-test/five_new_retrieval_experiments.py`

## Evaluation

Same 20 questions, 105 gold chunk IDs, 891 chunks, and MiniLM baseline were used. Metrics: Hit@1/5/10/20, Recall@5/10, NDCG@10, MRR@20, and latency.

## Results

| Metric | Baseline | Method | Change |
|---|---:|---:|---:|
| Hit@1 | 80.0% | 80.0% | +0.0 pp |
| Hit@5 | 85.0% | 90.0% | +5.0 pp |
| Hit@10 | 95.0% | 95.0% | +0.0 pp |
| Hit@20 | 100.0% | 100.0% | +0.0 pp |
| Recall@5 | 64.7% | 67.2% | +2.5 pp |
| NDCG@10 | 84.0% | 84.4% | +0.4 pp |
| MRR@20 | 82.9% | 83.2% | +0.4 pp |
| Mean latency | 26.9 ms | 3.6 ms | — |

## Inference

This was one of the practical wins: Hit@5 rose from 85.0% to 90.0%, Recall@5 from 64.7% to 67.2%, NDCG@10 from 84.0% to 84.4%, and MRR@20 from 82.9% to 83.3%. Mean latency fell to 3.6 ms for the reranking step.

## Files

- `paper.pdf` or `paper.md` — paper/source record
- `implementation.py` — implementation harness
- `metrics.json` — per-query and aggregate results
- `report.md` — detailed report
