# Retrieval-Augmented Generation with Estimation of Source Reliability

## Paper

**Retrieval-Augmented Generation with Estimation of Source Reliability**

Jeongyeon Hwang et al. (2024/2025)

Paper: https://arxiv.org/abs/2410.22954

## Why it is related

Reliability-Aware RAG argues that semantic similarity alone ignores heterogeneous source trustworthiness and proposes reliability-aware source selection and aggregation.

## What changed in this project

Applied an explicit provenance prior to dense scores: official 1.08, technical 1.03, unknown 1.00, community 0.96. This is isolated from the application pipeline and uses the existing source_type metadata.

Implementation: `dev-notes/model-test/five_new_retrieval_experiments.py`

## Evaluation

Same 20 questions, 105 gold chunk IDs, 891 chunks, and MiniLM baseline were used. Metrics: Hit@1/5/10/20, Recall@5/10, NDCG@10, MRR@20, and latency.

## Results

| Metric | Baseline | Method | Change |
|---|---:|---:|---:|
| Hit@1 | 80.0% | 80.0% | +0.0 pp |
| Hit@5 | 85.0% | 85.0% | +0.0 pp |
| Hit@10 | 95.0% | 95.0% | +0.0 pp |
| Hit@20 | 100.0% | 100.0% | +0.0 pp |
| Recall@5 | 64.7% | 64.7% | +0.0 pp |
| NDCG@10 | 84.0% | 84.0% | +0.0 pp |
| MRR@20 | 82.9% | 82.9% | +0.0 pp |
| Mean latency | 26.9 ms | 10.4 ms | — |

## Inference

This corpus is almost entirely official manufacturer documentation, so the prior did not change aggregate metrics: Hit@5 remained 85.0%, Recall@5 64.7%, NDCG@10 84.0%, and MRR@20 82.9%. It is a useful limitation finding: source-aware ranking needs heterogeneous sources to demonstrate value.

## Files

- `paper.pdf` or `paper.md` — paper/source record
- `implementation.py` — implementation harness
- `metrics.json` — per-query and aggregate results
- `report.md` — detailed report
