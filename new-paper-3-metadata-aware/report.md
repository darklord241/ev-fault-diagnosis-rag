# Utilizing Metadata for Better Retrieval-Augmented Generation

## Paper

**Utilizing Metadata for Better Retrieval-Augmented Generation**

Raquib Bin Yousuf et al. (ECIR 2026)

Paper: https://arxiv.org/abs/2601.11863

## Why it is related

The paper studies whether titles, document types, sections, company fields, and other metadata can disambiguate repetitive chunks better than text-only similarity.

## What changed in this project

Encoded the project metadata fields—title, source document, source type, document type, manufacturer, vehicle model, category, and group—and fused metadata similarity with content similarity at 8% metadata weight and 92% content weight.

Implementation: `dev-notes/model-test/five_new_retrieval_experiments.py`

## Evaluation

Same 20 questions, 105 gold chunk IDs, 891 chunks, and MiniLM baseline were used. Metrics: Hit@1/5/10/20, Recall@5/10, NDCG@10, MRR@20, and latency.

## Results

| Metric | Baseline | Method | Change |
|---|---:|---:|---:|
| Hit@1 | 80.0% | 85.0% | +5.0 pp |
| Hit@5 | 85.0% | 95.0% | +10.0 pp |
| Hit@10 | 95.0% | 95.0% | +0.0 pp |
| Hit@20 | 100.0% | 100.0% | +0.0 pp |
| Recall@5 | 64.7% | 73.1% | +8.3 pp |
| NDCG@10 | 84.0% | 86.0% | +2.0 pp |
| MRR@20 | 82.9% | 87.7% | +4.8 pp |
| Mean latency | 26.9 ms | 26.3 ms | — |

## Inference

This was the strongest balanced improvement: Hit@1 rose to 85.0%, Hit@5 to 95.0%, Recall@5 to 73.1%, NDCG@10 to 86.0%, and MRR@20 to 87.7%. Latency was 26.3 ms.

## Files

- `paper.pdf` or `paper.md` — paper/source record
- `implementation.py` — implementation harness
- `metrics.json` — per-query and aggregate results
- `report.md` — detailed report
