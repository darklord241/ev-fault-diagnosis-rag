# ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT

## Paper

**ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT**

Omar Khattab and Matei Zaharia (SIGIR 2020)

Paper: https://doi.org/10.1145/3397271.3401075

## Why it is related

ColBERT delays query-document interaction until after separate encoding and uses MaxSim to preserve fine-grained token evidence.

## What changed in this project

Implemented a ColBERT-lite ablation: tokenize the query and dense top-100 passages, embed tokens with the local MiniLM model, compute mean query-token MaxSim, and blend it with the original dense score at 18%.

Implementation: `dev-notes/model-test/five_new_retrieval_experiments.py`

## Evaluation

Same 20 questions, 105 gold chunk IDs, 891 chunks, and MiniLM baseline were used. Metrics: Hit@1/5/10/20, Recall@5/10, NDCG@10, MRR@20, and latency.

## Results

| Metric | Baseline | Method | Change |
|---|---:|---:|---:|
| Hit@1 | 80.0% | 85.0% | +5.0 pp |
| Hit@5 | 85.0% | 90.0% | +5.0 pp |
| Hit@10 | 95.0% | 100.0% | +5.0 pp |
| Hit@20 | 100.0% | 100.0% | +0.0 pp |
| Recall@5 | 64.7% | 68.8% | +4.1 pp |
| NDCG@10 | 84.0% | 88.1% | +4.2 pp |
| MRR@20 | 82.9% | 87.5% | +4.6 pp |
| Mean latency | 26.9 ms | 2650.8 ms | — |

## Inference

This improved ranking quality: Hit@1 rose to 85.0%, Hit@5 to 90.0%, Recall@5 to 68.8%, NDCG@10 to 88.1%, and MRR@20 to 87.5%. The tradeoff is high prototype latency—about 2.65 seconds—because token embeddings are recomputed instead of being persisted in an optimized index.

## Files

- `paper.pdf` or `paper.md` — paper/source record
- `implementation.py` — implementation harness
- `metrics.json` — per-query and aggregate results
- `report.md` — detailed report
