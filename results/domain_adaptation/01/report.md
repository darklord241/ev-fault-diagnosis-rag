# Domain adaptation evaluation: Pretrained vs Fine-Tuned MiniLM

- Run: `01`
- Objective: evaluate in-domain fine-tuning (InfoNCE / MultipleNegativesRankingLoss)   on EV technical manuals while holding corpus (891 chunks), cosine retrieval,   top-k, queries, and gold labels fixed.
- Training Data: 300 leak-free domain query-passage pairs extracted from non-gold corpus chunks.
- Validation Data: 75 leak-free domain query-passage pairs used for checkpoint selection.
- Zero Leakage: Strict verification confirmed zero overlap between training/val passages   and the 96 gold chunk citations in the 40 benchmark evaluation queries.
- Benchmark: 40 evaluation cases (20 exact-term technical queries, 20 colloquial/paraphrased queries);   39 have gold labels and are scored.
- Metrics: Hit@1/3/5, MRR@5, nDCG@5, and query latency.
- Each aggregate includes a 95% percentile bootstrap interval over scored queries.

## Overall results (39 scored queries)

| Model | Hit@1 | Hit@3 (95% CI) | Hit@5 | MRR@5 (95% CI) | nDCG@5 | Mean latency ms (95% CI) |
|---|---:|---:|---:|---:|---:|---:|
| pretrained | 0.615 | 0.718 [0.564, 0.846] | 0.744 | 0.659 [0.517, 0.795] | 0.636 | 27.71 [27.26, 28.12] |
| domain_adapted | 0.564 | 0.667 [0.513, 0.821] | 0.718 | 0.618 [0.474, 0.761] | 0.628 | 28.08 [27.89, 28.28] |

## Exact-Term Technical Queries (20 queries)

| Model | Hit@1 | Hit@3 | Hit@5 | MRR@5 | nDCG@5 |
|---|---:|---:|---:|---:|---:|
| pretrained | 0.600 | 0.750 | 0.800 | 0.668 | 0.662 |
| domain_adapted | 0.600 | 0.700 | 0.800 | 0.664 | 0.668 |

## Paraphrased / Colloquial Queries (19 scored queries)

| Model | Hit@1 | Hit@3 | Hit@5 | MRR@5 | nDCG@5 |
|---|---:|---:|---:|---:|---:|
| pretrained | 0.632 | 0.684 | 0.684 | 0.649 | 0.609 |
| domain_adapted | 0.526 | 0.632 | 0.632 | 0.570 | 0.585 |

## Model configuration

| Model | Model identifier | Revision | Dimension | Parameter Count |
|---|---|---|---:|---:|
| pretrained | sentence-transformers/all-MiniLM-L6-v2 | main | 384 | 22.7M |
| domain_adapted | experiments/models/minilm_ev_domain_adapted | local | 384 | 22.7M |

## Hit@3 changes versus pretrained baseline

- **Pretrained-only Hit@3 cases:** high-voltage-service, battery-capacity-warranty
- **Domain-adapted-only Hit@3 cases:** none

## Statistical Summary & Reproducibility
- Both models were evaluated on identical hardware, indexing the same 891 corpus chunks.
- All training and validation pairs, configurations, and evaluation metrics are preserved.
