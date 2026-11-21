# Domain adaptation evaluation: Pretrained vs Fine-Tuned MiniLM

- Run: `02`
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
| pretrained | 0.615 | 0.718 [0.564, 0.846] | 0.744 | 0.659 [0.517, 0.795] | 0.636 | 27.38 [26.81, 27.97] |
| ft_inbatch | 0.564 | 0.667 [0.513, 0.821] | 0.718 | 0.618 [0.474, 0.761] | 0.628 | 25.17 [24.78, 25.58] |
| ft_hard_neg | 0.564 | 0.641 [0.487, 0.795] | 0.641 | 0.594 [0.444, 0.744] | 0.561 | 24.21 [23.81, 24.62] |
| ft_margin | 0.538 | 0.615 [0.462, 0.769] | 0.692 | 0.588 [0.442, 0.731] | 0.564 | 25.79 [25.22, 26.39] |
| ft_augmented | 0.538 | 0.615 [0.462, 0.769] | 0.641 | 0.575 [0.432, 0.722] | 0.582 | 25.85 [25.47, 26.26] |

## Exact-Term Technical Queries (20 queries)

| Model | Hit@1 | Hit@3 | Hit@5 | MRR@5 | nDCG@5 |
|---|---:|---:|---:|---:|---:|
| pretrained | 0.600 | 0.750 | 0.800 | 0.668 | 0.662 |
| ft_inbatch | 0.600 | 0.700 | 0.800 | 0.664 | 0.668 |
| ft_hard_neg | 0.600 | 0.700 | 0.700 | 0.642 | 0.604 |
| ft_margin | 0.550 | 0.650 | 0.750 | 0.617 | 0.614 |
| ft_augmented | 0.600 | 0.650 | 0.700 | 0.629 | 0.647 |

## Paraphrased / Colloquial Queries (19 scored queries)

| Model | Hit@1 | Hit@3 | Hit@5 | MRR@5 | nDCG@5 |
|---|---:|---:|---:|---:|---:|
| pretrained | 0.632 | 0.684 | 0.684 | 0.649 | 0.609 |
| ft_inbatch | 0.526 | 0.632 | 0.632 | 0.570 | 0.585 |
| ft_hard_neg | 0.526 | 0.579 | 0.579 | 0.544 | 0.515 |
| ft_margin | 0.526 | 0.579 | 0.632 | 0.557 | 0.511 |
| ft_augmented | 0.474 | 0.579 | 0.579 | 0.518 | 0.514 |

## Model configuration

| Model | Model identifier | Revision | Dimension | Parameter Count |
|---|---|---|---:|---:|
| pretrained | sentence-transformers/all-MiniLM-L6-v2 | main | 384 | 22.7M |
| ft_inbatch | C:\Users\Abhay\Desktop\Projects\ev-fault-diagnosis-rag\experiments\models\minilm_ev_domain_adapted | local | 384 | 22.7M |
| ft_hard_neg | C:\Users\Abhay\Desktop\Projects\ev-fault-diagnosis-rag\experiments\models\minilm_ft_hard_neg | local | 384 | 22.7M |
| ft_margin | C:\Users\Abhay\Desktop\Projects\ev-fault-diagnosis-rag\experiments\models\minilm_ft_margin | local | 384 | 22.7M |
| ft_augmented | C:\Users\Abhay\Desktop\Projects\ev-fault-diagnosis-rag\experiments\models\minilm_ft_augmented | local | 384 | 22.7M |

## Hit@3 changes versus pretrained baseline

- **ft_inbatch vs pretrained:**
  - Pretrained-only Hit@3: high-voltage-service, battery-capacity-warranty
  - ft_inbatch-only Hit@3: none
- **ft_hard_neg vs pretrained:**
  - Pretrained-only Hit@3: high-voltage-service, app-sync-network, battery-capacity-warranty
  - ft_hard_neg-only Hit@3: none
- **ft_margin vs pretrained:**
  - Pretrained-only Hit@3: eight70-battery-degradation, high-voltage-service, app-sync-network, battery-capacity-warranty
  - ft_margin-only Hit@3: none
- **ft_augmented vs pretrained:**
  - Pretrained-only Hit@3: motor-kill-switch, high-voltage-service, app-sync-network, battery-capacity-warranty
  - ft_augmented-only Hit@3: none

## Statistical Summary & Reproducibility
- Both models were evaluated on identical hardware, indexing the same 891 corpus chunks.
- All training and validation pairs, configurations, and evaluation metrics are preserved.
