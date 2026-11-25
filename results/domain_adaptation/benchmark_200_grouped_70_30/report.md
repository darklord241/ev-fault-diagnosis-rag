# MiniLM domain adaptation on the grouped 200-query benchmark

Pretrained MiniLM and MiniLM fine-tuned with Multiple Negatives Ranking Loss were compared on the same held-out cosine index. The corpus contains 4,322 passages. The 70% training side includes the validation subset.

| Partition | Passages | Queries | Exact-term | Paraphrased | Labeled evidence groups |
|---|---:|---:|---:|---:|---:|
| Optimizer training | 2,723 | 126 | 63 | 63 | 66 |
| Checkpoint validation | 302 | 14 | 7 | 7 | 4 |
| Held-out test | 1,297 | 60 | 30 | 30 | 35 |

The split groups source records, identical normalized passage text, shared gold passages, and shared annotated facts. Test passages were excluded from training and validation. Only gold-linked passages supplied supervised training positives. The optimizer used all 126 training queries in each of three epochs, sampling one labeled positive per query per epoch. The earliest highest-validation-MRR checkpoint was epoch 2.

| Model | Query group | N | Hit@1 | Hit@3 | Hit@5 | MRR@5 | nDCG@5 | Mean latency (ms) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| pretrained | overall | 60 | 25.0% | 41.7% | 48.3% | 0.333 | 0.356 | 5.26 |
| pretrained | exact_term | 30 | 33.3% | 46.7% | 46.7% | 0.389 | 0.396 | 5.23 |
| pretrained | paraphrased | 30 | 16.7% | 36.7% | 50.0% | 0.278 | 0.316 | 5.29 |
| fine_tuned | overall | 60 | 31.7% | 48.3% | 51.7% | 0.389 | 0.405 | 6.02 |
| fine_tuned | exact_term | 30 | 43.3% | 50.0% | 50.0% | 0.456 | 0.454 | 5.90 |
| fine_tuned | paraphrased | 30 | 20.0% | 46.7% | 53.3% | 0.322 | 0.357 | 6.13 |

Paired 95% bootstrap intervals across held-out evidence groups were -3.0 to +10.3 percentage points for the Hit@5 difference and -0.003 to +0.125 for the MRR@5 difference. Intervals use 5,000 resamples with seed 42.

Latency covers query encoding and cosine ranking on CPU with a pre-encoded test index. The evaluation used one split and 35 labeled test evidence groups. Validation contained four labeled evidence groups. Results measure the held-out partition and should be reported with that index size.

See [configuration](config.json), [split manifest](split.json), [training history](training_history.json), and [per-query results](per_query_results.csv).
