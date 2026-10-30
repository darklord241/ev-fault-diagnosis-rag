# MiniLM retrieval evaluation — top-20 gold-label set

## Summary

Ran all 20 questions in `/Users/divyesamdani/Documents/work/retrieval_eval_top20.json` against the local 891-chunk corpus using the project's MiniLM cosine-similarity retriever. The dataset contains 105 gold chunk IDs, and all 105 were present in the loaded corpus.

| Metric | Result |
|---|---:|
| Hit rate @ 1 | 80.0% (16/20 queries) |
| Hit rate @ 5 | 85.0% (17/20 queries) |
| Hit rate @ 10 | 95.0% (19/20 queries) |
| Hit rate @ 20 | 100.0% (20/20 queries) |
| Mean Recall @ 5 | 64.7% |
| Mean Recall @ 10 | 92.9% |
| Mean Recall @ 20 | 100.0% |
| Mean Precision @ 5 | 66.0% |
| Mean Precision @ 10 | 47.5% |
| Mean Precision @ 20 | 26.3% |
| Mean NDCG @ 5 | 78.4% |
| Mean NDCG @ 10 | 84.0% |
| Mean NDCG @ 20 | 87.5% |
| MRR @ 20 | 82.9% |
| Query retrieval latency | mean 14.23 ms; median 14.04 ms |

Precision and recall are exact chunk-ID matches against this dataset's curated labels. Higher `k` naturally increases recall while also returning more unlabelled chunks, lowering precision.

## Per-query ranking

`First gold rank` is the rank of the first retrieved chunk whose ID is listed in that query's gold labels. `Gold found @ 5` and `Recall @ 5` show the number and fraction of that query's labeled chunk IDs found among the first five results.

| Query ID | First gold rank | Gold found @ 5 | Recall @ 5 |
|---|---:|---:|---:|
| `charging-compatible-700w` | 1 | 4/4 | 100.0% |
| `fallsafe-rider-fall` | 1 | 2/2 | 100.0% |
| `eight70-battery-degradation` | 1 | 3/3 | 100.0% |
| `charger-amber-light` | 1 | 5/7 | 71.4% |
| `charger-blinking-green` | 1 | 4/8 | 50.0% |
| `charger-solid-red` | 1 | 4/8 | 50.0% |
| `450x-solo-tyre-pressure` | 5 | 1/2 | 50.0% |
| `450x-underseat-storage` | 6 | 0/2 | 0.0% |
| `dashboard-software-version` | 1 | 4/4 | 100.0% |
| `combined-braking-system` | 12 | 0/7 | 0.0% |
| `charger-amber-confusion` | 1 | 5/8 | 62.5% |
| `charger-green-flashing-confusion` | 1 | 5/8 | 62.5% |
| `charger-red-problem` | 1 | 5/8 | 62.5% |
| `charger-unresponsive` | 1 | 2/2 | 100.0% |
| `dot-connector-wont-release` | 1 | 5/7 | 71.4% |
| `solo-rider-inflation` | 8 | 0/2 | 0.0% |
| `storage-luggage-limit` | 1 | 5/5 | 100.0% |
| `check-firmware-version` | 1 | 4/4 | 100.0% |
| `throttle-slowdown` | 1 | 3/7 | 42.9% |
| `app-sync-network` | 1 | 5/7 | 71.4% |

The three cases with no gold match in the top five were `450x-underseat-storage`, `combined-braking-system`, and `solo-rider-inflation`. Their first labeled chunks ranked 6th, 12th, and 8th respectively; all labeled IDs for all 20 queries were found within the top 20.

## Run configuration and methodology

- **Retriever/embedding model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Embedding revision:** `main` (mutable model revision)
- **Embedding dimension:** 384
- **Retrieval:** cosine similarity over the local 891-chunk corpus; top 20 returned per query
- **Dataset:** `retrieval_eval_top20.json`, version 1; status says gold evidence labels were synchronized from `retrieval_eval_40.json`
- **Evaluation:** binary exact-ID relevance; macro-averaged per-query hit rate, precision, recall, and NDCG at 1, 3, 5, 10, and 20; MRR through rank 20
- **Gold-label coverage:** 105/105 labeled chunk IDs found in the loaded corpus
- **Timing:** per-query `Retriever.search()` elapsed time after model and corpus index initialization; excludes model load, index construction, and application/API latency
- **Embedding cache:** temporary cache used for this run; persistent project cache was not changed

Recall denominator is the number of that query's gold IDs present in the corpus (all labels were present). Precision counts only explicitly labeled IDs as relevant. The dataset may label multiple equivalent chunks from duplicate manuals or FAQs; an unlisted but useful chunk is treated as non-relevant by these exact-ID metrics. Therefore, these scores describe agreement with the supplied judgments, not general answer correctness. This evaluates retrieval only—it does not run or score Gemini or Qwen answer generation.

## Interpretation

The retriever found at least one labeled passage for 17 of 20 questions in the user-facing top five, and all 20 had a labeled result by rank 20. The weakest top-five cases are clear candidates for query-specific inspection or retrieval tuning, especially braking-system operation and tyre pressure. Before treating this as a stable benchmark, review the gold labels and add independent relevance judgments for plausible useful chunks not currently labeled.

This report is local to the project folder. No files were committed or pushed.
