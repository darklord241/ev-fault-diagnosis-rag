# Exact Match evaluation of Gemini with and without R6 retrieval

## Objective

Measure whether providing retrieved EV documentation improves short-answer accuracy for the same Gemini model.

## Experiment setup

- **Model.** Gemini 3.5 Flash-Lite (`gemini-3.5-flash-lite`) with temperature set to 0 in both conditions.
- **Dataset.** 12 scored questions covering nine distinct facts, plus five separate pilot questions. Questions and source-checked reference answers were derived from the EV retrieval benchmark. The pilot was excluded from the main score. See the [answer key](../../experiments/datasets/exact_match_ev.json).
- **LLM-only condition.** Gemini received the question and an instruction to return the answer value with its unit where applicable, or `UNKNOWN`.
- **RAG condition.** Gemini received the same input plus five passages. R6 combines dense MiniLM cosine search and BM25 with Reciprocal Rank Fusion, then reranks the top 20 candidates with a cross-encoder. Passages were generated afresh for all questions on the local corpus. See the [retrieved contexts](contexts.json) for passage text, IDs, corpus fingerprint, and retrieval timings.
- **Scoring.** An answer counted as correct if it matched an accepted reference after the predefined normalization. The scorer normalized case, whitespace, and supported unit formats. An `UNKNOWN` on an answerable question counted as incorrect. See the [experiment runner](../../experiments/runners/exact_match.py).
- **Latency.** LLM-only latency measures the generation request. RAG latency adds separately measured retrieval and generation times. Request pacing is excluded.

## Main results

| Condition | Correct / N | Exact Match | Median latency | P95 latency |
|---|---:|---:|---:|---:|
| Gemini only | 0 / 12 | 0.0% | 1.00 s | 9.03 s |
| Gemini + R6 top five passages | 11 / 12 | 91.7% | 3.07 s | 5.70 s |

RAG improved Exact Match by **91.7 percentage points**. The paired 95% bootstrap confidence interval was **+72.7 to +100.0 percentage points**, with resampling grouped by fact so paraphrases stayed together. Eleven answers changed from incorrect to correct and zero changed from correct to incorrect. All 24 scored requests completed without unresolved API failures. [Main summary](main-20261122-113952.summary.json) · [Per-question responses](main-20261122-113952.jsonl)

## Error analysis and limitations

Gemini-only returned `UNKNOWN` on 10 of 12 questions. The only RAG miss was the Ather 450 Apex under-seat load limit. The retrieved passages omitted the reference value of **15 kg**, and Gemini returned `UNKNOWN`.

The five-question pilot scored 0/5 for Gemini-only and 4/5 for RAG. Strict Exact Match rejected one correct acronym expansion because the combined acronym and expansion was absent from the accepted references. The main results describe a small, curated short-answer set. With 12 latency observations per condition, the reported p95 is the maximum observed value. [Pilot summary](pilot-20261122-113839.summary.json) · [Pilot responses](pilot-20261122-113839.jsonl)
