"""Evaluate short-answer Gemini accuracy with and without R6 evidence."""

import argparse
import json
import math
import os
import random
import re
import statistics
import time
from collections import defaultdict
from pathlib import Path

import httpx
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "experiments/datasets/exact_match_ev.json"
INSTRUCTION = "Return only the answer value with its unit where applicable, or UNKNOWN."


def normalize(value):
    value = " ".join(value.strip().casefold().rstrip(".").split())
    value = re.sub(r"(?<=\d)[ ,](?=\d{3}\b)", "", value)
    hours = re.fullmatch(r"(\d+)\s*(?:h|hr|hours?)(?:\s*(\d+)\s*(?:m|min|minutes?))?", value)
    if hours:
        return f"{int(hours[1]) * 60 + int(hours[2] or 0)} min"
    return re.sub(
        r"^(\d+)\s*(minutes?|min|km|psi|kg|l|%)$",
        lambda m: m[1] + ("%" if m[2] == "%" else " " + ("min" if m[2].startswith("min") else m[2])),
        value,
    )


def prepare_contexts(cases, records, allow_download):
    from experiments.runners import retrieval_comparison as r6
    from scraper.vector_search.embedding import MiniLMEmbedder
    from scraper.vector_search.retriever import Retriever

    retriever = Retriever(records, MiniLMEmbedder(local_files_only=True), r6.DEFAULT_CACHE_PATH)
    bm25 = r6.build_bm25_index(records)
    reranker = r6._load_reranker(allow_download=allow_download)
    contexts = {}
    for case in cases:
        query = case["query"]
        start = time.perf_counter()
        dense = retriever.search(query, r6.CANDIDATE_POOL_SIZE)
        sparse = r6._bm25_rank(query, records, bm25)[:r6.CANDIDATE_POOL_SIZE]
        fused = r6.reciprocal_rank_fusion([dense, sparse], records)[:r6.CANDIDATE_POOL_SIZE]
        ranked = r6.rerank_candidates(query, fused, reranker)[:5]
        contexts[case["id"]] = {
            "query": query,
            "retrieval_seconds": time.perf_counter() - start,
            "passages": [{"id": row.record["chunk_id"], "text": row.record["text"]} for row in ranked],
        }
    return contexts


def generate(client, model, prompt):
    start = time.perf_counter()
    for attempt in range(3):
        try:
            response = client.post(
                f"/v1beta/models/{model}:generateContent",
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": 0},
                },
            )
        except httpx.RequestError:
            error = "Gemini connection failed"
        else:
            if response.is_success:
                candidates = response.json().get("candidates", [])
                parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
                answer = "".join(part.get("text", "") for part in parts if not part.get("thought"))
                return answer, time.perf_counter() - start, None
            error = f"HTTP {response.status_code}"
            if response.status_code not in (429, 500, 502, 503, 504):
                return None, time.perf_counter() - start, error
        if attempt < 2:
            time.sleep(max(5, 2 ** attempt))
    return None, time.perf_counter() - start, error


def summarize(rows):
    pairs = defaultdict(dict)
    for row in rows:
        pairs[row["id"]][row["condition"]] = row
    complete = [
        pair for pair in pairs.values()
        if len(pair) == 2 and all(row["error"] is None for row in pair.values())
    ]
    if not complete:
        return {"paired_n": 0, "unresolved_responses": sum(bool(row["error"]) for row in rows)}

    groups = defaultdict(list)
    for pair in complete:
        groups[pair["llm_only"]["fact_id"]].append(
            int(pair["rag"]["correct"]) - int(pair["llm_only"]["correct"])
        )
    deltas_by_fact = list(groups.values())
    rng = random.Random(42)
    boot = []
    for _ in range(5000):
        sampled = [delta for group in rng.choices(deltas_by_fact, k=len(groups)) for delta in group]
        boot.append(100 * statistics.mean(sampled))
    boot.sort()
    result = {
        "paired_n": len(complete),
        "distinct_facts": len(groups),
        "unresolved_responses": sum(bool(row["error"]) for row in rows),
        "rag_minus_llm_percentage_points": 100 * sum(map(sum, deltas_by_fact)) / len(complete),
        "paired_fact_bootstrap_95_ci": [boot[125], boot[4874]],
        "wrong_to_right": sum(delta == 1 for group in deltas_by_fact for delta in group),
        "right_to_wrong": sum(delta == -1 for group in deltas_by_fact for delta in group),
    }
    for condition in ("llm_only", "rag"):
        subset = [pair[condition] for pair in complete]
        latencies = sorted(row["elapsed_seconds"] for row in subset)
        correct = sum(row["correct"] for row in subset)
        result[condition] = {
            "correct": correct,
            "em_percent": 100 * correct / len(subset),
            "median_seconds": statistics.median(latencies),
            "p95_seconds": latencies[math.ceil(0.95 * len(latencies)) - 1],
        }
    return result


def main():
    from scraper.vector_search.retrieval import load_chunks
    from experiments.runners.retrieval_comparison import _corpus_sha256

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("validate", "prepare", "pilot", "main"))
    parser.add_argument("--output", type=Path, default=ROOT / "results/exact_match")
    parser.add_argument("--model")
    parser.add_argument("--allow-model-download", action="store_true")
    args = parser.parse_args()
    cases = json.loads(DATASET.read_text(encoding="utf-8"))
    corpus_dir = ROOT / "data/chunked"
    records = load_chunks(corpus_dir)
    by_id = {r["chunk_id"]: r["text"] for r in records}
    for case in cases:
        if case["source_marker"].casefold() not in by_id.get(case["source_id"], "").casefold():
            raise ValueError(f"Source marker missing for {case['id']}")
    print(f"Validated source markers for {len(cases)} cases.")
    if args.phase == "validate":
        return
    fingerprint = _corpus_sha256(corpus_dir)
    context_file = args.output / "contexts.json"
    if args.phase == "prepare":
        contexts = prepare_contexts(cases, records, args.allow_model_download)
        args.output.mkdir(parents=True, exist_ok=True)
        context_file.write_text(json.dumps({"corpus_sha256": fingerprint, "cases": contexts}, indent=2), encoding="utf-8")
        return

    env = dotenv_values(ROOT / "agent-layer/.env")
    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or env.get("GOOGLE_API_KEY")
    if not key:
        parser.error("Gemini credentials missing from environment and agent-layer/.env")
    model = args.model or env.get("DEFAULT_LLM_MODEL")
    if not model:
        parser.error("Set --model to the Gemini model version for this run")
    prepared = json.loads(context_file.read_text(encoding="utf-8"))
    if prepared["corpus_sha256"] != fingerprint:
        parser.error("Prepared contexts belong to a different corpus. Run prepare again.")
    contexts = prepared["cases"]
    selected = [case for case in cases if case["split"] == args.phase]
    if any(contexts.get(case["id"], {}).get("query") != case["query"] for case in selected):
        parser.error("Prepared contexts do not match the current questions")

    rows = []
    run_file = args.output / f"{args.phase}-{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}.jsonl"
    next_call = 0.0
    with (
        httpx.Client(
            base_url="https://generativelanguage.googleapis.com",
            headers={"x-goog-api-key": key},
            timeout=45,
        ) as client,
        run_file.open("x", encoding="utf-8") as stream,
    ):
        for index, case in enumerate(selected):
            context = contexts[case["id"]]
            passages = context["passages"]
            prompt = f"{INSTRUCTION}\n\nQuestion: {case['query']}"
            evidence = "\n\n".join(passage["text"] for passage in passages)
            conditions = [("llm_only", prompt), ("rag", f"{prompt}\n\nRetrieved passages:\n{evidence}")]
            if index % 2:
                conditions.reverse()
            for condition, condition_prompt in conditions:
                time.sleep(max(0, next_call - time.perf_counter()))
                answer, elapsed, error = generate(client, model, condition_prompt)
                next_call = time.perf_counter() + 5
                row = {
                    **case, "model": model, "condition": condition, "raw_answer": answer, "error": error,
                    "correct": answer is not None and normalize(answer) in set(map(normalize, case["accepted"])),
                    "chunk_ids": [p["id"] for p in passages] if condition == "rag" else [],
                    "generation_seconds": elapsed,
                    "elapsed_seconds": elapsed + (context["retrieval_seconds"] if condition == "rag" else 0),
                }
                rows.append(row)
                print(json.dumps(row), file=stream, flush=True)
                if error in ("HTTP 400", "HTTP 401", "HTTP 403", "HTTP 404"):
                    raise SystemExit(f"Gemini request rejected ({error}); partial log: {run_file}")
    summary = summarize(rows)
    run_file.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
