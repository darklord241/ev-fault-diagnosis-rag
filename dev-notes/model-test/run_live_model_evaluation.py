#!/usr/bin/env python3
"""Run the shared retrieval evaluation questions through the local diagnosis API."""

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = REPOSITORY_ROOT / "scraper/vector_search/retrieval_eval.json"
DEFAULT_OUTPUT = Path("/tmp/evrag-live-model-evaluation.json")


def request_json(url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method="POST" if data is not None else "GET",
    )
    with urlopen(request, timeout=180) as response:
        return json.load(response)


def run_evaluation(
    base_url: str,
    dataset_path: Path,
    top_k: int,
) -> dict[str, Any]:
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    cases = dataset.get("cases") if isinstance(dataset, dict) else None
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"Dataset must contain a non-empty cases list: {dataset_path}")

    health = request_json(f"{base_url.rstrip('/')}/api/v1/health")
    results = []
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("query"), str):
            raise ValueError("Each evaluation case must have a string query")

        payload = {
            "query": case["query"],
            "top_k": top_k,
            "case_id": case.get("id"),
        }
        started = time.perf_counter()
        try:
            response = request_json(
                f"{base_url.rstrip('/')}/api/v1/query",
                payload,
            )
            result = {
                "case_id": case.get("id"),
                "query": case["query"],
                "latency_seconds": round(time.perf_counter() - started, 3),
                "model_used": response.get("model_used"),
                "grounding_score_self_reported": response.get("grounding_score"),
                "raw_chunks_retrieved": response.get("raw_chunks_retrieved"),
                "unique_chunks_used": response.get("unique_chunks_used"),
                "answer": response.get("direct_answer"),
                "sources": [
                    {
                        "title": source.get("title"),
                        "chunk_id": source.get("chunk_id"),
                        "page_number": source.get("page_number"),
                        "similarity_score": source.get("similarity_score"),
                    }
                    for source in response.get("sources", [])
                ],
                "error": None,
            }
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
            result = {
                "case_id": case.get("id"),
                "query": case["query"],
                "latency_seconds": round(time.perf_counter() - started, 3),
                "error": f"{type(error).__name__}: {error}",
            }
        results.append(result)
        print(
            json.dumps(
                {
                    "case_id": result["case_id"],
                    "latency_seconds": result["latency_seconds"],
                    "model_used": result.get("model_used"),
                    "error": result.get("error"),
                }
            ),
            flush=True,
        )

    successful = [item for item in results if item.get("error") is None]
    latencies = [item["latency_seconds"] for item in successful]
    grounding = [
        item["grounding_score_self_reported"]
        for item in successful
        if isinstance(item.get("grounding_score_self_reported"), (int, float))
    ]
    distinct_chunks = [
        item["unique_chunks_used"]
        for item in successful
        if isinstance(item.get("unique_chunks_used"), int)
    ]
    return {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_url": base_url,
        "dataset": str(dataset_path),
        "top_k": top_k,
        "provider": health.get("default_llm_provider"),
        "configured_model": health.get("default_llm_model"),
        "backend_retrieval_mode": health.get("backend_retrieval_mode"),
        "case_count": len(cases),
        "successful_count": len(successful),
        "failed_count": len(results) - len(successful),
        "latency_seconds": {
            "mean": round(statistics.mean(latencies), 3) if latencies else None,
            "median": round(statistics.median(latencies), 3) if latencies else None,
            "min": round(min(latencies), 3) if latencies else None,
            "max": round(max(latencies), 3) if latencies else None,
        },
        "mean_grounding_score_self_reported": (
            round(statistics.mean(grounding), 4) if grounding else None
        ),
        "mean_unique_chunks_used": (
            round(statistics.mean(distinct_chunks), 2) if distinct_chunks else None
        ),
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="backend gateway URL (default: %(default)s)",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="JSON file of shared evaluation questions",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="write the full results JSON (default: %(default)s)",
    )
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")

    try:
        evaluation = run_evaluation(args.base_url, args.dataset, args.top_k)
    except (OSError, ValueError, HTTPError, URLError, TimeoutError) as error:
        print(f"Evaluation failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(evaluation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Completed {evaluation['successful_count']}/{evaluation['case_count']} "
        f"cases; results written to {args.output}"
    )
    print(json.dumps({key: evaluation[key] for key in (
        "provider",
        "configured_model",
        "backend_retrieval_mode",
        "latency_seconds",
        "mean_grounding_score_self_reported",
        "mean_unique_chunks_used",
    )}, indent=2))
    return 0 if evaluation["failed_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
