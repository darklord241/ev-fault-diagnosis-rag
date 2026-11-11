"""Compare fixed dense, BM25, RRF, and cross-encoder retrieval methods."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Protocol

from rank_bm25 import BM25Okapi

from experiments.evaluation import (
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED,
    RETRIEVAL_METHOD_METRICS,
    bootstrap_mean_ci,
    load_evaluation_inputs,
    retrieval_method_metrics,
    tokenize_bm25,
)
from experiments.run_utils import (
    DEFAULT_RESULTS_DIR,
    REPOSITORY_ROOT,
    create_run_directory,
    file_sha256,
    write_csv,
    write_json,
    write_run_metadata,
)
from scraper.vector_search.embedding import MiniLMEmbedder
from scraper.vector_search.index_embeddings import DEFAULT_CACHE_PATH
from scraper.vector_search.retrieval import SearchResult, load_chunks
from scraper.vector_search.retriever import Retriever


SCRIPT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CASES_PATH = SCRIPT_DIR / "datasets" / "retrieval_eval_40.json"
DEFAULT_ANNOTATIONS_PATH = SCRIPT_DIR / "datasets" / "evaluation_annotations.json"
DEFAULT_CHUNKED_DIR = REPOSITORY_ROOT / "data" / "chunked"
DEFAULT_OUTPUT_DIR = DEFAULT_RESULTS_DIR / "retrieval_comparison"

EXPECTED_CHUNK_COUNT = 891
EXPECTED_QUERY_COUNT = 40
EXPECTED_SCORED_QUERY_COUNT = 39
BM25_K1 = 1.2
BM25_B = 0.75
RRF_K = 60
CANDIDATE_POOL_SIZE = 20
FINAL_TOP_K = 5
EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL_ID = "BAAI/bge-reranker-base"
RERANKER_REVISION = "main"
LATENCY_REPEATS = 5
RERANKER_BATCH_SIZE = 16

METHODS = (
    {
        "method": "R1",
        "name": "current_baseline",
        "retrieval": "production_dense_cosine",
    },
    {
        "method": "R2",
        "name": "dense_cosine",
        "retrieval": "dense_cosine",
    },
    {
        "method": "R3",
        "name": "bm25",
        "retrieval": "bm25",
        "k1": BM25_K1,
        "b": BM25_B,
    },
    {
        "method": "R4",
        "name": "dense_bm25_rrf",
        "retrieval": "reciprocal_rank_fusion",
        "rrf_k": RRF_K,
    },
    {
        "method": "R5",
        "name": "dense_cross_encoder",
        "retrieval": "dense_then_cross_encoder",
        "candidate_pool_size": CANDIDATE_POOL_SIZE,
        "final_top_k": FINAL_TOP_K,
        "reranker_model_id": RERANKER_MODEL_ID,
        "reranker_revision": RERANKER_REVISION,
    },
    {
        "method": "R6",
        "name": "dense_bm25_rrf_cross_encoder",
        "retrieval": "rrf_then_cross_encoder",
        "rrf_k": RRF_K,
        "candidate_pool_size": CANDIDATE_POOL_SIZE,
        "final_top_k": FINAL_TOP_K,
        "reranker_model_id": RERANKER_MODEL_ID,
        "reranker_revision": RERANKER_REVISION,
    },
)
METHOD_BY_ID = {str(method["method"]): method for method in METHODS}


class CrossEncoderLike(Protocol):
    def predict(
        self,
        inputs: Sequence[tuple[str, str]],
        *,
        batch_size: int,
        show_progress_bar: bool,
        apply_softmax: bool,
        convert_to_numpy: bool,
    ) -> object: ...


def build_bm25_index(records: Sequence[Mapping[str, object]]) -> BM25Okapi:
    """Build the fixed BM25 index over the unchanged corpus texts."""
    return BM25Okapi(
        [tokenize_bm25(str(record["text"])) for record in records],
        k1=BM25_K1,
        b=BM25_B,
    )


def _dense_rank(
    retriever: Retriever,
    query: str,
    *,
    top_k: int = CANDIDATE_POOL_SIZE,
) -> list[SearchResult]:
    return retriever.search(query, top_k=top_k)


def _bm25_rank(
    query: str,
    records: Sequence[Mapping[str, object]],
    index: BM25Okapi,
) -> list[SearchResult]:
    scores = index.get_scores(tokenize_bm25(query))
    if len(scores) != len(records):
        raise ValueError(
            f"BM25 returned {len(scores)} scores for {len(records)} chunks"
        )
    ranked = sorted(
        (
            (float(score), position, record)
            for position, (score, record) in enumerate(zip(scores, records))
        ),
        key=lambda item: (-item[0], item[1]),
    )
    return [
        SearchResult(score=score, record=record)
        for score, _, record in ranked
    ]


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[SearchResult]],
    records: Sequence[Mapping[str, object]],
    *,
    k: int = RRF_K,
) -> list[SearchResult]:
    """Fuse ranked lists using 1 / (k + one-based rank), with stable ties."""
    if k <= 0:
        raise ValueError("RRF k must be positive")
    record_by_id = {str(record["chunk_id"]): record for record in records}
    original_order = {
        str(record["chunk_id"]): index for index, record in enumerate(records)
    }
    fused_scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, result in enumerate(ranking, start=1):
            chunk_id = str(result.record["chunk_id"])
            if chunk_id not in record_by_id:
                raise ValueError(f"ranking contains unknown chunk ID {chunk_id!r}")
            fused_scores[chunk_id] += 1.0 / (k + rank)
    ordered_ids = sorted(
        fused_scores,
        key=lambda chunk_id: (
            -fused_scores[chunk_id],
            original_order[chunk_id],
        ),
    )
    return [
        SearchResult(fused_scores[chunk_id], record_by_id[chunk_id])
        for chunk_id in ordered_ids
    ]


def rerank_candidates(
    query: str,
    candidates: Sequence[SearchResult],
    reranker: CrossEncoderLike,
    *,
    candidate_pool_size: int = CANDIDATE_POOL_SIZE,
) -> list[SearchResult]:
    """Cross-encode at most the fixed candidate pool and return its full ranking."""
    selected = list(candidates[:candidate_pool_size])
    if not selected:
        return []
    pairs = [
        (query, str(result.record["text"]))
        for result in selected
    ]
    scores = reranker.predict(
        pairs,
        batch_size=RERANKER_BATCH_SIZE,
        show_progress_bar=False,
        apply_softmax=False,
        convert_to_numpy=True,
    )
    if hasattr(scores, "reshape"):
        scores = scores.reshape(-1)
    if hasattr(scores, "tolist"):
        scores = scores.tolist()
    if isinstance(scores, (int, float)):
        score_values = [float(scores)]
    elif isinstance(scores, Sequence) and not isinstance(
        scores, (str, bytes, bytearray)
    ):
        score_values = [float(score) for score in scores]
    else:
        raise ValueError("reranker returned scores in an unsupported format")
    if len(score_values) != len(selected):
        raise ValueError(
            f"reranker returned {len(score_values)} scores for "
            f"{len(selected)} candidates"
        )
    if not all(math.isfinite(score) for score in score_values):
        raise ValueError("reranker returned a non-finite score")
    reranked = [
        SearchResult(score, candidate.record)
        for score, candidate in zip(score_values, selected)
    ]
    return [
        result
        for _, result in sorted(
            enumerate(reranked),
            key=lambda item: (-item[1].score, item[0]),
        )
    ]


def final_results(ranking: Sequence[SearchResult]) -> list[SearchResult]:
    """Return the production-facing top-five slice without truncating metrics."""
    return list(ranking[:FINAL_TOP_K])


def _load_reranker(*, allow_download: bool) -> CrossEncoderLike:
    from sentence_transformers import CrossEncoder

    try:
        return CrossEncoder(
            RERANKER_MODEL_ID,
            revision=RERANKER_REVISION,
            local_files_only=True,
            trust_remote_code=False,
        )
    except OSError:
        if not allow_download:
            raise
    return CrossEncoder(
        RERANKER_MODEL_ID,
        revision=RERANKER_REVISION,
        local_files_only=False,
        trust_remote_code=False,
    )


def _corpus_sha256(chunked_dir: Path) -> str:
    digest = hashlib.sha256()
    paths = sorted(Path(chunked_dir).glob("*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"No JSONL chunks found in {chunked_dir}")
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as file:
            for block in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest()


def _timed_retrieval(
    operation: Callable[[], list[SearchResult]],
    *,
    repeats: int,
) -> tuple[list[SearchResult], float]:
    results: list[SearchResult] = []
    durations = []
    for _ in range(repeats):
        started = time.perf_counter()
        results = operation()
        durations.append(time.perf_counter() - started)
    return results, math.fsum(durations) / len(durations)


def _query_rankings(
    query: str,
    *,
    retriever: Retriever,
    records: Sequence[Mapping[str, object]],
    bm25_index: BM25Okapi,
    reranker: CrossEncoderLike | None,
    repeats: int,
) -> tuple[dict[str, list[SearchResult]], dict[str, float | None]]:
    dense_r1, latency_r1 = _timed_retrieval(
        lambda: _dense_rank(retriever, query),
        repeats=repeats,
    )
    dense_r2, latency_r2 = _timed_retrieval(
        lambda: _dense_rank(retriever, query),
        repeats=repeats,
    )
    bm25_top20, latency_r3 = _timed_retrieval(
        lambda: _bm25_rank(query, records, bm25_index)[:CANDIDATE_POOL_SIZE],
        repeats=repeats,
    )
    rrf_top20, latency_r4 = _timed_retrieval(
        lambda: reciprocal_rank_fusion(
            [
                _dense_rank(
                    retriever,
                    query,
                    top_k=CANDIDATE_POOL_SIZE,
                ),
                _bm25_rank(query, records, bm25_index)[:CANDIDATE_POOL_SIZE],
            ],
            records,
        )[:CANDIDATE_POOL_SIZE],
        repeats=repeats,
    )
    rankings: dict[str, list[SearchResult]] = {
        "R1": dense_r1,
        "R2": dense_r2,
        "R3": bm25_top20,
        "R4": rrf_top20,
    }
    latencies: dict[str, float | None] = {
        "R1": latency_r1,
        "R2": latency_r2,
        "R3": latency_r3,
        "R4": latency_r4,
        "R5": None,
        "R6": None,
    }
    if reranker is not None:
        r5_top20, latency_r5 = _timed_retrieval(
            lambda: rerank_candidates(
                query,
                _dense_rank(retriever, query),
                reranker,
            ),
            repeats=repeats,
        )
        r6_top20, latency_r6 = _timed_retrieval(
            lambda: rerank_candidates(
                query,
                reciprocal_rank_fusion(
                    [
                        _dense_rank(
                            retriever,
                            query,
                            top_k=CANDIDATE_POOL_SIZE,
                        ),
                        _bm25_rank(
                            query,
                            records,
                            bm25_index,
                        )[:CANDIDATE_POOL_SIZE],
                    ],
                    records,
                )[:CANDIDATE_POOL_SIZE],
                reranker,
            ),
            repeats=repeats,
        )
        rankings["R5"] = r5_top20
        rankings["R6"] = r6_top20
        latencies["R5"] = latency_r5
        latencies["R6"] = latency_r6
    return rankings, latencies


def _evidence_rows(
    case: Mapping[str, object],
    method: str,
    results: Sequence[SearchResult],
) -> list[dict[str, object]]:
    return [
        {
            "query_id": case["id"],
            "method": method,
            "rank": rank,
            "chunk_id": result.record["chunk_id"],
            "score": result.score,
            "source_document": result.record["source_document"],
            "chunk_text": result.record["text"],
            "is_final_top5": rank <= FINAL_TOP_K,
        }
        for rank, result in enumerate(results[:CANDIDATE_POOL_SIZE], start=1)
    ]


def _percentile_nearest_rank(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _aggregate_results(
    per_query_rows: Sequence[Mapping[str, object]],
    *,
    bootstrap_samples: int,
    seed: int,
    reranker_error: str | None,
) -> list[dict[str, object]]:
    aggregate_rows = []
    for method_index, method in enumerate(METHODS):
        method_id = str(method["method"])
        rows = [row for row in per_query_rows if row["method"] == method_id]
        completed_rows = [row for row in rows if row["status"] == "complete"]
        scored_rows = [
            row for row in completed_rows if row["status"] == "complete"
            and row["hit_at_1"] is not None
        ]
        latencies = [
            float(row["latency_seconds"])
            for row in completed_rows
            if row["latency_seconds"] is not None
        ]
        status = (
            "complete"
            if len(completed_rows) == EXPECTED_QUERY_COUNT
            else "blocked_reranker_unavailable"
        )
        result: dict[str, object] = {
            "method": method_id,
            "method_name": method["name"],
            "retrieval": method["retrieval"],
            "status": status,
            "reranker_error": reranker_error if status != "complete" else "",
            "n_queries": len(completed_rows),
            "n_scored": len(scored_rows),
            "n_excluded_null_gold": EXPECTED_QUERY_COUNT - len(scored_rows),
        }
        for metric in RETRIEVAL_METHOD_METRICS:
            values = [float(row[metric]) for row in scored_rows]
            low, high = bootstrap_mean_ci(
                values,
                samples=bootstrap_samples,
                seed=seed + method_index,
            )
            result[metric] = math.fsum(values) / len(values) if values else None
            result[f"{metric}_ci_low"] = low if values else None
            result[f"{metric}_ci_high"] = high if values else None
        latency_low, latency_high = bootstrap_mean_ci(
            latencies,
            samples=bootstrap_samples,
            seed=seed + method_index + 100,
        )
        result["mean_latency_seconds"] = (
            math.fsum(latencies) / len(latencies) if latencies else None
        )
        result["mean_latency_ci_low"] = latency_low if latencies else None
        result["mean_latency_ci_high"] = latency_high if latencies else None
        result["median_latency_seconds"] = (
            statistics.median(latencies) if latencies else None
        )
        result["p95_latency_seconds"] = _percentile_nearest_rank(
            latencies, 0.95
        )
        aggregate_rows.append(result)
    return aggregate_rows


def _write_report(
    path: Path,
    aggregate_rows: Sequence[Mapping[str, object]],
    per_query_rows: Sequence[Mapping[str, object]],
    *,
    corpus_count: int,
    reranker_error: str | None,
) -> None:
    lines = [
        "# Retrieval-method comparison",
        "",
        "- Embedding is fixed to `sentence-transformers/all-MiniLM-L6-v2`.",
        f"- Corpus: {corpus_count} unchanged chunks; benchmark: 40 queries, "
        "39 scored and one null-gold.",
        "- R1 and R2 both use production dense cosine without vehicle filtering; "
        "their rankings should be identical.",
        "- BM25: `rank_bm25.BM25Okapi`, `k1=1.2`, `b=0.75`.",
        "- RRF: `1 / (60 + rank)` per input ranking; no weighted score fusion.",
        "- R5/R6 rerank the top 20 candidates and evaluate the full reranked "
        "20-item list so Recall@20 is not truncated; final serving cutoff is 5.",
        "- Latency is mean per-query retrieval latency over the configured repeats; "
        "index/model setup time is excluded.",
        "",
        "## Aggregate results",
        "",
        "| Method | Status | Hit@1 | Hit@3 | Hit@5 | Recall@5 | Recall@10 | "
        "Recall@20 | MRR@5 | nDCG@10 | Mean latency (ms) | Median (ms) | "
        "P95 (ms) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    displayed_metrics = (
        ("hit_at_1", "Hit@1"),
        ("hit_at_3", "Hit@3"),
        ("hit_at_5", "Hit@5"),
        ("recall_at_5", "Recall@5"),
        ("recall_at_10", "Recall@10"),
        ("recall_at_20", "Recall@20"),
        ("mrr_at_5", "MRR@5"),
        ("ndcg_at_10", "nDCG@10"),
    )
    for row in aggregate_rows:
        values = [
            "n/a" if row[metric] is None else f"{float(row[metric]):.3f}"
            for metric, _ in displayed_metrics
        ]
        latency_values = [
            "n/a"
            if row[field] is None
            else f"{1000 * float(row[field]):.2f}"
            for field in (
                "mean_latency_seconds",
                "median_latency_seconds",
                "p95_latency_seconds",
            )
        ]
        lines.append(
            "| "
            + " | ".join(
                [str(row["method"]), str(row["status"])]
                + values
                + latency_values
            )
            + " |"
        )
    lines.extend(["", "## Method comparisons", ""])
    row_by_method = {str(row["method"]): row for row in aggregate_rows}
    r1 = row_by_method["R1"]
    r2 = row_by_method["R2"]
    lines.append(
        "- R1/R2 equivalence: "
        + (
            "identical per-query rankings and metrics."
            if all(
                r1[metric] == r2[metric]
                for metric in RETRIEVAL_METHOD_METRICS
            )
            else "aggregate values differ; inspect per-query output."
        )
    )
    for method_id, label, reference_id in (
        ("R3", "BM25 vs R1", "R1"),
        ("R4", "RRF vs R1", "R1"),
        ("R4", "RRF vs BM25", "R3"),
    ):
        row = row_by_method[method_id]
        reference = row_by_method[reference_id]
        if row["status"] != "complete" or reference["status"] != "complete":
            continue
        lines.append(
            f"- {label}: Hit@5 "
            f"{float(row['hit_at_5']):.3f} vs {float(reference['hit_at_5']):.3f}; "
            f"Recall@20 {float(row['recall_at_20']):.3f} vs "
            f"{float(reference['recall_at_20']):.3f}; nDCG@10 "
            f"{float(row['ndcg_at_10']):.3f} vs "
            f"{float(reference['ndcg_at_10']):.3f}."
        )
    for method_id, label, reference_id in (
        ("R5", "dense reranking vs R1", "R1"),
        ("R6", "RRF reranking vs R4", "R4"),
    ):
        row = row_by_method[method_id]
        reference = row_by_method[reference_id]
        if row["status"] != "complete" or reference["status"] != "complete":
            continue
        lines.append(
            f"- {label}: Hit@5 "
            f"{float(row['hit_at_5']):.3f} vs {float(reference['hit_at_5']):.3f}; "
            f"nDCG@10 {float(row['ndcg_at_10']):.3f} vs "
            f"{float(reference['ndcg_at_10']):.3f}."
        )
    available = [
        row for row in aggregate_rows
        if row["status"] == "complete"
    ]
    if available:
        best = max(
            available,
            key=lambda row: (
                float(row["ndcg_at_10"]),
                float(row["recall_at_20"]),
            ),
        )
        lines.append(
            f"- Best completed method by nDCG@10 (recall@20 tie-break): "
            f"{best['method']} ({float(best['ndcg_at_10']):.3f})."
        )
    for method_id in ("R1", "R2", "R3", "R4", "R5", "R6"):
        misses = [
            str(row["query_id"])
            for row in per_query_rows
            if row["method"] == method_id
            and row["status"] == "complete"
            and row["hit_at_5"] == 0.0
        ]
        if misses:
            lines.append(
                f"- {method_id} missed gold evidence within top 5 for: "
                f"{', '.join(misses)}."
            )
    if reranker_error:
        lines.extend(
            [
                "",
                "## Reranker availability",
                "",
                f"R5 and R6 were not run: `{reranker_error}`",
                "No substitute model was used. Re-run with "
                "`--allow-model-download` when Hugging Face access is available, "
                "or after the specified model is cached.",
            ]
        )
    lines.extend(
        [
            "",
            "Per-query metrics are in `per_query_results.csv`; ranked chunk evidence "
            "and text are in `retrieved_evidence.csv`.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare fixed retrieval methods on the 40-query benchmark."
    )
    parser.add_argument("--chunked-dir", type=Path, default=DEFAULT_CHUNKED_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-id", help="optional explicit run directory ID")
    parser.add_argument("--repeats", type=int, default=LATENCY_REPEATS)
    parser.add_argument(
        "--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES
    )
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument(
        "--allow-model-download",
        action="store_true",
        help="if the reranker is not cached, allow one Hugging Face download attempt",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.repeats <= 0 or args.bootstrap_samples <= 0:
        parser.error("--repeats and --bootstrap-samples must be positive")

    cases_path = DEFAULT_CASES_PATH
    annotations_path = DEFAULT_ANNOTATIONS_PATH
    records = load_chunks(args.chunked_dir)
    cases, annotations = load_evaluation_inputs(
        cases_path, annotations_path, records
    )
    if len(records) != EXPECTED_CHUNK_COUNT:
        raise ValueError(
            f"expected {EXPECTED_CHUNK_COUNT} chunks; found {len(records)}"
        )
    if len(cases) != EXPECTED_QUERY_COUNT:
        raise ValueError(
            f"expected {EXPECTED_QUERY_COUNT} benchmark queries; found {len(cases)}"
        )
    scored_count = sum(case["gold_chunk_ids"] is not None for case in cases)
    if scored_count != EXPECTED_SCORED_QUERY_COUNT:
        raise ValueError(
            f"expected {EXPECTED_SCORED_QUERY_COUNT} scored queries; "
            f"found {scored_count}"
        )

    embedding = MiniLMEmbedder(local_files_only=True)
    if embedding.identity.model_id != EMBEDDING_MODEL_ID:
        raise ValueError(
            f"expected embedding model {EMBEDDING_MODEL_ID}; "
            f"loaded {embedding.identity.model_id}"
        )
    retriever = Retriever(records, embedding, DEFAULT_CACHE_PATH)
    bm25_index = build_bm25_index(records)
    try:
        reranker = _load_reranker(allow_download=args.allow_model_download)
        reranker_error = None
    except OSError as error:
        reranker = None
        reranker_error = f"{type(error).__name__}: {error}"
        print(
            f"R5/R6 unavailable; required reranker could not load: {reranker_error}"
        )

    corpus_hash = _corpus_sha256(args.chunked_dir)
    run_id, output_dir = create_run_directory(args.output_dir, args.run_id)
    per_query_rows = []
    evidence_rows = []
    query_types = {
        case["id"]: annotations[case["id"]]["query_type"] for case in cases
    }
    for case in cases:
        rankings, latencies = _query_rankings(
            str(case["query"]),
            retriever=retriever,
            records=records,
            bm25_index=bm25_index,
            reranker=reranker,
            repeats=args.repeats,
        )
        r1_results = [
            (result.record["chunk_id"], result.score)
            for result in rankings["R1"]
        ]
        r2_results = [
            (result.record["chunk_id"], result.score)
            for result in rankings["R2"]
        ]
        if r1_results != r2_results:
            raise RuntimeError("R1 and R2 dense rankings are not equivalent")
        gold_ids = case["gold_chunk_ids"]
        for method in METHODS:
            method_id = str(method["method"])
            results = rankings.get(method_id)
            status = (
                "blocked_reranker_unavailable"
                if results is None
                else "complete"
            )
            metrics = (
                retrieval_method_metrics(
                    [str(result.record["chunk_id"]) for result in results],
                    gold_ids=set(gold_ids),
                )
                if results is not None and gold_ids is not None
                else None
            )
            per_query_rows.append(
                {
                    "query_id": case["id"],
                    "query_type": query_types[case["id"]],
                    "method": method_id,
                    "method_name": method["name"],
                    "status": status,
                    "n_candidates": len(results) if results is not None else None,
                    "latency_seconds": latencies[method_id],
                    "top5_chunk_ids": (
                        json.dumps(
                            [
                                result.record["chunk_id"]
                                for result in final_results(results)
                            ],
                            ensure_ascii=False,
                        )
                        if results is not None
                        else ""
                    ),
                    **{
                        metric: (
                            metrics[metric] if metrics is not None else None
                        )
                        for metric in RETRIEVAL_METHOD_METRICS
                    },
                }
            )
            if results is not None:
                evidence_rows.extend(
                    _evidence_rows(case, method_id, results)
                )

    aggregate_rows = _aggregate_results(
        per_query_rows,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        reranker_error=reranker_error,
    )
    configuration = {
        "experiment": "retrieval_comparison",
        "run_id": run_id,
        "embedding": {
            "model_id": embedding.identity.model_id,
            "revision": embedding.identity.revision,
            "preprocessing": embedding.identity.preprocessing,
            "dimension": embedding.identity.dimension,
            "cache_path": str(DEFAULT_CACHE_PATH),
        },
        "corpus": {
            "chunk_directory": str(args.chunked_dir.resolve()),
            "chunk_count": len(records),
            "sha256": corpus_hash,
        },
        "benchmark": {
            "dataset": str(cases_path.resolve()),
            "dataset_sha256": file_sha256(cases_path),
            "annotation_file": str(annotations_path.resolve()),
            "annotation_sha256": file_sha256(annotations_path),
            "query_count": len(cases),
            "scored_query_count": scored_count,
            "null_gold_query_count": len(cases) - scored_count,
        },
        "methods": list(METHODS),
        "bm25": {"implementation": "rank_bm25.BM25Okapi", "k1": BM25_K1, "b": BM25_B},
        "rrf": {"formula": "sum(1 / (k + one_based_rank))", "k": RRF_K},
        "reranker": {
            "model_id": RERANKER_MODEL_ID,
            "revision": RERANKER_REVISION,
            "candidate_pool_size": CANDIDATE_POOL_SIZE,
            "final_top_k": FINAL_TOP_K,
            "loaded": reranker is not None,
            "load_error": reranker_error,
            "download_allowed": args.allow_model_download,
        },
        "evaluation": {
            "metrics": list(RETRIEVAL_METHOD_METRICS),
            "final_top_k": FINAL_TOP_K,
            "latency_repeats": args.repeats,
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed": args.seed,
        },
    }
    config_path = output_dir / "config.json"
    write_json(config_path, configuration)
    write_csv(
        output_dir / "aggregate_results.csv",
        aggregate_rows,
        require_nonempty=True,
    )
    write_csv(
        output_dir / "per_query_results.csv",
        per_query_rows,
        require_nonempty=True,
    )
    write_csv(
        output_dir / "retrieved_evidence.csv",
        evidence_rows,
        require_nonempty=True,
    )
    _write_report(
        output_dir / "report.md",
        aggregate_rows,
        per_query_rows,
        corpus_count=len(records),
        reranker_error=reranker_error,
    )
    write_run_metadata(
        output_dir,
        run_id=run_id,
        config_path=config_path,
        input_files={
            "benchmark": cases_path,
            "annotations": annotations_path,
        },
        corpus={
            "chunk_directory": str(args.chunked_dir.resolve()),
            "chunk_count": len(records),
            "sha256": corpus_hash,
        },
        benchmark={
            "query_count": len(cases),
            "scored_query_count": scored_count,
            "metrics": list(RETRIEVAL_METHOD_METRICS),
        },
        packages=("rank-bm25", "sentence-transformers", "transformers"),
        experiment_details={
            "embedding_model": {
                "model_id": embedding.identity.model_id,
                "revision": embedding.identity.revision,
                "preprocessing": embedding.identity.preprocessing,
            },
            "reranker": configuration["reranker"],
            "bm25": configuration["bm25"],
            "rrf": configuration["rrf"],
            "cache_hit": retriever.cache_hit,
        },
    )
    print(f"Run artifacts: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
