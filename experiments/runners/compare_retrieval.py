"""Run dense/BM25/hybrid retrieval and embedding/chunk-size evaluations."""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Sequence

from rank_bm25 import BM25Okapi

from experiments.evaluation import (
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED,
    METRICS,
    QUERY_TYPES,
    TOP_K,
    EvaluationInputError,
    bootstrap_mean_ci,
    fuse_scores,
    load_evaluation_inputs,
    normalize_text,
    ranking_metrics,
    tokenize_bm25,
)
from experiments.run_utils import (
    DEFAULT_RESULTS_DIR,
    REPOSITORY_ROOT,
    create_run_directory,
    file_sha256,
    git_commit,
    write_csv,
    write_json,
)
from scraper.vector_search.embedding import (
    EmbeddingIdentity,
    MiniLMEmbedder,
    embed_query,
)
from scraper.vector_search.index_embeddings import (
    DEFAULT_CACHE_PATH,
    load_or_build_embeddings,
)
from scraper.vector_search.retrieval import (
    CHUNKED_DIR,
    CosineIndex,
    SearchResult,
    _validate_chunk,
    load_chunks,
)


SCRIPT_DIR = Path(__file__).resolve().parents[1]
SCRAPER_DIR = REPOSITORY_ROOT / "scraper"
DEFAULT_CASES_PATH = SCRIPT_DIR / "datasets" / "retrieval_eval_40.json"
DEFAULT_ANNOTATIONS_PATH = SCRIPT_DIR / "datasets" / "evaluation_annotations.json"
DEFAULT_PROCESSED_DIR = SCRAPER_DIR / "data" / "processed"
DEFAULT_OUTPUT_DIR = DEFAULT_RESULTS_DIR / "retrieval_comparison"
CHUNK_SIZES = (128, 256)
HYBRID_ALPHAS = (0.3, 0.5, 0.7, 0.9)
LATENCY_REPEATS = 5
BM25_K1 = 1.2
BM25_B = 0.75
def _vehicle_matches(
    record: Mapping[str, object], requested_model: object
) -> bool:
    if not isinstance(requested_model, str) or not requested_model.strip():
        return True
    provenance = record.get("provenance")
    metadata = record.get("metadata")
    provenance = provenance if isinstance(provenance, Mapping) else {}
    metadata = metadata if isinstance(metadata, Mapping) else {}
    document_model = provenance.get("vehicle_model") or metadata.get("vehicle_model")
    if not isinstance(document_model, str) or not document_model.strip():
        return True
    normalized_document = document_model.casefold()
    normalized_request = requested_model.casefold()
    return (
        normalized_request in normalized_document
        or normalized_document in normalized_request
        or any(
            generic in normalized_document
            for generic in ("generic", "all models", "all vehicles", "standard")
        )
    )


def _rank_dense(
    query: str,
    backend: MiniLMEmbedder | "LocalSentenceTransformerBackend",
    index: CosineIndex,
    records: Sequence[Mapping[str, object]],
    requested_model: object,
) -> list[SearchResult]:
    vector = embed_query(backend, query)
    return index.search(
        vector,
        top_k=len(records),
        record_filter=lambda record: _vehicle_matches(record, requested_model),
    )


def _rank_bm25(
    query: str,
    records: Sequence[Mapping[str, object]],
    index: BM25Okapi,
    requested_model: object,
) -> list[SearchResult]:
    eligible_positions = [
        position for position, record in enumerate(records)
        if _vehicle_matches(record, requested_model)
    ]
    if not eligible_positions:
        return []
    scores = index.get_scores(tokenize_bm25(query))
    ranked = sorted(
        (
            (float(scores[position]), position, records[position])
            for position in eligible_positions
        ),
        key=lambda item: (-item[0], item[1]),
    )
    return [
        SearchResult(score=score, record=record)
        for score, _, record in ranked
    ]


def _rank_hybrid(
    dense_results: Sequence[SearchResult],
    bm25_results: Sequence[SearchResult],
    alpha: float,
    records: Sequence[Mapping[str, object]],
) -> list[SearchResult]:
    dense_by_id = {result.record["chunk_id"]: result.score for result in dense_results}
    bm25_by_id = {result.record["chunk_id"]: result.score for result in bm25_results}
    record_by_id = {
        record["chunk_id"]: record
        for record in records
        if record["chunk_id"] in dense_by_id and record["chunk_id"] in bm25_by_id
    }
    ordered_ids = list(record_by_id)
    dense = [dense_by_id[chunk_id] for chunk_id in ordered_ids]
    sparse = [bm25_by_id[chunk_id] for chunk_id in ordered_ids]
    combined = fuse_scores(dense, sparse, alpha)
    scored = sorted(
        enumerate(zip(ordered_ids, combined)),
        key=lambda item: (-item[1][1], item[0]),
    )
    return [
        SearchResult(score=score, record=record_by_id[chunk_id])
        for _, (chunk_id, score) in scored
    ]


def _build_bm25_index(
    records: Sequence[Mapping[str, object]],
) -> BM25Okapi:
    return BM25Okapi(
        [tokenize_bm25(str(record["text"])) for record in records],
        k1=BM25_K1,
        b=BM25_B,
    )


def _query_groups(
    cases: Sequence[dict], annotations: Mapping[str, Mapping[str, object]]
) -> dict[str, list[dict]]:
    groups = {"overall": list(cases)}
    for query_type in QUERY_TYPES:
        groups[query_type] = [
            case for case in cases
            if annotations[case["id"]]["query_type"] == query_type
        ]
    return groups


def _metric_row(
    *,
    config: Mapping[str, object],
    query_type: str,
    query_results: Mapping[
        str, tuple[Mapping[str, float | int | None], float]
    ],
    total_cases: int,
    excluded_null_gold: int,
    repeats: int,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, object]:
    row: dict[str, object] = {
        **config,
        "query_type": query_type,
        "n_total_cases": total_cases,
        "n_scored": len(query_results),
        "n_excluded_null_gold": excluded_null_gold,
        "latency_repeats": repeats,
        "bootstrap_samples": bootstrap_samples,
        "status": config.get("status", "complete"),
    }
    for metric in METRICS:
        values = [metrics[metric] for metrics, _ in query_results.values()]
        low, high = bootstrap_mean_ci(
            values, samples=bootstrap_samples, seed=seed
        )
        row[metric] = math.fsum(values) / len(values) if values else None
        row[f"{metric}_ci_low"] = low if values else None
        row[f"{metric}_ci_high"] = high if values else None

    latencies = [latency for _, latency in query_results.values()]
    low, high = bootstrap_mean_ci(
        latencies, samples=bootstrap_samples, seed=seed + 1
    )
    row["mean_query_latency_seconds"] = (
        math.fsum(latencies) / len(latencies) if latencies else None
    )
    row["mean_query_latency_ci_low"] = low if latencies else None
    row["mean_query_latency_ci_high"] = high if latencies else None
    return row


def _per_query_row(
    *,
    case: Mapping[str, object],
    config: Mapping[str, object],
    query_type: str,
    metrics: Mapping[str, float | int | None] | None,
    latency: float | None,
    status: str,
) -> dict[str, object]:
    return {
        "query_id": case["id"],
        "config": config["config_name"],
        "query_type": query_type,
        "status": status,
        "first_relevant_rank": (
            metrics["first_relevant_rank"] if metrics is not None else None
        ),
        "hit_at_1": metrics["hit_at_1"] if metrics is not None else None,
        "hit_at_3": metrics["hit_at_3"] if metrics is not None else None,
        "hit_at_5": metrics["hit_at_5"] if metrics is not None else None,
        "reciprocal_rank": metrics["mrr"] if metrics is not None else None,
        "ndcg_at_5": metrics["ndcg_at_5"] if metrics is not None else None,
        "latency_seconds": latency,
    }


def _evaluate_search_config(
    *,
    config: Mapping[str, object],
    cases: Sequence[dict],
    annotations: Mapping[str, Mapping[str, object]],
    records: Sequence[Mapping[str, object]],
    dense_index: CosineIndex,
    dense_backend: MiniLMEmbedder | "LocalSentenceTransformerBackend",
    bm25_indexes: Mapping[str, BM25Okapi],
    alpha: float,
    repeats: int,
    bootstrap_samples: int,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    query_results: dict[
        str, tuple[dict[str, float | int | None], float]
    ] = {}
    for case in cases:
        if case["gold_chunk_ids"] is None:
            continue
        requested_model = case.get("vehicle_model")
        bm25_index = bm25_indexes[""]
        rankings = []
        latency_samples = []
        for _ in range(repeats):
            started = time.perf_counter()
            dense_results = (
                _rank_dense(
                    case["query"], dense_backend, dense_index, records,
                    requested_model,
                )
                if alpha > 0.0
                else []
            )
            sparse_results = (
                _rank_bm25(
                    case["query"], records, bm25_index, requested_model
                )
                if alpha < 1.0
                else []
            )
            if alpha == 1.0:
                ranked = dense_results
            elif alpha == 0.0:
                ranked = sparse_results
            else:
                ranked = _rank_hybrid(
                    dense_results, sparse_results, alpha, records
                )
            latency_samples.append(time.perf_counter() - started)
            rankings.append(ranked)

        query_results[case["id"]] = (
            ranking_metrics(
                [str(result.record["chunk_id"]) for result in rankings[-1]],
                gold_ids=set(case["gold_chunk_ids"]),
            ),
            math.fsum(latency_samples) / len(latency_samples),
        )

    per_query_rows = [
        _per_query_row(
            case=case,
            config=config,
            query_type=str(annotations[case["id"]]["query_type"]),
            metrics=(
                query_results[case["id"]][0]
                if case["id"] in query_results else None
            ),
            latency=(
                query_results[case["id"]][1]
                if case["id"] in query_results else None
            ),
            status=(
                "complete"
                if case["id"] in query_results
                else "excluded_null_gold"
            ),
        )
        for case in cases
    ]
    rows = []
    groups = _query_groups(cases, annotations)
    for split_name, group in groups.items():
        group_results = {
            case["id"]: query_results[case["id"]]
            for case in group
            if case["gold_chunk_ids"] is not None
        }
        rows.append(
            _metric_row(
                config=config,
                query_type=split_name,
                query_results=group_results,
                total_cases=len(group),
                excluded_null_gold=sum(
                    case["gold_chunk_ids"] is None for case in group
                ),
                repeats=repeats,
                bootstrap_samples=bootstrap_samples,
                seed=seed,
            )
        )
    return rows, per_query_rows


def _copy_reusable_embedding_cache(
    reuse_roots: Sequence[Path],
    candidate_path: Callable[[Path], Path],
    destination: Path,
) -> Path | None:
    if destination.exists():
        return None
    for root in reuse_roots:
        source = candidate_path(root)
        if source.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            return source
    return None


def _load_reusable_chunks(
    reuse_roots: Sequence[Path],
    model_slug: str,
    token_limit: int,
    expected_chunks: Sequence[Mapping[str, object]],
) -> tuple[list[dict] | None, Path | None]:
    expected_by_id = {
        str(chunk["chunk_id"]): json.dumps(
            chunk, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        for chunk in expected_chunks
    }
    for root in reuse_roots:
        source = root / "experiment_3" / "chunks" / f"{model_slug}_{token_limit}"
        if not source.is_dir():
            continue
        try:
            cached_chunks = load_chunks(source)
        except (OSError, ValueError) as error:
            print(f"Skipping invalid reusable chunks at {source}: {error}", file=sys.stderr)
            continue
        cached_by_id = {
            str(chunk["chunk_id"]): json.dumps(
                chunk, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
            for chunk in cached_chunks
        }
        if cached_by_id == expected_by_id:
            ordered_cache = {str(chunk["chunk_id"]): chunk for chunk in cached_chunks}
            return (
                [ordered_cache[str(chunk["chunk_id"])] for chunk in expected_chunks],
                source,
            )
        print(
            f"Reusable chunks do not match current inputs; rebuilding: {source}",
            file=sys.stderr,
        )
    return None, None


class LocalSentenceTransformerBackend:
    """Model-neutral local Sentence Transformers adapter for Experiment 3."""

    def __init__(
        self,
        model_id: str,
        *,
        query_prefix: str = "",
        allow_download: bool = False,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError(
                "Experiment 3 requires sentence-transformers; install the "
                "evaluation requirements and model dependency first."
            ) from error

        self.model = SentenceTransformer(
            model_id,
            device="cpu",
            local_files_only=not allow_download,
        )
        self.tokenizer = self.model.tokenizer
        self.max_seq_length = self.model.max_seq_length
        dimension = self.model.get_sentence_embedding_dimension()
        if not isinstance(dimension, int) or dimension <= 0:
            raise ValueError(f"{model_id} did not expose a valid dimension")
        self.query_prefix = query_prefix
        self.identity = EmbeddingIdentity(
            model_id=model_id,
            revision="local",
            preprocessing=(
                f"sentence-transformers; query_prefix={query_prefix!r}; "
                f"max_seq_length={self.max_seq_length}; reject overlong inputs"
            ),
            dimension=dimension,
        )

    def token_count(self, text: str) -> int:
        return len(
            self.tokenizer.encode(
                text, add_special_tokens=True, truncation=False
            )
        )

    def _encode(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        if any(self.token_count(text) > self.max_seq_length for text in texts):
            raise ValueError("embedding input exceeds the model token limit")
        vectors = self.model.encode(
            list(texts),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        return [tuple(float(value) for value in vector) for vector in vectors]

    def embed(self, text: str) -> Sequence[float]:
        return self._encode([text])[0]

    def embed_query(self, text: str) -> Sequence[float]:
        return self.embed(self.query_prefix + text)

    def embed_documents(
        self, texts: Sequence[str]
    ) -> Sequence[Sequence[float]]:
        return self._encode(texts)


def split_record_by_tokens(
    record: Mapping[str, object],
    tokenizer,
    token_limit: int,
) -> list[dict]:
    """Split a processed record into non-overlapping tokenizer-aligned chunks."""
    if isinstance(token_limit, bool) or not isinstance(token_limit, int):
        raise TypeError("token_limit must be an integer")
    special_tokens = tokenizer.num_special_tokens_to_add(pair=False)
    payload_limit = token_limit - special_tokens
    if payload_limit <= 0:
        raise ValueError("token_limit is too small for model special tokens")

    text = record.get("text")
    record_id = record.get("record_id")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("processed record text must not be empty")
    if not isinstance(record_id, str) or not record_id.strip():
        raise ValueError("processed record record_id must not be empty")

    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
        verbose=False,
    )
    offsets = encoded["offset_mapping"]
    chunks = []
    start = 0
    while start < len(offsets):
        end = min(start + payload_limit, len(offsets))
        segment = ""
        while end > start:
            segment = text[offsets[start][0]:offsets[end - 1][1]].strip()
            token_count = len(
                tokenizer.encode(
                    segment,
                    add_special_tokens=True,
                    truncation=False,
                )
            )
            if token_count <= token_limit:
                break
            end -= 1
        if end == start:
            raise EvaluationInputError(
                f"could not fit a token from processed record {record_id!r} "
                f"within the {token_limit}-token limit"
            )
        if segment:
            chunk_index = len(chunks)
            chunk = {
                "chunk_id": f"{record_id}_chunk_{chunk_index}",
                "record_id": record_id,
                "source_document": record["source_document"],
                "source_type": record["source_type"],
                "content_type": record["content_type"],
                "title": record["title"],
                "text": segment,
                "provenance": record["provenance"],
                "metadata": record["metadata"],
            }
            _validate_chunk(
                chunk, f"processed record {record_id} chunk {chunk_index}"
            )
            chunks.append(chunk)
        start = end
    return chunks


def load_processed_records(directory: Path) -> list[dict]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Processed-record directory does not exist: {directory}")
    records = []
    seen_ids = set()
    for path in sorted(directory.glob("*.jsonl")):
        with path.open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise EvaluationInputError(
                        f"{path}:{line_number}: invalid JSON: {error.msg}"
                    ) from error
                if not isinstance(record, dict):
                    raise EvaluationInputError(
                        f"{path}:{line_number}: expected a JSON object"
                    )
                record_id = record.get("record_id")
                if not isinstance(record_id, str) or record_id in seen_ids:
                    raise EvaluationInputError(
                        f"{path}:{line_number}: missing or duplicate record_id"
                    )
                seen_ids.add(record_id)
                records.append(record)
    return records


def exclude_empty_processed_records(
    records: Sequence[Mapping[str, object]],
) -> tuple[list[Mapping[str, object]], list[str]]:
    usable = []
    excluded_ids = []
    for record in records:
        text = record.get("text")
        record_id = record.get("record_id")
        if not isinstance(text, str):
            raise EvaluationInputError(
                f"processed record {record_id!r} has non-string text"
            )
        if not text.strip():
            if not isinstance(record_id, str) or not record_id:
                raise EvaluationInputError(
                    "processed record with empty text has no valid record_id"
                )
            excluded_ids.append(record_id)
        else:
            usable.append(record)
    return usable, excluded_ids


def build_token_chunks(
    processed_records: Sequence[Mapping[str, object]],
    tokenizer,
    token_limit: int,
) -> tuple[list[dict], list[int]]:
    chunks = []
    for record in processed_records:
        chunks.extend(split_record_by_tokens(record, tokenizer, token_limit))
    chunk_ids = [chunk["chunk_id"] for chunk in chunks]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise EvaluationInputError("re-chunking produced duplicate chunk IDs")
    token_counts = [
        len(tokenizer.encode(text, add_special_tokens=True, truncation=False))
        for text in (chunk["text"] for chunk in chunks)
    ]
    if any(count > token_limit for count in token_counts):
        raise EvaluationInputError("re-chunked text exceeds the requested token limit")
    return chunks, token_counts


def _tokenizer_for_backend(backend):
    tokenizer = getattr(backend, "tokenizer", None)
    if tokenizer is not None:
        return tokenizer
    tokenizer = getattr(backend, "_tokenizer", None)
    if tokenizer is not None:
        return tokenizer
    raise ValueError("embedding backend does not expose a tokenizer")


def _write_chunks(chunks: Sequence[Mapping[str, object]], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    grouped = defaultdict(list)
    for chunk in chunks:
        grouped[str(chunk["source_document"])].append(chunk)
    for source_document, source_chunks in grouped.items():
        path = directory / f"{source_document}.jsonl"
        with path.open("w", encoding="utf-8") as file:
            for chunk in source_chunks:
                file.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def _eligible_records(
    records: Sequence[Mapping[str, object]], requested_model: object
) -> list[Mapping[str, object]]:
    return [
        record for record in records
        if _vehicle_matches(record, requested_model)
    ]


def _score_fact_rankings(
    rankings: Sequence[SearchResult],
    fact_phrases: Sequence[str],
    total_relevant: int,
) -> dict[str, float | int | None]:
    ranked_ids = [str(result.record["chunk_id"]) for result in rankings]
    text_by_id = {
        str(result.record["chunk_id"]): str(result.record["text"])
        for result in rankings
    }
    return ranking_metrics(
        ranked_ids,
        fact_phrases=fact_phrases,
        text_by_id=text_by_id,
        total_relevant=total_relevant,
    )


def _evaluate_fact_config(
    *,
    config: Mapping[str, object],
    cases: Sequence[dict],
    annotations: Mapping[str, Mapping[str, object]],
    records: Sequence[Mapping[str, object]],
    index: CosineIndex,
    backend: MiniLMEmbedder | LocalSentenceTransformerBackend,
    repeats: int,
    bootstrap_samples: int,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    query_results: dict[
        str, tuple[dict[str, float | int | None], float]
    ] = {}
    for case in cases:
        facts = annotations[case["id"]]["key_facts"]
        if case["gold_chunk_ids"] is None:
            continue
        eligible = _eligible_records(records, case.get("vehicle_model"))
        total_relevant = sum(
            any(
                normalize_text(fact) in normalize_text(str(record["text"]))
                for fact in facts
            )
            for record in eligible
        )
        timings = []
        rankings = []
        for _ in range(repeats):
            started = time.perf_counter()
            vector = embed_query(backend, case["query"])
            results = index.search(
                vector,
                top_k=TOP_K,
                record_filter=lambda record: _vehicle_matches(
                    record, case.get("vehicle_model")
                ),
            )
            timings.append(time.perf_counter() - started)
            rankings.append(results)
        query_results[case["id"]] = (
            _score_fact_rankings(rankings[-1], facts, total_relevant),
            math.fsum(timings) / len(timings),
        )

    per_query_rows = [
        _per_query_row(
            case=case,
            config=config,
            query_type=str(annotations[case["id"]]["query_type"]),
            metrics=(
                query_results[case["id"]][0]
                if case["id"] in query_results else None
            ),
            latency=(
                query_results[case["id"]][1]
                if case["id"] in query_results else None
            ),
            status=(
                "complete"
                if case["id"] in query_results
                else "excluded_null_gold"
            ),
        )
        for case in cases
    ]
    rows = []
    groups = _query_groups(cases, annotations)
    for split_name, group in groups.items():
        group_results = {
            case["id"]: query_results[case["id"]]
            for case in group
            if case["id"] in query_results
        }
        rows.append(
            _metric_row(
                config=config,
                query_type=split_name,
                query_results=group_results,
                total_cases=len(group),
                excluded_null_gold=sum(
                    case["gold_chunk_ids"] is None for case in group
                ),
                repeats=repeats,
                bootstrap_samples=bootstrap_samples,
                seed=seed,
            )
        )
    return rows, per_query_rows


def _experiment2(
    *,
    cases: Sequence[dict],
    annotations: Mapping[str, Mapping[str, object]],
    records: Sequence[Mapping[str, object]],
    backend: MiniLMEmbedder,
    reuse_roots: Sequence[Path],
    output_dir: Path,
    repeats: int,
    bootstrap_samples: int,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    output_cache_path = (
        DEFAULT_CACHE_PATH.parent / "experiments" / "phase2_dense_embeddings.json"
    )
    _copy_reusable_embedding_cache(
        reuse_roots,
        lambda root: root / "experiment_2" / "dense_embeddings.json",
        output_cache_path,
    )
    cache = load_or_build_embeddings(
        records, backend, output_cache_path
    )
    dense_index = CosineIndex(
        records,
        [
            cache.vectors_by_chunk_id[str(record["chunk_id"])]
            for record in records
        ],
    )
    bm25_indexes = {"": _build_bm25_index(records)}

    configs = [
        ("2a_dense", "dense", 1.0),
        ("2b_bm25", "bm25", 0.0),
        *[
            (f"2c_hybrid_alpha_{alpha:.1f}", "hybrid", alpha)
            for alpha in HYBRID_ALPHAS
        ],
    ]
    rows = []
    per_query_rows = []
    for name, method, alpha in configs:
        config = {
            "config_name": name,
            "method": method,
            "alpha": alpha,
            "k1": BM25_K1 if method != "dense" else None,
            "b": BM25_B if method != "dense" else None,
            "model_id": backend.identity.model_id,
            "chunk_size_tokens": None,
            "status": "complete",
        }
        config_rows, config_query_rows = _evaluate_search_config(
            config=config,
            cases=cases,
            annotations=annotations,
            records=records,
            dense_index=dense_index,
            dense_backend=backend,
            bm25_indexes=bm25_indexes,
            alpha=alpha,
            repeats=repeats,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        )
        rows.extend(config_rows)
        per_query_rows.extend(config_query_rows)
    return rows, per_query_rows


def _load_generic_model(
    model_id: str,
    query_prefix: str,
    allow_download: bool,
) -> LocalSentenceTransformerBackend | None:
    try:
        return LocalSentenceTransformerBackend(
            model_id,
            query_prefix=query_prefix,
            allow_download=allow_download,
        )
    except OSError as error:
        if allow_download:
            raise
        print(
            f"Optional model unavailable locally; skipping it: {model_id} ({error})",
            file=sys.stderr,
        )
        return None


def _experiment3(
    *,
    cases: Sequence[dict],
    annotations: Mapping[str, Mapping[str, object]],
    processed_records: Sequence[Mapping[str, object]],
    baseline_backend: MiniLMEmbedder,
    alternative_model: str,
    alternative_query_prefix: str,
    allow_download: bool,
    empty_processed_records_excluded: int,
    reuse_roots: Sequence[Path],
    output_dir: Path,
    repeats: int,
    bootstrap_samples: int,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    run_directory = output_dir
    chunk_root = run_directory / "chunks"
    cache_root = DEFAULT_CACHE_PATH.parent / "experiments" / "rechunked"
    model_backends = {
        "all-MiniLM-L6-v2": baseline_backend,
        alternative_model: _load_generic_model(
            alternative_model,
            alternative_query_prefix,
            allow_download,
        ),
    }
    all_rows = []
    all_per_query_rows = []
    for model_name, backend in model_backends.items():
        for token_limit in CHUNK_SIZES:
            config_name = (
                f"3a_minilm_{token_limit}" if model_name == "all-MiniLM-L6-v2"
                and token_limit == 128 else
                f"3b_minilm_{token_limit}" if model_name == "all-MiniLM-L6-v2"
                else f"3c_alternative_{token_limit}" if token_limit == 128
                else f"3d_alternative_{token_limit}"
            )
            config = {
                "config_name": config_name,
                "method": "dense",
                "alpha": None,
                "k1": None,
                "b": None,
                "model_id": (
                    backend.identity.model_id if backend is not None else model_name
                ),
                "chunk_size_tokens": token_limit,
                "empty_processed_records_excluded": empty_processed_records_excluded,
                "status": "complete" if backend is not None else "model_not_available",
            }
            if backend is None:
                all_rows.extend(
                    _metric_row(
                        config=config,
                        query_type=split_name,
                        query_results={},
                        total_cases=len(group),
                        excluded_null_gold=sum(
                            case["gold_chunk_ids"] is None for case in group
                        ),
                        repeats=repeats,
                        bootstrap_samples=bootstrap_samples,
                        seed=seed,
                    )
                    for split_name, group in _query_groups(cases, annotations).items()
                )
                all_per_query_rows.extend(
                    _per_query_row(
                        case=case,
                        config=config,
                        query_type=str(annotations[case["id"]]["query_type"]),
                        metrics=None,
                        latency=None,
                        status="model_not_available",
                    )
                    for case in cases
                )
                continue
            if token_limit > backend.max_seq_length:
                config["status"] = "token_limit_exceeds_model_limit"
                all_rows.extend(
                    _metric_row(
                        config=config,
                        query_type=split_name,
                        query_results={},
                        total_cases=len(group),
                        excluded_null_gold=sum(
                            case["gold_chunk_ids"] is None for case in group
                        ),
                        repeats=repeats,
                        bootstrap_samples=bootstrap_samples,
                        seed=seed,
                    )
                    for split_name, group in _query_groups(cases, annotations).items()
                )
                all_per_query_rows.extend(
                    _per_query_row(
                        case=case,
                        config=config,
                        query_type=str(annotations[case["id"]]["query_type"]),
                        metrics=None,
                        latency=None,
                        status="token_limit_exceeds_model_limit",
                    )
                    for case in cases
                )
                continue

            model_slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", model_name)
            variant_dir = chunk_root / f"{model_slug}_{token_limit}"
            cache_path = cache_root / f"{model_slug}_{token_limit}.json"
            chunks, token_counts = build_token_chunks(
                processed_records, _tokenizer_for_backend(backend), token_limit
            )
            cached_chunks, reused_chunk_path = _load_reusable_chunks(
                reuse_roots, model_slug, token_limit, chunks
            )
            if cached_chunks is not None:
                if reused_chunk_path is None:
                    raise EvaluationInputError(
                        "reusable chunk data was loaded without its source path"
                    )
                chunks = cached_chunks
                shutil.copytree(reused_chunk_path, variant_dir)
                chunks_reused = True
            else:
                _write_chunks(chunks, variant_dir)
                chunks_reused = False
            _copy_reusable_embedding_cache(
                reuse_roots,
                lambda root: (
                    root
                    / "experiment_3"
                    / "embedding_caches"
                    / f"{model_slug}_{token_limit}.json"
                ),
                cache_path,
            )
            started = time.perf_counter()
            cache = load_or_build_embeddings(chunks, backend, cache_path)
            embedding_build_time = (
                0.0
                if cache.cache_hit
                else time.perf_counter() - started
            )
            index = CosineIndex(
                chunks,
                [
                    cache.vectors_by_chunk_id[str(chunk["chunk_id"])]
                    for chunk in chunks
                ],
            )
            token_ci_low, token_ci_high = bootstrap_mean_ci(
                [float(count) for count in token_counts],
                samples=bootstrap_samples,
                seed=seed + token_limit,
            )
            config.update(
                {
                    "n_chunks": len(chunks),
                    "avg_tokens_per_chunk": (
                        math.fsum(token_counts) / len(token_counts)
                        if token_counts else None
                    ),
                    "avg_tokens_ci_low": token_ci_low,
                    "avg_tokens_ci_high": token_ci_high,
                    "embedding_build_time_seconds": embedding_build_time,
                    "embedding_cache_hit": cache.cache_hit,
                    "chunks_reused": chunks_reused,
                    "reused_chunks_from": (
                        str(reused_chunk_path) if reused_chunk_path else None
                    ),
                }
            )
            config_rows, config_query_rows = _evaluate_fact_config(
                config=config,
                cases=cases,
                annotations=annotations,
                records=chunks,
                index=index,
                backend=backend,
                repeats=repeats,
                bootstrap_samples=bootstrap_samples,
                seed=seed,
            )
            all_rows.extend(config_rows)
            all_per_query_rows.extend(config_query_rows)
    return all_rows, all_per_query_rows


def _plot_experiment2(rows: Sequence[Mapping[str, object]], output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    overall = [row for row in rows if row["query_type"] == "overall"]
    by_alpha = sorted(overall, key=lambda row: float(row["alpha"]))
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(
        [float(row["alpha"]) for row in by_alpha],
        [float(row["hit_at_3"]) for row in by_alpha],
        marker="o",
        label="Hit@3",
    )
    axis.plot(
        [float(row["alpha"]) for row in by_alpha],
        [float(row["ndcg_at_5"]) for row in by_alpha],
        marker="s",
        label="nDCG@5",
    )
    axis.set(xlabel="Dense weight α", ylabel="Score", ylim=(0, 1))
    axis.set_title("Experiment 2: Hit@3 and nDCG@5 by α")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "experiment_2_alpha_sweep.png", dpi=160)
    plt.close(figure)

    names = [str(row["config_name"]) for row in overall]
    metric_names = ("hit_at_1", "hit_at_3", "hit_at_5", "mrr", "ndcg_at_5")
    width = 0.8 / len(metric_names)
    figure, axis = plt.subplots(figsize=(12, 6))
    positions = list(range(len(names)))
    for index, metric in enumerate(metric_names):
        offset = (index - (len(metric_names) - 1) / 2) * width
        axis.bar(
            [position + offset for position in positions],
            [float(row[metric]) for row in overall],
            width,
            label=metric.replace("_", " "),
        )
    axis.set_xticks(positions, names, rotation=25, ha="right")
    axis.set(ylabel="Score", ylim=(0, 1))
    axis.set_title("Experiment 2: retrieval configurations")
    axis.legend(ncols=3)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "experiment_2_configurations.png", dpi=160)
    plt.close(figure)


def _plot_experiment3(rows: Sequence[Mapping[str, object]], output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    overall = [
        row for row in rows
        if row["query_type"] == "overall" and row["status"] == "complete"
    ]
    names = [str(row["config_name"]) for row in overall]
    metrics = ("hit_at_3", "mrr")
    width = 0.36
    figure, axis = plt.subplots(figsize=(8, 5))
    positions = list(range(len(names)))
    for index, metric in enumerate(metrics):
        axis.bar(
            [position + (index - 0.5) * width for position in positions],
            [float(row[metric]) for row in overall],
            width,
            label=metric.replace("_", " "),
        )
    axis.set_xticks(positions, names, rotation=20, ha="right")
    axis.set(ylabel="Score", ylim=(0, 1))
    axis.set_title("Experiment 3: model and chunk-size settings")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "experiment_3_configurations.png", dpi=160)
    plt.close(figure)


def _save_config(
    path: Path,
    *,
    run_id: str,
    cases_path: Path,
    annotations_path: Path,
    args: argparse.Namespace,
) -> dict:
    config = {
        "run_id": run_id,
        "run_date_utc": datetime.now(timezone.utc).isoformat(),
        "commit_hash": git_commit(REPOSITORY_ROOT),
        "evaluation_dataset": str(cases_path),
        "evaluation_dataset_sha256": file_sha256(cases_path),
        "annotations_file": str(annotations_path),
        "annotations_sha256": file_sha256(annotations_path),
        "latency_repeats": args.repeats,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.seed,
        "top_k": TOP_K,
        "mrr_cutoff": TOP_K,
        "query_type_split": "first 20 exact_term; final 20 paraphrased",
        "null_gold_policy": "exclude from metric denominators and report count",
        "reused_artifact_roots": [str(path.resolve()) for path in args.reuse_artifacts_from],
    }
    write_json(path, config)
    return config


def _config_for_experiment2(config: Mapping[str, object]) -> dict:
    return {
        **config,
        "experiment": 2,
        "hybrid_alphas": list(HYBRID_ALPHAS),
        "bm25_k1": BM25_K1,
        "bm25_b": BM25_B,
        "score_fusion": "min-max normalize dense and BM25, then alpha weighted sum",
    }


def _config_for_experiment3(
    config: Mapping[str, object], args: argparse.Namespace
) -> dict:
    return {
        **config,
        "experiment": 3,
        "models": [
            MiniLMEmbedder.MODEL_ID,
            args.alternative_model,
        ],
        "alternative_query_prefix": args.alternative_query_prefix,
        "chunk_sizes_tokens": list(CHUNK_SIZES),
        "chunking": "processed-record text, tokenizer offsets, no overlap",
        "relevance": (
            "a result is relevant when it contains an annotated key-fact phrase"
        ),
        "model_downloads_allowed": args.allow_model_download,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Phase 2 retrieval experiments 2 and 3."
    )
    parser.add_argument("--experiment", choices=("2", "3", "both"), default="both")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS_PATH)
    parser.add_argument("--chunked-dir", type=Path, default=CHUNKED_DIR)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--reuse-artifacts-from",
        type=Path,
        action="append",
        default=[],
        help=(
            "previous run directory to reuse validated chunks and embedding caches; "
            "may be supplied more than once"
        ),
    )
    parser.add_argument(
        "--alternative-model",
        default="sentence-transformers/all-mpnet-base-v2",
    )
    parser.add_argument("--alternative-query-prefix", default="")
    parser.add_argument(
        "--allow-model-download",
        action="store_true",
        help="allow Sentence Transformers to download the alternative model",
    )
    parser.add_argument("--repeats", type=int, default=LATENCY_REPEATS)
    parser.add_argument(
        "--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES
    )
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--run-id", help="optional stable output-directory name")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.repeats <= 0 or args.bootstrap_samples <= 0:
        parser.error("--repeats and --bootstrap-samples must be positive")

    cases_path = args.cases.resolve()
    annotations_path = args.annotations.resolve()
    reuse_roots = [path.resolve() for path in args.reuse_artifacts_from]
    missing_reuse_roots = [path for path in reuse_roots if not path.is_dir()]
    if missing_reuse_roots:
        parser.error(
            "reuse artifact directory does not exist: "
            + ", ".join(str(path) for path in missing_reuse_roots)
        )
    chunk_records = load_chunks(args.chunked_dir)
    cases, annotations = load_evaluation_inputs(
        cases_path, annotations_path, chunk_records
    )
    scored_cases = [case for case in cases if case["gold_chunk_ids"] is not None]
    if not scored_cases:
        raise EvaluationInputError("no cases have gold chunk IDs")

    run_id, output_dir = create_run_directory(args.output_dir, args.run_id)
    common_config = _save_config(
        output_dir / "run_config.json",
        run_id=run_id,
        cases_path=cases_path,
        annotations_path=annotations_path,
        args=args,
    )

    backend = MiniLMEmbedder(local_files_only=True)
    if args.experiment in ("2", "both"):
        experiment_dir = output_dir / "experiment_2"
        experiment_dir.mkdir()
        rows, per_query_rows = _experiment2(
            cases=cases,
            annotations=annotations,
            records=chunk_records,
            backend=backend,
            reuse_roots=reuse_roots,
            output_dir=experiment_dir,
            repeats=args.repeats,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )
        csv_path = experiment_dir / "results.csv"
        write_csv(csv_path, rows)
        per_query_path = experiment_dir / "per_query_results.csv"
        write_csv(per_query_path, per_query_rows)
        write_json(
            experiment_dir / "config.json",
            _config_for_experiment2(common_config)
            | {
                "corpus_chunk_count": len(chunk_records),
                "results_csv": str(csv_path),
                "per_query_results_csv": str(per_query_path),
                "mrr_cutoff": TOP_K,
            },
            ensure_ascii=True,
        )
        _plot_experiment2(rows, experiment_dir)
        print(f"Experiment 2 CSV: {csv_path}")

    if args.experiment in ("3", "both"):
        experiment_dir = output_dir / "experiment_3"
        experiment_dir.mkdir()
        processed_records = load_processed_records(args.processed_dir)
        processed_record_count = len(processed_records)
        processed_records, empty_record_ids = exclude_empty_processed_records(
            processed_records
        )
        if empty_record_ids:
            print(
                f"Experiment 3 excludes {len(empty_record_ids)} processed records "
                f"with empty text: {', '.join(empty_record_ids)}",
                file=sys.stderr,
            )
        rows, per_query_rows = _experiment3(
            cases=cases,
            annotations=annotations,
            processed_records=processed_records,
            baseline_backend=backend,
            alternative_model=args.alternative_model,
            alternative_query_prefix=args.alternative_query_prefix,
            allow_download=args.allow_model_download,
            empty_processed_records_excluded=len(empty_record_ids),
            reuse_roots=reuse_roots,
            output_dir=experiment_dir,
            repeats=args.repeats,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )
        csv_path = experiment_dir / "results.csv"
        write_csv(csv_path, rows)
        per_query_path = experiment_dir / "per_query_results.csv"
        write_csv(per_query_path, per_query_rows)
        write_json(
            experiment_dir / "config.json",
            _config_for_experiment3(common_config, args)
            | {
                "processed_record_count": processed_record_count,
                "empty_processed_records_excluded": len(empty_record_ids),
                "excluded_empty_record_ids": empty_record_ids,
                "usable_processed_record_count": len(processed_records),
                "results_csv": str(csv_path),
                "per_query_results_csv": str(per_query_path),
                "mrr_cutoff": TOP_K,
            },
            ensure_ascii=True,
        )
        _plot_experiment3(rows, experiment_dir)
        print(f"Experiment 3 CSV: {csv_path}")

    print(f"Run artifacts: {output_dir}")
    print(
        f"Scored {len(scored_cases)}/{len(cases)} queries; "
        f"excluded {len(cases) - len(scored_cases)} with null gold IDs."
    )
    print(f"Commit: {common_config['commit_hash']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
