"""Compare encoder embedding models on the fixed retrieval benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

from experiments.evaluation import (
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED,
    METRICS,
    TOP_K,
    bootstrap_mean_ci,
    load_evaluation_inputs,
    ranking_metrics,
)
from experiments.run_utils import (
    DEFAULT_RESULTS_DIR,
    create_run_directory,
    file_sha256,
    write_json,
    write_metric_plot,
    write_result_tables,
    write_run_metadata,
)
from scraper.vector_search.embedding import (
    EmbeddingBackend,
    MiniLMEmbedder,
    SentenceTransformerTaskEmbedder,
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
    load_chunks,
)


SCRIPT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "configs" / "embedding_models.json"
DEFAULT_CASES_PATH = SCRIPT_DIR / "datasets" / "retrieval_eval_40.json"
DEFAULT_ANNOTATIONS_PATH = SCRIPT_DIR / "datasets" / "evaluation_annotations.json"
DEFAULT_OUTPUT_DIR = DEFAULT_RESULTS_DIR / "embedding_comparison"
EMBEDDING_CACHE_DIR = DEFAULT_CACHE_PATH.parent / "experiments"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    model_id: str
    query_prefix: str = ""
    document_prefix: str = ""
    revision: str = "main"


def load_model_specs(path: Path) -> dict[str, ModelSpec]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not load embedding model config {path}: {error}") from error
    entries = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(entries, list) or not entries:
        raise ValueError("embedding model config must contain a nonempty models list")

    specs = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("each embedding model config entry must be an object")
        key = entry.get("key")
        model_id = entry.get("model_id")
        revision = entry.get("revision", "main")
        query_prefix = entry.get("query_prefix", "")
        document_prefix = entry.get("document_prefix", "")
        if not isinstance(key, str) or not key.strip():
            raise ValueError("each embedding model must have a nonempty key")
        if key in specs:
            raise ValueError(f"duplicate embedding model key: {key}")
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError(f"{key}: model_id must be a nonempty string")
        if not isinstance(revision, str) or not revision.strip():
            raise ValueError(f"{key}: revision must be a nonempty string")
        if not isinstance(query_prefix, str) or not isinstance(document_prefix, str):
            raise ValueError(f"{key}: embedding prefixes must be strings")
        specs[key] = ModelSpec(
            key=key,
            model_id=model_id,
            revision=revision,
            query_prefix=query_prefix,
            document_prefix=document_prefix,
        )
    return specs


def create_backend(
    spec: ModelSpec,
    *,
    local_files_only: bool,
) -> EmbeddingBackend:
    if spec.key == "current":
        return MiniLMEmbedder(
            revision=spec.revision,
            local_files_only=local_files_only,
        )
    return SentenceTransformerTaskEmbedder(
        spec.model_id,
        revision=spec.revision,
        query_prefix=spec.query_prefix,
        document_prefix=spec.document_prefix,
        local_files_only=local_files_only,
    )


def _seed_cache(
    candidates: Sequence[Path],
    destination: Path,
) -> Path | None:
    if destination.exists():
        return None
    for source in candidates:
        if source.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            return source
    return None


def evaluate_embedding_model(
    model: ModelSpec,
    chunks: Sequence[Mapping[str, object]],
    cases: Sequence[Mapping[str, object]],
    annotations: Mapping[str, Mapping[str, object]],
    *,
    cache_path: Path,
    reuse_caches: Sequence[Path] = (),
    local_files_only: bool = False,
    repeats: int = 5,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    print(f"Loading embedding model {model.key}: {model.model_id}")
    backend = create_backend(model, local_files_only=local_files_only)
    reused_cache = _seed_cache(reuse_caches, cache_path)
    cache = load_or_build_embeddings(chunks, backend, cache_path)
    index = CosineIndex(
        chunks,
        [cache.vectors_by_chunk_id[str(chunk["chunk_id"])] for chunk in chunks],
    )
    model_metadata: dict[str, object] = {
        "model_id": backend.identity.model_id,
        "revision": backend.identity.revision,
        "preprocessing": backend.identity.preprocessing,
        "dimension": backend.identity.dimension,
        "max_seq_length": getattr(backend, "max_seq_length", None),
        "cache_hit": cache.cache_hit,
        "cache_rebuild_reason": cache.rebuild_reason,
        "cache_path": str(cache_path),
        "copied_candidate_cache": str(reused_cache) if reused_cache else None,
    }

    query_types = {
        case["id"]: annotations[case["id"]]["query_type"] for case in cases
    }
    per_query_rows = []
    for case in cases:
        latency_samples = []
        results: Sequence[SearchResult] = []
        for _ in range(repeats):
            started = time.perf_counter()
            query_vector = embed_query(backend, str(case["query"]))
            results = index.search(query_vector, top_k=TOP_K)
            latency_samples.append(time.perf_counter() - started)
        metrics = (
            ranking_metrics(
                [str(result.record["chunk_id"]) for result in results],
                gold_ids=set(case["gold_chunk_ids"]),
            )
            if case["gold_chunk_ids"] is not None
            else None
        )
        per_query_rows.append(
            _per_query_result(
                case,
                str(query_types[case["id"]]),
                model,
                backend,
                results,
                metrics,
                math.fsum(latency_samples) / len(latency_samples),
            )
        )
    print(
        f"Completed {model.key}; cache_hit={cache.cache_hit}; "
        f"dimension={backend.identity.dimension}"
    )
    return per_query_rows, model_metadata


def _per_query_result(
    case: Mapping[str, object],
    query_type: str,
    model: ModelSpec,
    backend: EmbeddingBackend,
    results: Sequence[SearchResult],
    metrics: Mapping[str, float | int | None] | None,
    latency_seconds: float,
) -> dict[str, object]:
    gold_ids = case["gold_chunk_ids"]
    return {
        "model_key": model.key,
        "model_id": backend.identity.model_id,
        "query_id": case["id"],
        "query_text": case["query"],
        "query_type": query_type,
        "status": "scored" if gold_ids is not None else "unscored_null_gold",
        "retrieved_chunk_ids": json.dumps(
            [result.record["chunk_id"] for result in results],
            ensure_ascii=False,
        ),
        "retrieval_scores": json.dumps(
            [result.score for result in results],
            ensure_ascii=False,
        ),
        "gold_chunk_ids": (
            json.dumps(gold_ids, ensure_ascii=False) if gold_ids is not None else ""
        ),
        "first_relevant_rank": (
            metrics["first_relevant_rank"] if metrics is not None else None
        ),
        "hit_at_1": metrics["hit_at_1"] if metrics is not None else None,
        "hit_at_3": metrics["hit_at_3"] if metrics is not None else None,
        "hit_at_5": metrics["hit_at_5"] if metrics is not None else None,
        "mrr_at_5": metrics["mrr"] if metrics is not None else None,
        "ndcg_at_5": metrics["ndcg_at_5"] if metrics is not None else None,
        "latency_seconds": latency_seconds,
    }


def _aggregate_results(
    per_query_results: Sequence[Mapping[str, object]],
    models: Sequence[ModelSpec],
    model_metadata: Mapping[str, Mapping[str, object]],
    *,
    total_cases: int,
    exact_case_count: int,
    paraphrased_case_count: int,
    bootstrap_samples: int,
    seed: int,
) -> list[dict[str, object]]:
    rows = []
    groups = {
        "overall": None,
        "exact_term": "exact_term",
        "paraphrased": "paraphrased",
    }
    for model in models:
        model_queries = [
            row for row in per_query_results
            if row["model_key"] == model.key and row["status"] == "scored"
        ]
        for group_index, (group_name, query_type) in enumerate(groups.items()):
            group_queries = [
                row for row in model_queries
                if query_type is None or row["query_type"] == query_type
            ]
            total = (
                total_cases if query_type is None
                else exact_case_count if query_type == "exact_term"
                else paraphrased_case_count
            )
            row: dict[str, object] = {
                "model_key": model.key,
                "model_id": model_metadata[model.key]["model_id"],
                "revision": model_metadata[model.key]["revision"],
                "query_type": group_name,
                "n_total_cases": total,
                "n_scored": len(group_queries),
                "n_excluded_null_gold": total - len(group_queries),
            }
            for metric in METRICS:
                per_query_metric = "mrr_at_5" if metric == "mrr" else metric
                values = [
                    float(item[per_query_metric]) for item in group_queries
                ]
                low, high = bootstrap_mean_ci(
                    values,
                    samples=bootstrap_samples,
                    seed=seed + group_index,
                )
                row[metric] = math.fsum(values) / len(values) if values else None
                row[f"{metric}_ci_low"] = low if values else None
                row[f"{metric}_ci_high"] = high if values else None

            latencies = [
                float(item["latency_seconds"]) for item in group_queries
            ]
            low, high = bootstrap_mean_ci(
                latencies,
                samples=bootstrap_samples,
                seed=seed + group_index + 10,
            )
            row["mean_query_latency_seconds"] = (
                math.fsum(latencies) / len(latencies) if latencies else None
            )
            row["mean_query_latency_ci_low"] = low if latencies else None
            row["mean_query_latency_ci_high"] = high if latencies else None
            rows.append(row)
    return rows


def _write_report(
    path: Path,
    *,
    models: Sequence[ModelSpec],
    model_metadata: Mapping[str, Mapping[str, object]],
    aggregate_rows: Sequence[Mapping[str, object]],
    per_query_rows: Sequence[Mapping[str, object]],
    historical_mpnet_rows: Sequence[Mapping[str, object]],
    run_id: str,
) -> None:
    def value_with_ci(row: Mapping[str, object], metric: str) -> str:
        value = row.get(metric)
        low = row.get(f"{metric}_ci_low")
        high = row.get(f"{metric}_ci_high")
        if value is None or low is None or high is None:
            return "n/a"
        return f"{float(value):.3f} [{float(low):.3f}, {float(high):.3f}]"

    overall = {
        str(row["model_key"]): row
        for row in aggregate_rows
        if row["query_type"] == "overall"
    }
    lines = [
        "# Encoder embedding model comparison",
        "",
        f"- Run: `{run_id}`",
        f"- Objective: compare embedding models while holding the existing 891 chunks,",
        "  chunk IDs, cosine retrieval, top-k, queries, and gold labels fixed.",
        "- Chunking is unchanged: FAQs remain single chunks; PDFs use 600-word chunks",
        "  with 100-word overlap.",
        "- Benchmark: 40 evaluation cases; 39 have gold labels and are scored.",
        "- Metrics: Hit@1/3/5, MRR@5, nDCG@5, and query latency.",
        "- Latency measures query embedding plus top-5 cosine retrieval; model loading,",
        "  document embedding, and cache/index construction are excluded.",
        "- Each aggregate includes a 95% percentile bootstrap interval over scored queries.",
        "- This small benchmark is descriptive; intervals do not establish superiority.",
        "",
        "## Overall results",
        "",
        "| Model | Hit@1 | Hit@3 (95% CI) | Hit@5 | MRR@5 (95% CI) | nDCG@5 | Mean latency ms (95% CI) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in models:
        row = overall[model.key]
        lines.append(
            "| "
            + " | ".join(
                (
                    model.key,
                    f"{float(row['hit_at_1']):.3f}",
                    value_with_ci(row, "hit_at_3"),
                    f"{float(row['hit_at_5']):.3f}",
                    value_with_ci(row, "mrr"),
                    f"{float(row['ndcg_at_5']):.3f}",
                    (
                        f"{1000 * float(row['mean_query_latency_seconds']):.2f} "
                        f"[{1000 * float(row['mean_query_latency_ci_low']):.2f}, "
                        f"{1000 * float(row['mean_query_latency_ci_high']):.2f}]"
                    ),
                )
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Model configuration",
            "",
            "| Model | Model identifier | Revision | Query prefix | Document prefix | Dimension |",
            "|---|---|---|---|---|---:|",
        ]
    )
    for model in models:
        metadata = model_metadata[model.key]
        lines.append(
            "| "
            + " | ".join(
                (
                    model.key,
                    str(metadata["model_id"]),
                    str(metadata["revision"]),
                    f"`{model.query_prefix}`",
                    f"`{model.document_prefix}`",
                    str(metadata["dimension"]),
                )
            )
            + " |"
        )

    baseline_by_id = {
        str(row["query_id"]): row
        for row in per_query_rows
        if row["model_key"] == "current" and row["status"] == "scored"
    }
    lines.extend(["", "## Hit@3 changes versus current model", ""])
    for model in models:
        if model.key == "current":
            continue
        compared = [
            row for row in per_query_rows
            if row["model_key"] == model.key and row["status"] == "scored"
        ]
        current_only = []
        model_only = []
        for row in compared:
            baseline = baseline_by_id[str(row["query_id"])]
            current_hit = int(float(baseline["hit_at_3"]))
            model_hit = int(float(row["hit_at_3"]))
            if current_hit and not model_hit:
                current_only.append(str(row["query_id"]))
            elif model_hit and not current_hit:
                model_only.append(str(row["query_id"]))
        lines.append(
            f"- **{model.key}:** current-only Hit@3 cases: "
            f"{', '.join(current_only) if current_only else 'none'}; "
            f"{model.key}-only Hit@3 cases: "
            f"{', '.join(model_only) if model_only else 'none'}."
        )

    if historical_mpnet_rows:
        lines.extend(
            [
                "",
                "## Existing MPNet results (historical, not directly comparable)",
                "",
                "These are retained as context only. They were produced by the prior "
                "Experiment 3 setup, which re-chunked processed records to 128/256 "
                "tokens and judged relevance by annotated key-fact phrases. This "
                "comparison uses the fixed 891 chunks and gold chunk IDs, so the "
                "numbers below must not be ranked against the table above.",
                "",
                "| Model/configuration | Hit@3 | MRR | nDCG@5 | Mean latency (ms) |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in historical_mpnet_rows:
            lines.append(
                "| "
                + " | ".join(
                    (
                        f"{row['model_id']} / {row['chunk_size_tokens']} tokens",
                        f"{float(row['hit_at_3']):.3f}",
                        f"{float(row['mrr']):.3f}",
                        f"{float(row['ndcg_at_5']):.3f}",
                        f"{1000 * float(row['mean_query_latency_seconds']):.2f}",
                    )
                )
                + " |"
            )

    lines.extend(
        [
            "",
            "## Preliminary conclusion",
            "",
            "Use the aggregate table and per-query CSV to identify observed differences. "
            "No model is declared statistically superior from these results alone; "
            "the paired per-query outcomes and uncertainty intervals should be reviewed "
            "before drawing broader conclusions.",
            "",
            "Detailed query-level results are in `per_query_results.csv`; aggregate "
            "results and confidence intervals are in `aggregate_results.csv`.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare encoder embeddings on the fixed 39-query benchmark."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="model keys to select from --config; defaults to all configured models",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument(
        "--annotations", type=Path, default=DEFAULT_ANNOTATIONS_PATH
    )
    parser.add_argument("--chunks", type=Path, default=CHUNKED_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--historical-mpnet-results",
        type=Path,
        help=(
            "optional prior Experiment 3 aggregate CSV; reported separately "
            "because its chunking/relevance protocol differs"
        ),
    )
    parser.add_argument(
        "--reuse-cache",
        type=Path,
        action="append",
        default=[],
        help="existing embedding cache to validate and reuse, repeatable",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="fail if any selected model is not already cached locally",
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES
    )
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--run-id", help="optional output folder name")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config_path = args.config.resolve()
    try:
        configured_models = load_model_specs(config_path)
    except ValueError as error:
        parser.error(str(error))
    selected_keys = args.models or list(configured_models)
    unknown_models = sorted(set(selected_keys) - configured_models.keys())
    if unknown_models:
        parser.error(
            "unknown model key(s) in --models: " + ", ".join(unknown_models)
        )
    if "current" not in selected_keys:
        parser.error("--models must include current as the baseline")
    if len(selected_keys) != len(set(selected_keys)):
        parser.error("--models must not contain duplicates")
    if args.repeats <= 0 or args.bootstrap_samples <= 0:
        parser.error("--repeats and --bootstrap-samples must be positive")

    reuse_caches = [path.resolve() for path in args.reuse_cache]
    missing_caches = [path for path in reuse_caches if not path.is_file()]
    if missing_caches:
        parser.error(
            "reuse cache file does not exist: "
            + ", ".join(str(path) for path in missing_caches)
        )

    cases_path = args.cases.resolve()
    annotations_path = args.annotations.resolve()
    historical_mpnet_path = (
        args.historical_mpnet_results.resolve()
        if args.historical_mpnet_results is not None
        else None
    )
    historical_mpnet_rows: list[dict[str, str]] = []
    if historical_mpnet_path is not None:
        if not historical_mpnet_path.is_file():
            parser.error(
                f"historical MPNet results CSV does not exist: {historical_mpnet_path}"
            )
        with historical_mpnet_path.open(
            encoding="utf-8", newline=""
        ) as file:
            historical_mpnet_rows = [
                row
                for row in csv.DictReader(file)
                if row.get("query_type") == "overall"
                and row.get("model_id") == "sentence-transformers/all-mpnet-base-v2"
                and row.get("status") == "complete"
            ]
    chunks = load_chunks(args.chunks)
    cases, annotations = load_evaluation_inputs(
        cases_path, annotations_path, chunks
    )
    if len(chunks) != 891:
        raise ValueError(
            f"controlled comparison expects the existing 891 chunks; found {len(chunks)}"
        )
    scored_count = sum(case["gold_chunk_ids"] is not None for case in cases)
    if scored_count != 39:
        raise ValueError(
            f"controlled comparison expects 39 scored cases; found {scored_count}"
        )

    models = [configured_models[key] for key in selected_keys]
    run_id, output_dir = create_run_directory(args.output_dir, args.run_id)
    EMBEDDING_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    metadata: dict[str, dict[str, object]] = {}
    per_query_rows: list[dict[str, object]] = []
    for model in models:
        cache_path = EMBEDDING_CACHE_DIR / f"{model.key}.json"
        model_rows, model_metadata = evaluate_embedding_model(
            model,
            chunks,
            cases,
            annotations,
            cache_path=cache_path,
            reuse_caches=reuse_caches,
            local_files_only=args.local_files_only,
            repeats=args.repeats,
        )
        metadata[model.key] = model_metadata
        per_query_rows.extend(model_rows)

    exact_count = sum(
        annotations[case["id"]]["query_type"] == "exact_term" for case in cases
    )
    paraphrased_count = sum(
        annotations[case["id"]]["query_type"] == "paraphrased" for case in cases
    )
    aggregate_rows = _aggregate_results(
        per_query_rows,
        models,
        metadata,
        total_cases=len(cases),
        exact_case_count=exact_count,
        paraphrased_case_count=paraphrased_count,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    config = {
        "experiment": "embedding_comparison",
        "varied_parameter": "embedding_model",
        "objective": "compare encoder embedding models only",
        "models": [asdict(model) for model in models],
        "model_order": [model.key for model in models],
        "retrieval": {"method": "dense_cosine", "top_k": TOP_K},
        "benchmark": {
            "dataset": "retrieval_eval_40.json",
            "case_count": len(cases),
            "scored_query_count": scored_count,
            "unscored_null_gold_count": len(cases) - scored_count,
        },
        "metrics": list(METRICS),
        "latency_repeats": args.repeats,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.seed,
        "controlled_variables": [
            "same corpus and chunk IDs",
            "same query texts and gold labels",
            "same exact cosine index and top-k",
            "same CPU and batch settings",
        ],
        "chunking": {
            "faq": "one self-contained chunk per record",
            "pdf_chunk_words": 600,
            "pdf_overlap_words": 100,
        },
    }
    write_json(output_dir / "config.json", config)
    write_result_tables(output_dir, aggregate_rows, per_query_rows)
    write_metric_plot(
        output_dir / "aggregate_metrics.png",
        aggregate_rows,
        [model.key for model in models],
        title="Embedding-model retrieval comparison (top 5)",
    )
    _write_report(
        output_dir / "report.md",
        models=models,
        model_metadata=metadata,
        aggregate_rows=aggregate_rows,
        per_query_rows=per_query_rows,
        historical_mpnet_rows=historical_mpnet_rows,
        run_id=run_id,
    )

    input_files = {
        "evaluation_dataset": cases_path,
        "annotations": annotations_path,
        "model_config": config_path,
    }
    if historical_mpnet_path is not None:
        input_files["historical_mpnet_results"] = historical_mpnet_path
    experiment_details = {
        "model_runtime": metadata,
        "model_order": [model.key for model in models],
        "reused_cache_candidates": [str(path) for path in reuse_caches],
        "historical_mpnet_results": (
            {
                "path": str(historical_mpnet_path),
                "sha256": file_sha256(historical_mpnet_path),
                "rows_in_report": len(historical_mpnet_rows),
                "comparability": (
                    "not directly comparable: prior re-chunking and "
                    "key-fact-phrase relevance"
                ),
            }
            if historical_mpnet_path is not None
            else None
        ),
    }
    write_run_metadata(
        output_dir,
        run_id=run_id,
        config_path=output_dir / "config.json",
        input_files=input_files,
        corpus={
            "chunk_directory": str(args.chunks.resolve()),
            "chunk_count": len(chunks),
        },
        benchmark={
            "case_count": len(cases),
            "scored_query_count": scored_count,
            "unscored_null_gold_count": len(cases) - scored_count,
            "top_k": TOP_K,
            "metrics": list(METRICS),
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed": args.seed,
            "latency_repeats": args.repeats,
        },
        packages=("sentence-transformers", "torch"),
        experiment_details=experiment_details,
    )
    print(f"Aggregate results: {output_dir / 'aggregate_results.csv'}")
    print(f"Per-query results: {output_dir / 'per_query_results.csv'}")
    print(f"Report: {output_dir / 'report.md'}")
    print(f"Run artifacts: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
