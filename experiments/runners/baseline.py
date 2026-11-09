"""Run the production dense-retrieval configuration on the fixed benchmark."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Mapping, Sequence

from experiments.evaluation import (
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED,
    METRICS,
    TOP_K,
    load_evaluation_inputs,
)
from experiments.run_utils import (
    DEFAULT_RESULTS_DIR,
    create_run_directory,
    write_json,
    write_metric_plot,
    write_result_tables,
    write_run_metadata,
)
from experiments.runners.embedding_comparison import (
    ModelSpec,
    _aggregate_results,
    evaluate_embedding_model,
    load_model_specs,
)
from scraper.vector_search.index_embeddings import DEFAULT_CACHE_PATH
from scraper.vector_search.retrieval import CHUNKED_DIR, load_chunks


SCRIPT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "configs" / "embedding_models.json"
DEFAULT_CASES_PATH = SCRIPT_DIR / "datasets" / "retrieval_eval_40.json"
DEFAULT_ANNOTATIONS_PATH = SCRIPT_DIR / "datasets" / "evaluation_annotations.json"
DEFAULT_OUTPUT_DIR = DEFAULT_RESULTS_DIR / "baseline"
BASELINE_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"


def _baseline_model(path: Path = DEFAULT_CONFIG_PATH) -> ModelSpec:
    specs = load_model_specs(path)
    model = specs.get("current")
    if model is None or model.model_id != BASELINE_MODEL_ID:
        raise ValueError(
            "baseline requires the configured current model "
            f"{BASELINE_MODEL_ID!r}"
        )
    return model


def _write_report(
    path: Path,
    *,
    run_id: str,
    model: ModelSpec,
    aggregate_rows: Sequence[Mapping[str, object]],
) -> None:
    overall = next(
        row for row in aggregate_rows if row["query_type"] == "overall"
    )
    lines = [
        "# Baseline retrieval results",
        "",
        f"- Run: `{run_id}`",
        f"- Embedding model: `{model.model_id}`",
        "- Retrieval: dense cosine similarity, top-5",
        "- Corpus: the current 891 chunks",
        "- Benchmark: 40 evaluation cases; 39 have gold labels and are scored.",
        "- Metrics: Hit@1, Hit@3, Hit@5, MRR@5, and nDCG@5.",
        "- This run establishes the reference point for subsequent experiments.",
        "",
        "| Metric | Score | 95% bootstrap CI |",
        "|---|---:|---:|",
    ]
    for metric, label in (
        ("hit_at_1", "Hit@1"),
        ("hit_at_3", "Hit@3"),
        ("hit_at_5", "Hit@5"),
        ("mrr", "MRR@5"),
        ("ndcg_at_5", "nDCG@5"),
    ):
        value = overall[metric]
        low = overall[f"{metric}_ci_low"]
        high = overall[f"{metric}_ci_high"]
        if value is None or low is None or high is None:
            score = interval = "n/a"
        else:
            score = f"{float(value):.3f}"
            interval = f"[{float(low):.3f}, {float(high):.3f}]"
        lines.append(f"| {label} | {score} | {interval} |")
    lines.extend(
        [
            "",
            "Per-query retrievals and scores are in `per_query_results.csv`; "
            "aggregate metrics are in `aggregate_results.csv`.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the production retrieval baseline on the fixed benchmark."
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS_PATH)
    parser.add_argument("--chunks", type=Path, default=CHUNKED_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="fail if the baseline model is not already cached locally",
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
    if args.repeats <= 0 or args.bootstrap_samples <= 0:
        parser.error("--repeats and --bootstrap-samples must be positive")

    model = _baseline_model()
    cases_path = args.cases.resolve()
    annotations_path = args.annotations.resolve()
    config_path = DEFAULT_CONFIG_PATH.resolve()
    chunks = load_chunks(args.chunks)
    cases, annotations = load_evaluation_inputs(
        cases_path, annotations_path, chunks
    )
    if len(chunks) != 891:
        raise ValueError(
            f"baseline expects the existing 891 chunks; found {len(chunks)}"
        )
    scored_count = sum(case["gold_chunk_ids"] is not None for case in cases)
    if len(cases) != 40 or scored_count != 39:
        raise ValueError(
            "baseline expects 40 cases with 39 scored cases; found "
            f"{len(cases)} cases and {scored_count} scored"
        )

    run_id, output_dir = create_run_directory(args.output_dir, args.run_id)
    per_query_rows, model_metadata = evaluate_embedding_model(
        model,
        chunks,
        cases,
        annotations,
        cache_path=DEFAULT_CACHE_PATH,
        local_files_only=args.local_files_only,
        repeats=args.repeats,
    )
    exact_count = sum(
        annotations[case["id"]]["query_type"] == "exact_term" for case in cases
    )
    paraphrased_count = sum(
        annotations[case["id"]]["query_type"] == "paraphrased"
        for case in cases
    )
    aggregate_rows = _aggregate_results(
        per_query_rows,
        [model],
        {model.key: model_metadata},
        total_cases=len(cases),
        exact_case_count=exact_count,
        paraphrased_case_count=paraphrased_count,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )

    config = {
        "experiment": "baseline",
        "varied_parameter": None,
        "embedding_model": asdict(model),
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
    }
    write_json(output_dir / "config.json", config)
    write_result_tables(output_dir, aggregate_rows, per_query_rows)
    write_metric_plot(
        output_dir / "aggregate_metrics.png",
        aggregate_rows,
        [model.key],
        title="Baseline retrieval performance (top 5)",
    )
    _write_report(
        output_dir / "report.md",
        run_id=run_id,
        model=model,
        aggregate_rows=aggregate_rows,
    )
    write_run_metadata(
        output_dir,
        run_id=run_id,
        config_path=output_dir / "config.json",
        input_files={
            "evaluation_dataset": cases_path,
            "annotations": annotations_path,
            "model_config": config_path,
        },
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
        experiment_details={"model_runtime": {model.key: model_metadata}},
    )
    print(f"Run artifacts: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
