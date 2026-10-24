"""Run a manually inspectable retrieval baseline over candidate evaluation queries."""

import argparse
import html
import json
import sys
import tempfile
from pathlib import Path
from typing import Mapping, Sequence, TextIO

from .embedding import MiniLMEmbedder
from .retrieval import CHUNKED_DIR, load_chunks
from .retriever import Retriever


EVALUATION_CASES_PATH = Path(__file__).resolve().with_name("retrieval_eval.json")
EXCERPT_LENGTH = 600


def _format_result(rank: int, score: float, record: Mapping[str, object]) -> str:
    text = str(record["text"])
    excerpt = text[:EXCERPT_LENGTH]
    if len(text) > EXCERPT_LENGTH:
        excerpt += "..."

    details = {
        "provenance": record["provenance"],
        "metadata": record["metadata"],
    }
    return "\n".join(
        (
            f"### Rank {rank} — `{record['chunk_id']}`",
            f"- Cosine similarity: **{score:.6f}**",
            f"- Content type: `{record['content_type']}`",
            f"- Title: {record['title']}",
            "",
            f"**Text excerpt ({min(len(text), EXCERPT_LENGTH)} "
            f"of {len(text)} characters):**",
            "",
            "<pre>" + html.escape(excerpt) + "</pre>",
            "",
            "<details><summary>Full chunk text</summary>",
            "",
            "<pre>" + html.escape(text) + "</pre>",
            "</details>",
            "",
            "<details><summary>Original provenance and metadata</summary>",
            "",
            "<pre>" + html.escape(
                json.dumps(details, ensure_ascii=False, indent=2)
            ) + "</pre>",
            "</details>",
            "</details>",
        )
    )


def render_evaluation_report(
    cases: Sequence[Mapping[str, object]],
    chunks: Sequence[Mapping[str, object]],
    backend: MiniLMEmbedder,
    cache_path: Path,
    top_k: int = 5,
) -> str:
    """Retrieve and render candidate queries without assigning relevance labels."""
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise TypeError("top_k must be a positive integer")
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    candidates = []
    excluded_ids = []
    for record in chunks:
        if backend.token_count(record["text"]) > backend.max_seq_length:
            excluded_ids.append(record["chunk_id"])
        else:
            candidates.append(record)

    retriever = Retriever(candidates, backend, cache_path)
    sections = [
        "# MiniLM retrieval baseline",
        "",
        f"- Model: `{backend.identity.model_id}`",
        f"- Revision: `{backend.identity.revision}`",
        f"- Candidate chunks: {len(candidates)}",
        f"- Top-k: {top_k}",
        "- Scores are cosine similarity, not relevance judgments.",
        "- These candidate queries have no human-assigned gold chunk IDs yet.",
        "",
        "## Chunks excluded for exceeding the model token limit",
        "",
        f"Excluded {len(excluded_ids)} of {len(chunks)} chunks "
        f"(limit: {backend.max_seq_length} tokens).",
    ]
    sections.extend(f"- `{chunk_id}`" for chunk_id in excluded_ids)
    if not excluded_ids:
        sections.append("- None")

    for case in cases:
        case_id = case.get("id")
        query = case.get("query")
        if not isinstance(case_id, str) or not isinstance(query, str):
            raise ValueError("each evaluation case must have string id and query")

        sections.extend(
            (
                "",
                f"## Query: {case_id}",
                "",
                f"> {query}",
                "",
                "Similarity ordering is not a manual relevance label.",
                "",
            )
        )
        results = retriever.search(query, top_k=top_k)
        if not results:
            sections.append("No candidate results.")
        else:
            sections.extend(
                _format_result(rank, result.score, result.record)
                for rank, result in enumerate(results, start=1)
            )
            sections.append("")
    return "\n".join(sections) + "\n"


def _load_cases(path: Path) -> list[Mapping[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get("cases") if isinstance(payload, dict) else None
    if not isinstance(cases, list):
        raise ValueError(f"Evaluation file must contain a cases list: {path}")
    return cases


def main(argv: Sequence[str] | None = None, output: TextIO = sys.stdout) -> int:
    parser = argparse.ArgumentParser(
        description="Run a manually inspectable MiniLM retrieval baseline."
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=Path, help="write Markdown report to this path")
    args = parser.parse_args(argv)
    if args.top_k <= 0:
        parser.error("--top-k must be greater than zero")

    cases = _load_cases(EVALUATION_CASES_PATH)
    chunks = load_chunks(CHUNKED_DIR)
    backend = MiniLMEmbedder(local_files_only=True)

    with tempfile.TemporaryDirectory(prefix="ev-rag-minilm-eval-") as temporary:
        cache_path = Path(temporary) / "temporary_embeddings.json"
        report = render_evaluation_report(
            cases, chunks, backend, cache_path, top_k=args.top_k
        )

    if args.output is None:
        output.write(report)
    else:
        args.output.write_text(report, encoding="utf-8")
        output.write(f"Wrote evaluation report to {args.output}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
