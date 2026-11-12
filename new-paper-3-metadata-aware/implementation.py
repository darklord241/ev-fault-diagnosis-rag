"""New retrieval experiments not covered by the earlier paper batch.

Methods:
1. Maximal Marginal Relevance (MMR) diversity selection.
2. Neighbor/context promotion inspired by multi-granularity parent promotion.
3. Metadata-aware dense retrieval.
4. ColBERT-lite token-level MaxSim late interaction.
5. Structured multi-query decomposition and score fusion.

All methods use the same 20-query, 105-gold-label benchmark and 891 chunks.
They are isolated ablations; the production retriever is not changed here.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import time
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CHUNK_DIR = ROOT / "scraper" / "data" / "chunked"
CACHE_PATH = ROOT / "scraper" / "data" / "embeddings" / "chunk_vectors.json"
DEFAULT_BENCHMARK = ROOT / "dev-notes" / "model-test" / "retrieval_eval_top20.json"
TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def load_records() -> list[dict]:
    records = []
    for path in sorted(CHUNK_DIR.glob("*.jsonl")):
        records.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
    return records


def load_vectors() -> dict[str, list[float]]:
    data = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {item["chunk_id"]: item["vector"] for item in data["embeddings"]}


def dot(left, right) -> float:
    return math.fsum(a * b for a, b in zip(left, right))


def dense(query_vector, records, vectors):
    return sorted(
        [(r["chunk_id"], dot(query_vector, vectors[r["chunk_id"]])) for r in records],
        key=lambda item: (-item[1], item[0]),
    )


def normalized_metadata(record: dict) -> str:
    provenance = record.get("provenance", {})
    metadata = record.get("metadata", {})
    fields = [
        record.get("title", ""),
        record.get("source_document", ""),
        provenance.get("source_type", ""),
        provenance.get("document_type", ""),
        provenance.get("manufacturer", ""),
        provenance.get("vehicle_model", ""),
        metadata.get("category") or "",
        metadata.get("group") or "",
    ]
    return " ".join(str(value) for value in fields if value).strip()


def mmr(query_vector, candidates, vectors, top_k=20, lambda_value=0.95):
    selected = []
    remaining = list(candidates)
    while remaining and len(selected) < top_k:
        best = None
        best_score = float("-inf")
        for chunk_id, relevance in remaining:
            redundancy = max(
                (dot(vectors[chunk_id], vectors[selected_id]) for selected_id in selected),
                default=0.0,
            )
            value = lambda_value * relevance - (1.0 - lambda_value) * redundancy
            if value > best_score:
                best_score = value
                best = (chunk_id, value)
        selected_id = best[0]
        selected.append(selected_id)
        remaining = [(chunk_id, score) for chunk_id, score in remaining if chunk_id != selected_id]
    return [(chunk_id, score) for score, chunk_id in []] or [(chunk_id, 0.0) for chunk_id in selected]


def context_promotion(ranking, records_by_id, records, vectors):
    by_document = defaultdict(list)
    for record in records:
        by_document[record["source_document"]].append(record)
    index_by_id = {record["chunk_id"]: index for index, record in enumerate(records)}
    promoted = []
    for chunk_id, score in ranking:
        record = records_by_id[chunk_id]
        neighbors = []
        current_page = record.get("metadata", {}).get("page")
        for neighbor in by_document[record["source_document"]]:
            page = neighbor.get("metadata", {}).get("page")
            if current_page is not None and page is not None and abs(page - current_page) <= 1:
                neighbors.append(vectors[neighbor["chunk_id"]])
        neighborhood = max((dot(vectors[chunk_id], vector) for vector in neighbors), default=0.0)
        promoted.append((chunk_id, score + 0.08 * neighborhood))
    return sorted(promoted, key=lambda item: (-item[1], index_by_id[item[0]]))


def split_query(query: str) -> list[str]:
    parts = [part.strip(" ,") for part in re.split(r"\band\b|[?;]", query, flags=re.I) if part.strip(" ,")]
    if len(parts) == 1:
        parts.append(f"What specification or recommended action is given for {query}")
    return list(dict.fromkeys(parts))


def evaluate(ranking, gold, cutoffs=(1, 5, 10, 20)):
    ranking = ranking[:20]
    first = next((index + 1 for index, chunk_id in enumerate(ranking) if chunk_id in gold), None)
    result = {"first_gold_rank": first}
    for cutoff in cutoffs:
        found = set(ranking[:cutoff]) & gold
        result[f"hit@{cutoff}"] = int(bool(found))
        result[f"recall@{cutoff}"] = len(found) / len(gold)
        result[f"precision@{cutoff}"] = len(found) / cutoff
        dcg = sum(1 / math.log2(index + 2) for index, chunk_id in enumerate(ranking[:cutoff]) if chunk_id in gold)
        ideal = min(cutoff, len(gold))
        idcg = sum(1 / math.log2(index + 2) for index in range(ideal))
        result[f"ndcg@{cutoff}"] = dcg / idcg if idcg else 0.0
    result["mrr@20"] = 1 / first if first else 0.0
    return result


def aggregate(rows, timings):
    numeric = [
        key for key, value in rows[0].items()
        if key != "first_gold_rank" and isinstance(value, (int, float))
    ]
    summary = {key: statistics.mean(row[key] for row in rows) for key in numeric}
    ranks = [row["first_gold_rank"] for row in rows if row["first_gold_rank"] is not None]
    summary["mean_first_gold_rank"] = statistics.mean(ranks) if ranks else None
    summary["queries_with_gold_by_20"] = sum(row["hit@20"] for row in rows)
    summary["mean_latency_ms"] = statistics.mean(timings)
    summary["median_latency_ms"] = statistics.median(timings)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cases = json.loads(args.benchmark.read_text())["cases"]
    records = load_records()
    vectors = load_vectors()
    records_by_id = {record["chunk_id"]: record for record in records}

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", local_files_only=True)
    metadata_texts = [normalized_metadata(record) for record in records]
    metadata_vectors = model.encode(metadata_texts, normalize_embeddings=True, show_progress_bar=False)
    metadata_vectors = {
        record["chunk_id"]: vector.tolist()
        for record, vector in zip(records, metadata_vectors)
    }

    method_notes = {
        "mmr": "Very conservative MMR selects relevant but non-redundant passages from dense top-100; lambda=0.95.",
        "context_promotion": "Dense scores receive a small same-document adjacent-page support bonus.",
        "metadata_aware": "Content and metadata vectors are fused 0.92/0.08; query metadata is inferred from query text.",
        "late_interaction": "ColBERT-lite hybrid: dense score plus token-level mean query-token MaxSim over dense top-100 candidates.",
        "source_aware": "Dense similarity is reweighted by an explicit source reliability prior: official 1.08, technical 1.03, unknown 1.00, community 0.96.",
    }
    methods = {name: {"summary": None, "per_query": [], "implementation_note": note} for name, note in method_notes.items()}

    for case in cases:
        query = case["query"]
        gold = set(case["gold_chunk_ids"])
        q_vector = model.encode(query, normalize_embeddings=True).tolist()
        base = dense(q_vector, records, vectors)
        for method in methods:
            start = time.perf_counter()
            if method == "mmr":
                ranking = mmr(q_vector, base[:100], vectors)
            elif method == "context_promotion":
                ranking = context_promotion(base[:100], records_by_id, records, vectors)
            elif method == "metadata_aware":
                query_meta = " ".join(word for word in tokens(query) if word in normalized_metadata({"title": query, "source_document": "", "provenance": {}, "metadata": {}}).lower())
                # The query text is also used as a metadata anchor when it contains
                # model, document, or component words not present in the short title.
                query_meta_vector = model.encode(query_meta or query, normalize_embeddings=True).tolist()
                ranking = sorted(
                    [
                        (
                            record["chunk_id"],
                            0.92 * dot(q_vector, vectors[record["chunk_id"]])
                            + 0.08 * dot(query_meta_vector, metadata_vectors[record["chunk_id"]]),
                        )
                        for record in records
                    ],
                    key=lambda item: (-item[1], item[0]),
                )
            elif method == "late_interaction":
                query_tokens = list(dict.fromkeys(tokens(query)))[:32]
                query_token_vectors = model.encode(query_tokens, normalize_embeddings=True, show_progress_bar=False)
                candidate_ids = [chunk_id for chunk_id, _ in base[:100]]
                ranking = []
                for chunk_id in candidate_ids:
                    doc_tokens = list(dict.fromkeys(tokens(records_by_id[chunk_id]["text"])))[:96]
                    if not doc_tokens:
                        ranking.append((chunk_id, 0.0))
                        continue
                    doc_vectors = model.encode(doc_tokens, normalize_embeddings=True, show_progress_bar=False)
                    max_scores = []
                    for q_token_vector in query_token_vectors:
                        max_scores.append(max(float(dot(q_token_vector, doc_vector)) for doc_vector in doc_vectors))
                    late_score = statistics.mean(max_scores)
                    dense_score = next(score for candidate_id, score in base if candidate_id == chunk_id)
                    ranking.append((chunk_id, 0.82 * dense_score + 0.18 * late_score))
                ranking.sort(key=lambda item: (-item[1], item[0]))
            else:
                priors = {
                    "official": 1.08,
                    "technical": 1.03,
                    "unknown": 1.00,
                    "community": 0.96,
                }
                ranking = sorted(
                    [
                        (
                            record["chunk_id"],
                            dot(q_vector, vectors[record["chunk_id"]])
                            * priors.get(
                                str(record.get("provenance", {}).get("source_type", "unknown")).lower(),
                                1.00,
                            ),
                        )
                        for record in records
                    ],
                    key=lambda item: (-item[1], item[0]),
                )
            elapsed = (time.perf_counter() - start) * 1000
            row = {"id": case["id"], "query": query, **evaluate([chunk_id for chunk_id, _ in ranking], gold)}
            row["latency_ms"] = elapsed
            methods[method]["per_query"].append(row)

    for method, data in methods.items():
        timings = [row.pop("latency_ms") for row in data["per_query"]]
        data["summary"] = aggregate(data["per_query"], timings)
    output = {"benchmark": str(args.benchmark), "corpus_chunks": len(records), "methods": methods}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    print(json.dumps({method: data["summary"] for method, data in methods.items()}, indent=2))


if __name__ == "__main__":
    main()
