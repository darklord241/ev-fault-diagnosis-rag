"""Shared input validation and evaluation statistics for experiment runners."""

from __future__ import annotations

import json
import math
import random
import re
from pathlib import Path
from typing import Mapping, Sequence


QUERY_TYPES = ("exact_term", "paraphrased")
TOP_K = 5
BOOTSTRAP_SAMPLES = 5000
BOOTSTRAP_SEED = 20261028
TOKEN_PATTERN = re.compile(r"(?u)\b[\w]+(?:[.-][\w]+)*\b")
METRICS = ("hit_at_1", "hit_at_3", "hit_at_5", "mrr", "ndcg_at_5")


class EvaluationInputError(ValueError):
    """Raised when evaluation annotations or corpus inputs are inconsistent."""


def normalize_text(text: str) -> str:
    """Normalize text to word tokens for punctuation-insensitive fact matching."""
    return " ".join(re.findall(r"(?u)\w+", text.casefold()))


def tokenize_bm25(text: str) -> list[str]:
    """Tokenize without splitting identifiers such as 700W or 450X."""
    return [token.casefold() for token in TOKEN_PATTERN.findall(text)]


def min_max_normalize(scores: Sequence[float]) -> list[float]:
    if not scores:
        return []
    minimum = min(scores)
    maximum = max(scores)
    if math.isclose(maximum, minimum):
        return [0.0] * len(scores)
    return [(score - minimum) / (maximum - minimum) for score in scores]


def fuse_scores(
    dense_scores: Sequence[float],
    bm25_scores: Sequence[float],
    alpha: float,
) -> list[float]:
    if len(dense_scores) != len(bm25_scores):
        raise ValueError("dense and BM25 score lists must have equal lengths")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    dense_normalized = min_max_normalize(dense_scores)
    bm25_normalized = min_max_normalize(bm25_scores)
    return [
        alpha * dense + (1.0 - alpha) * sparse
        for dense, sparse in zip(dense_normalized, bm25_normalized)
    ]


def _percentile(sorted_values: Sequence[float], percentile: float) -> float:
    if not sorted_values:
        return math.nan
    position = (len(sorted_values) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """Return a deterministic percentile bootstrap interval for a sample mean."""
    if not values:
        return math.nan, math.nan
    if samples <= 0:
        raise ValueError("samples must be positive")
    if len(values) == 1:
        return values[0], values[0]

    generator = random.Random(seed)
    means = sorted(
        math.fsum(generator.choice(values) for _ in values) / len(values)
        for _ in range(samples)
    )
    return _percentile(means, 0.025), _percentile(means, 0.975)


def ranking_metrics(
    ranked_ids: Sequence[str],
    *,
    gold_ids: set[str] | None = None,
    fact_phrases: Sequence[str] | None = None,
    text_by_id: Mapping[str, str] | None = None,
    total_relevant: int | None = None,
) -> dict[str, float | int | None]:
    """Calculate binary relevance metrics for a single ranked query."""
    if gold_ids is None and (fact_phrases is None or text_by_id is None):
        raise ValueError("gold IDs or key facts and chunk text are required")

    relevant = []
    for chunk_id in ranked_ids:
        if gold_ids is not None:
            is_relevant = chunk_id in gold_ids
        else:
            text = normalize_text(text_by_id.get(chunk_id, ""))
            is_relevant = any(
                normalize_text(fact) in text for fact in fact_phrases or ()
            )
        relevant.append(is_relevant)

    top_five_relevant = relevant[:TOP_K]
    first_rank = next(
        (
            index
            for index, is_relevant in enumerate(top_five_relevant, start=1)
            if is_relevant
        ),
        None,
    )
    dcg = math.fsum(
        1.0 / math.log2(rank + 1)
        for rank, is_relevant in enumerate(relevant[:5], start=1)
        if is_relevant
    )
    if gold_ids is not None:
        total_relevant = len(gold_ids)
    elif total_relevant is None:
        total_relevant = sum(relevant)
    ideal_relevant = min(5, total_relevant)
    ideal_dcg = math.fsum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_relevant + 1)
    )
    return {
        "hit_at_1": float(any(relevant[:1])),
        "hit_at_3": float(any(relevant[:3])),
        "hit_at_5": float(any(relevant[:5])),
        "mrr": 1.0 / first_rank if first_rank is not None else 0.0,
        "ndcg_at_5": dcg / ideal_dcg if ideal_dcg else 0.0,
        "first_relevant_rank": first_rank,
    }


def load_evaluation_inputs(
    cases_path: Path,
    annotations_path: Path,
    chunk_records: Sequence[Mapping[str, object]],
) -> tuple[list[dict], dict[str, dict]]:
    try:
        payload = json.loads(cases_path.read_text(encoding="utf-8"))
        annotations_payload = json.loads(
            annotations_path.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EvaluationInputError(f"Could not load evaluation inputs: {error}") from error

    cases = payload.get("cases") if isinstance(payload, dict) else None
    annotations = (
        annotations_payload.get("cases")
        if isinstance(annotations_payload, dict)
        else None
    )
    if not isinstance(cases, list) or not isinstance(annotations, dict):
        raise EvaluationInputError("evaluation files must contain cases")

    corpus_by_id = {record["chunk_id"]: record for record in chunk_records}
    if len(corpus_by_id) != len(chunk_records):
        raise EvaluationInputError("corpus contains duplicate chunk IDs")

    seen_case_ids = set()
    parsed_annotations = {}
    for case in cases:
        if not isinstance(case, dict):
            raise EvaluationInputError("each evaluation case must be an object")
        case_id = case.get("id")
        query = case.get("query")
        gold_ids = case.get("gold_chunk_ids")
        if not isinstance(case_id, str) or not case_id.strip():
            raise EvaluationInputError("each case must have a nonempty string id")
        if case_id in seen_case_ids:
            raise EvaluationInputError(f"duplicate evaluation case ID: {case_id}")
        seen_case_ids.add(case_id)
        if not isinstance(query, str) or not query.strip():
            raise EvaluationInputError(f"{case_id}: query must not be empty")
        if gold_ids is not None:
            if (
                not isinstance(gold_ids, list)
                or not gold_ids
                or not all(isinstance(item, str) for item in gold_ids)
            ):
                raise EvaluationInputError(
                    f"{case_id}: gold_chunk_ids must be null or a nonempty string list"
                )
            if len(gold_ids) != len(set(gold_ids)):
                raise EvaluationInputError(
                    f"{case_id}: gold_chunk_ids contains duplicates"
                )
            missing = set(gold_ids) - corpus_by_id.keys()
            if missing:
                raise EvaluationInputError(
                    f"{case_id}: gold chunk IDs are absent from the corpus: "
                    f"{', '.join(sorted(missing))}"
                )

        annotation = annotations.get(case_id)
        if not isinstance(annotation, dict):
            raise EvaluationInputError(f"{case_id}: annotation is missing")
        query_type = annotation.get("query_type")
        facts = annotation.get("key_facts")
        if query_type not in QUERY_TYPES:
            raise EvaluationInputError(
                f"{case_id}: query_type must be one of {QUERY_TYPES}"
            )
        if not isinstance(facts, list) or not all(
            isinstance(fact, str) and fact.strip() for fact in facts
        ):
            raise EvaluationInputError(f"{case_id}: key_facts must be strings")
        if gold_ids is None and facts:
            raise EvaluationInputError(
                f"{case_id}: null-gold cases must not have key facts"
            )
        if gold_ids:
            gold_text = " ".join(corpus_by_id[gold_id]["text"] for gold_id in gold_ids)
            normalized_gold = normalize_text(gold_text)
            for fact in facts:
                if normalize_text(fact) not in normalized_gold:
                    raise EvaluationInputError(
                        f"{case_id}: key fact is not supported by its gold chunks: "
                        f"{fact}"
                    )
        parsed_annotations[case_id] = annotation

    if set(annotations) != seen_case_ids:
        raise EvaluationInputError(
            "annotation IDs must match evaluation case IDs exactly"
        )
    return cases, parsed_annotations
