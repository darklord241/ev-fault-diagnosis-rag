"""Model-independent loading and exact cosine search over chunk vectors."""

import json
import math
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Mapping, Protocol, Sequence


CHUNKED_DIR = Path(__file__).resolve().parent / "data" / "chunked"
REQUIRED_CHUNK_FIELDS = {
    "chunk_id",
    "record_id",
    "source_document",
    "source_type",
    "content_type",
    "title",
    "text",
    "provenance",
    "metadata",
}


class ChunkFormatError(ValueError):
    """Raised when a JSONL chunk cannot be used for retrieval."""


class Embedder(Protocol):
    """Interface for a future model that embeds one piece of text at a time."""

    def embed(self, text: str) -> Sequence[float]:
        """Return the vector representation of ``text``."""


@dataclass(frozen=True)
class SearchResult:
    """A similarity score paired with the original, unchanged chunk record."""

    score: float
    record: Mapping[str, object]


def _validate_chunk(record: object, source: str) -> dict:
    if not isinstance(record, dict):
        raise ChunkFormatError(f"{source}: expected a JSON object")

    missing = REQUIRED_CHUNK_FIELDS - record.keys()
    if missing:
        raise ChunkFormatError(
            f"{source}: missing required fields: {', '.join(sorted(missing))}"
        )

    for field in (
        "chunk_id",
        "record_id",
        "source_document",
        "source_type",
        "content_type",
        "title",
        "text",
    ):
        if not isinstance(record[field], str):
            raise ChunkFormatError(f"{source}: {field} must be a string")
        if field != "title" and not record[field].strip():
            raise ChunkFormatError(f"{source}: {field} must not be empty")

    if not isinstance(record["provenance"], dict):
        raise ChunkFormatError(f"{source}: provenance must be a JSON object")
    if not isinstance(record["metadata"], dict):
        raise ChunkFormatError(f"{source}: metadata must be a JSON object")

    return record


def load_chunks(directory: Path = CHUNKED_DIR) -> list[dict]:
    """Load and validate all ``*.jsonl`` chunks in deterministic file order.

    Invalid JSON or records raise ``ChunkFormatError`` with the file and line
    number. An existing directory with no JSONL files returns an empty list.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Chunk directory does not exist: {directory}")

    records = []
    for path in sorted(directory.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ChunkFormatError(
                        f"{path}:{line_number}: invalid JSON: {error.msg}"
                    ) from error
                records.append(_validate_chunk(record, f"{path}:{line_number}"))
    return records


def _normalized_vector(vector: Sequence[float], source: str) -> tuple[float, ...]:
    if isinstance(vector, (str, bytes, bytearray)):
        raise ValueError(f"{source} must be a non-empty numeric vector")

    try:
        values = tuple(vector)
    except TypeError as error:
        raise ValueError(f"{source} must be a non-empty numeric vector") from error

    if not values:
        raise ValueError(f"{source} must not be empty")

    normalized_values = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{source}[{index}] must be a finite number")
        try:
            numeric_value = float(value)
        except (OverflowError, ValueError) as error:
            raise ValueError(f"{source}[{index}] must be a finite number") from error
        if not math.isfinite(numeric_value):
            raise ValueError(f"{source}[{index}] must be a finite number")
        normalized_values.append(numeric_value)

    scale = max(abs(value) for value in normalized_values)
    if scale == 0:
        raise ValueError(f"{source} must not have zero norm")
    scaled = tuple(value / scale for value in normalized_values)
    norm = math.hypot(*scaled)
    return tuple(value / norm for value in scaled)


class CosineIndex:
    """Exact in-memory cosine search over precomputed chunk vectors."""

    def __init__(
        self,
        records: Sequence[Mapping[str, object]],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        if len(records) != len(vectors):
            raise ValueError(
                f"Expected one vector per chunk: {len(records)} chunks, "
                f"{len(vectors)} vectors"
            )

        self._records = []
        for index, record in enumerate(records):
            self._records.append(_validate_chunk(record, f"chunk {index}"))

        self._vectors = [
            _normalized_vector(vector, f"chunk vector {index}")
            for index, vector in enumerate(vectors)
        ]
        dimensions = {len(vector) for vector in self._vectors}
        if len(dimensions) > 1:
            raise ValueError("All chunk vectors must have the same dimensions")
        self._dimensions = next(iter(dimensions), None)

    def search(
        self, query_vector: Sequence[float], top_k: int = 5
    ) -> list[SearchResult]:
        """Return the top ``top_k`` chunks, with ties kept in input order."""
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise TypeError("top_k must be a positive integer")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        query = _normalized_vector(query_vector, "query vector")
        if self._dimensions is not None and len(query) != self._dimensions:
            raise ValueError(
                f"Query vector has {len(query)} dimensions; "
                f"expected {self._dimensions}"
            )

        scored = []
        for index, vector in enumerate(self._vectors):
            score = math.fsum(
                query_value * vector_value
                for query_value, vector_value in zip(query, vector)
            )
            score = max(-1.0, min(1.0, score))
            scored.append((score, index))

        scored.sort(key=lambda item: (-item[0], item[1]))
        return [
            SearchResult(score=score, record=self._records[index])
            for score, index in scored[:top_k]
        ]
