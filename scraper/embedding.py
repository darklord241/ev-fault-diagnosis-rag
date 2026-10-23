# Model-neutral contracts and validation for text embedding backends.

import math
from dataclasses import dataclass
from numbers import Real
from typing import Protocol, Sequence

from retrieval import Embedder


Vector = Sequence[float]


class EmbeddingError(ValueError):
    """Raised when embedding input or output is invalid."""


class InputTooLongError(EmbeddingError):
    """Raised by a backend when text exceeds its supported token limit."""


@dataclass(frozen=True)
class EmbeddingIdentity:
    """Stable model and preprocessing identity for future cache validation."""

    model_id: str
    revision: str
    preprocessing: str
    dimension: int

    def __post_init__(self):
        for field_name in ("model_id", "revision", "preprocessing"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        if isinstance(self.dimension, bool) or not isinstance(self.dimension, int):
            raise TypeError("dimension must be a positive integer")
        if self.dimension <= 0:
            raise ValueError("dimension must be a positive integer")


class EmbeddingBackend(Embedder, Protocol):
    """Stage 1 embedder plus stable identity needed by the embedding cache."""

    identity: EmbeddingIdentity


class TaskAwareBackend(Protocol):
    """Optional methods for providers with query/document-specific formatting."""

    def embed_query(self, text: str) -> Vector:
        """Embed a query using this model's query-specific preprocessing."""

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Vector]:
        """Embed documents as a batch using document-specific preprocessing."""


def _validate_text(text: str, context: str) -> str:
    if not isinstance(text, str):
        raise TypeError(f"{context} must be a string")
    if not text.strip():
        raise ValueError(f"{context} must not be empty")
    return text


def validate_vector(
    vector: Vector, expected_dimension: int, context: str
) -> tuple[float, ...]:
    """Validate and L2-normalize one finite, non-zero vector."""
    if isinstance(vector, (str, bytes, bytearray)):
        raise EmbeddingError(f"{context} must be a numeric vector")
    try:
        values = tuple(vector)
    except TypeError as error:
        raise EmbeddingError(f"{context} must be a numeric vector") from error

    if len(values) != expected_dimension:
        raise EmbeddingError(
            f"{context} has {len(values)} dimensions; expected {expected_dimension}"
        )
    if not values:
        raise EmbeddingError(f"{context} must not be empty")

    result = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise EmbeddingError(f"{context}[{index}] must be a finite number")
        try:
            numeric_value = float(value)
        except (OverflowError, ValueError) as error:
            raise EmbeddingError(
                f"{context}[{index}] must be a finite number"
            ) from error
        if not math.isfinite(numeric_value):
            raise EmbeddingError(f"{context}[{index}] must be a finite number")
        result.append(numeric_value)

    scale = max(abs(value) for value in result)
    if scale == 0:
        raise EmbeddingError(f"{context} must not have zero norm")
    scaled = tuple(value / scale for value in result)
    norm = math.hypot(*scaled)
    return tuple(value / norm for value in scaled)


def embed_query(backend: EmbeddingBackend, text: str) -> tuple[float, ...]:
    """Embed a query with task-specific paths only when both paths exist."""
    text = _validate_text(text, "query")
    query_encoder = getattr(backend, "embed_query", None)
    document_encoder = getattr(backend, "embed_documents", None)
    if callable(query_encoder) and callable(document_encoder):
        vector = query_encoder(text)
    else:
        vector = backend.embed(text)
    return validate_vector(vector, backend.identity.dimension, "query embedding")


def embed_documents(
    backend: EmbeddingBackend, texts: Sequence[str]
) -> list[tuple[float, ...]]:
    """Batch-embed and validate documents in input order.

    Empty input returns an empty list without calling the backend.
    """
    if isinstance(texts, (str, bytes, bytearray)):
        raise TypeError("texts must be a sequence of strings, not a string")
    texts = list(texts)
    if not texts:
        return []

    validated_texts = [
        _validate_text(text, f"document {index}")
        for index, text in enumerate(texts)
    ]
    query_encoder = getattr(backend, "embed_query", None)
    document_encoder = getattr(backend, "embed_documents", None)
    if callable(query_encoder) and callable(document_encoder):
        vectors = document_encoder(validated_texts)
    else:
        vectors = [backend.embed(text) for text in validated_texts]
    if len(vectors) != len(validated_texts):
        raise EmbeddingError(
            f"backend returned {len(vectors)} vectors for "
            f"{len(validated_texts)} documents"
        )
    return [
        validate_vector(
            vector, backend.identity.dimension, f"document embedding {index}"
        )
        for index, vector in enumerate(vectors)
    ]
