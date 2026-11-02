# Note

import math
from dataclasses import dataclass
from numbers import Real
from typing import Protocol, Sequence

from .retrieval import Embedder


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


class MiniLMEmbedder:
    """Sentence Transformers backend for all-MiniLM-L6-v2.

    The default ``main`` revision is an unpinned branch because this project
    does not currently pin a model commit. This model accepts at most 256
    tokens, including special tokens; longer inputs are rejected rather than
    silently truncated or rewritten.
    """

    MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
    DEFAULT_REVISION = "main"

    def __init__(
        self,
        revision: str = DEFAULT_REVISION,
        device: str = "cpu",
        batch_size: int = 32,
        local_files_only: bool = False,
        model=None,
    ) -> None:
        if not isinstance(revision, str) or not revision.strip():
            raise ValueError("revision must be a non-empty string")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError("batch_size must be a positive integer")
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        if model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as error:
                raise RuntimeError(
                    "MiniLMEmbedder requires the sentence-transformers package"
                ) from error

            model = SentenceTransformer(
                self.MODEL_ID,
                revision=revision,
                device=device,
                local_files_only=local_files_only,
            )

        self._model = model
        self.batch_size = batch_size
        self.max_seq_length = getattr(model, "max_seq_length", None)
        if (
            isinstance(self.max_seq_length, bool)
            or not isinstance(self.max_seq_length, int)
            or self.max_seq_length <= 0
        ):
            raise ValueError("model must expose a positive max_seq_length")

        try:
            self._tokenizer = model[0].tokenizer
        except (IndexError, KeyError, TypeError, AttributeError) as error:
            raise ValueError(
                "model must expose its tokenizer on the first module"
            ) from error

        dimension_getter = getattr(model, "get_embedding_dimension", None)
        if not callable(dimension_getter):
            dimension_getter = getattr(model, "get_sentence_embedding_dimension", None)
        if not callable(dimension_getter):
            raise ValueError("model must expose its sentence embedding dimension")
        dimension = dimension_getter()
        if (
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension <= 0
        ):
            raise ValueError("model must expose a positive embedding dimension")

        self.identity = EmbeddingIdentity(
            model_id=self.MODEL_ID,
            revision=revision,
            preprocessing=(
                "sentence-transformers encode; no task prefix; "
                f"reject over {self.max_seq_length} tokens"
            ),
            dimension=dimension,
        )

    def token_count(self, text: str) -> int:
        """Count tokens including special tokens without truncating."""
        text = _validate_text(text, "text")
        token_ids = self._tokenizer.encode(
            text,
            add_special_tokens=True,
            truncation=False,
        )
        return len(token_ids)

    def _validate_input_length(self, text: str, context: str) -> str:
        text = _validate_text(text, context)
        token_count = self.token_count(text)
        if token_count > self.max_seq_length:
            raise InputTooLongError(
                f"{context} has {token_count} tokens; "
                f"{self.MODEL_ID} supports at most {self.max_seq_length}. "
                "The input was not truncated or rewritten."
            )
        return text

    def _encode(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        if not texts:
            return []
        encoded = self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        if hasattr(encoded, "tolist"):
            encoded = encoded.tolist()
        try:
            vectors = [tuple(float(value) for value in vector) for vector in encoded]
        except (TypeError, ValueError, OverflowError) as error:
            raise EmbeddingError(
                "Sentence Transformer returned invalid vectors"
            ) from error
        if len(vectors) != len(texts):
            raise EmbeddingError(
                f"Sentence Transformer returned {len(vectors)} vectors "
                f"for {len(texts)} inputs"
            )
        return vectors

    def embed(self, text: str) -> Vector:
        """Embed one query or document without modifying its text."""
        text = self._validate_input_length(text, "text")
        return self._encode([text])[0]

    def embed_query(self, text: str) -> Vector:
        """Embed a query with the same preprocessing used for documents."""
        return self.embed(text)

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Vector]:
        """Embed documents in a single Sentence Transformers batch call."""
        if isinstance(texts, (str, bytes, bytearray)):
            raise TypeError("texts must be a sequence of strings, not a string")
        validated_texts = [
            self._validate_input_length(text, f"document {index}")
            for index, text in enumerate(texts)
        ]
        return self._encode(validated_texts)


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
