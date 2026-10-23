"""Build and validate an ID-keyed JSON cache of chunk embeddings."""

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

from embedding import EmbeddingBackend, EmbeddingError, embed_documents, validate_vector
from retrieval import _validate_chunk


CACHE_VERSION = 1
DEFAULT_CACHE_PATH = (
    Path(__file__).resolve().parent / "data" / "embeddings" / "chunk_vectors.json"
)


class EmbeddingCacheError(ValueError):
    """Raised when source chunks cannot safely map to cached vectors."""


@dataclass(frozen=True)
class EmbeddingCache:
    vectors_by_chunk_id: Mapping[str, tuple[float, ...]]
    cache_hit: bool
    rebuild_reason: str | None = None


def _corpus_fingerprint(records: Sequence[Mapping[str, object]]) -> str:
    digest = hashlib.sha256()
    for chunk_id, text in sorted(
        ((record["chunk_id"], record["text"]) for record in records),
        key=lambda item: item[0],
    ):
        digest.update(
            json.dumps(
                [chunk_id, text], ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _validate_records(records: Sequence[Mapping[str, object]]) -> None:
    seen_ids = set()
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise EmbeddingCacheError(f"chunk {index} must be a mapping")
        chunk_id = record.get("chunk_id")
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise EmbeddingCacheError(
                f"chunk {index} has a missing or invalid chunk_id"
            )
        text = record.get("text")
        if not isinstance(text, str) or not text.strip():
            raise EmbeddingCacheError(f"chunk {chunk_id} has missing or empty text")
        _validate_chunk(record, f"chunk {index}")
        if chunk_id in seen_ids:
            raise EmbeddingCacheError(f"duplicate chunk_id: {chunk_id}")
        seen_ids.add(chunk_id)


def _read_cache(
    cache_path: Path,
    records: Sequence[Mapping[str, object]],
    backend: EmbeddingBackend,
    fingerprint: str,
) -> EmbeddingCache:
    try:
        with cache_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        return EmbeddingCache({}, False, f"cache cannot be read: {error}")

    if not isinstance(payload, dict):
        return EmbeddingCache({}, False, "cache root is not an object")
    if payload.get("cache_version") != CACHE_VERSION:
        return EmbeddingCache({}, False, "cache version differs")
    if payload.get("embedding_identity") != asdict(backend.identity):
        return EmbeddingCache({}, False, "embedding model or preprocessing differs")
    if payload.get("corpus_fingerprint") != fingerprint:
        return EmbeddingCache({}, False, "chunk IDs or text changed")
    if payload.get("dimension") != backend.identity.dimension:
        return EmbeddingCache({}, False, "vector dimension differs")

    entries = payload.get("embeddings")
    if not isinstance(entries, list):
        return EmbeddingCache({}, False, "embedding entries are missing or invalid")

    expected_ids = {record["chunk_id"] for record in records}
    vectors_by_chunk_id = {}
    try:
        for entry in entries:
            if not isinstance(entry, dict):
                raise EmbeddingCacheError("cache entry is not an object")
            chunk_id = entry.get("chunk_id")
            if not isinstance(chunk_id, str) or chunk_id not in expected_ids:
                raise EmbeddingCacheError("cache contains an unexpected chunk_id")
            if chunk_id in vectors_by_chunk_id:
                raise EmbeddingCacheError(f"cache repeats chunk_id {chunk_id}")
            vectors_by_chunk_id[chunk_id] = validate_vector(
                entry.get("vector"),
                backend.identity.dimension,
                f"cached vector for {chunk_id}",
            )
    except (EmbeddingCacheError, EmbeddingError, TypeError):
        return EmbeddingCache({}, False, "cache entries are invalid")

    if set(vectors_by_chunk_id) != expected_ids:
        return EmbeddingCache({}, False, "cached chunk IDs do not match the corpus")
    return EmbeddingCache(vectors_by_chunk_id, True)


def _write_cache(
    cache_path: Path,
    records: Sequence[Mapping[str, object]],
    backend: EmbeddingBackend,
    fingerprint: str,
    vectors_by_chunk_id: Mapping[str, Sequence[float]],
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cache_version": CACHE_VERSION,
        "embedding_identity": asdict(backend.identity),
        "dimension": backend.identity.dimension,
        "corpus_fingerprint": fingerprint,
        "embeddings": [
            {
                "chunk_id": record["chunk_id"],
                "vector": list(vectors_by_chunk_id[record["chunk_id"]]),
            }
            for record in sorted(records, key=lambda item: item["chunk_id"])
        ],
    }

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=cache_path.parent,
            prefix=f".{cache_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary_path = Path(file.name)
            json.dump(payload, file, ensure_ascii=False, indent=2)
            file.write("\n")
        os.replace(temporary_path, cache_path)
    except OSError:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def load_or_build_embeddings(
    records: Sequence[Mapping[str, object]],
    backend: EmbeddingBackend,
    cache_path: Path = DEFAULT_CACHE_PATH,
) -> EmbeddingCache:
    """Load compatible cached vectors, or batch-embed and atomically replace it."""
    _validate_records(records)
    cache_path = Path(cache_path)
    fingerprint = _corpus_fingerprint(records)

    if cache_path.exists():
        cached = _read_cache(cache_path, records, backend, fingerprint)
        if cached.cache_hit:
            return cached
        rebuild_reason = cached.rebuild_reason
    else:
        rebuild_reason = "cache does not exist"

    vectors = embed_documents(
        backend, [record["text"] for record in records]
    )
    vectors_by_chunk_id = {
        record["chunk_id"]: vector for record, vector in zip(records, vectors)
    }
    _write_cache(cache_path, records, backend, fingerprint, vectors_by_chunk_id)
    return EmbeddingCache(vectors_by_chunk_id, False, rebuild_reason)
