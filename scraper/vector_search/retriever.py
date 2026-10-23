"""Small model-independent retrieval flow over the chunked JSONL corpus."""

from pathlib import Path
from typing import Mapping, Sequence

from .embedding import EmbeddingBackend, embed_query
from .index_embeddings import DEFAULT_CACHE_PATH, load_or_build_embeddings
from .retrieval import CHUNKED_DIR, CosineIndex, SearchResult, load_chunks


class Retriever:
    """Connect a pluggable embedding backend, cache, and exact cosine index."""

    def __init__(
        self,
        records: Sequence[Mapping[str, object]],
        backend: EmbeddingBackend,
        cache_path: Path = DEFAULT_CACHE_PATH,
    ) -> None:
        self._backend = backend
        self._records = list(records)
        cache = load_or_build_embeddings(self._records, backend, cache_path)
        vectors = [
            cache.vectors_by_chunk_id[record["chunk_id"]]
            for record in self._records
        ]
        self._index = CosineIndex(self._records, vectors)
        self.cache_hit = cache.cache_hit
        self.cache_rebuild_reason = cache.rebuild_reason

    @classmethod
    def from_directory(
        cls,
        backend: EmbeddingBackend,
        chunk_directory: Path = CHUNKED_DIR,
        cache_path: Path = DEFAULT_CACHE_PATH,
    ) -> "Retriever":
        """Load validated JSONL chunks and initialize their embeddings."""
        return cls(load_chunks(chunk_directory), backend, cache_path)

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Return ranked original chunk records and cosine similarity scores."""
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise TypeError("top_k must be a positive integer")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if not query.strip():
            raise ValueError("query must not be empty")
        if not self._records:
            return []

        query_vector = embed_query(self._backend, query)
        return self._index.search(query_vector, top_k=top_k)
