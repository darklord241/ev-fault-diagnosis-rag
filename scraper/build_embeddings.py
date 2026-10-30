from vector_search.embedding import MiniLMEmbedder
from vector_search.index_embeddings import (
    DEFAULT_CACHE_PATH,
    load_or_build_embeddings,
)
from vector_search.retrieval import load_chunks


def main():
    print("Loading and validating chunks...")
    records = load_chunks()

    if not records:
        raise SystemExit("No chunks found. Aborting.")

    print(f"Chunks loaded: {len(records)}")
    print("Initializing MiniLM...")
    backend = MiniLMEmbedder(device="cpu")

    print(f"Model: {backend.identity.model_id}")
    print(f"Token limit: {backend.max_seq_length}")
    print(f"Cache path: {DEFAULT_CACHE_PATH}")
    print("Building or loading embeddings...")

    cache = load_or_build_embeddings(records, backend)

    print("\nEmbedding cache ready.")
    print(f"Chunks: {len(records)}")
    print(f"Vectors: {len(cache.vectors_by_chunk_id)}")
    print(f"Cache hit: {cache.cache_hit}")
    if cache.rebuild_reason:
        print(f"Reason: {cache.rebuild_reason}")
    print(f"Cache exists: {DEFAULT_CACHE_PATH.is_file()}")
    print(f"Vector dimension: {backend.identity.dimension}")

    assert len(cache.vectors_by_chunk_id) == len(records)
    print("Validation passed: one cached vector per chunk.")


if __name__ == "__main__":
    main()
