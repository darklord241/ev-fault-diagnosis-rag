## retriever.py

### overview

**Where does it fall in the overall flow?**

- `retriever.py` connects the embedding backend, embedding cache, and cosine-similarity index.
- It acts as the main interface for searching the corpus.
- It accepts a user query, generates its embedding, retrieves relevant chunks, and returns ranked results.
- It does not display results or generate an LLM answer itself.

### code sections

#### imports

- `EmbeddingBackend`, `embed_query` from `embedding.py`: provide the embedding interface and convert the query into a validated, normalized vector.
- `DEFAULT_CACHE_PATH`, `load_or_build_embeddings` from `index_embeddings.py`: load existing cached chunk vectors or generate and save them if necessary.
- `CHUNKED_DIR`, `CosineIndex`, `SearchResult`, `load_chunks` from `retrieval.py`: provide the corpus location, cosine index, result structure, and chunk-loading functionality.

#### Retriever class

- `Retriever` connects the different components into a usable retrieval pipeline.
- It accepts the chunk records, an embedding backend, and an optional cache path.

#### constructor: `__init__()`

- Stores the embedding backend in `self._backend`.
- Copies the supplied records into a list and stores them in `self._records`.
- Calls `load_or_build_embeddings()` to obtain compatible cached vectors or generate new ones.
- Retrieves vectors in the same order as the corresponding chunk records by looking up each `chunk_id`.
- Constructs a `CosineIndex` using the original records and their vectors.
- Stores `cache_hit` and `cache_rebuild_reason` so the caller can check whether the cache was reused or rebuilt.

**Important:** The constructor initializes the index immediately. Creating a `Retriever` can therefore involve reading the corpus, loading the cache, and potentially generating embeddings.

#### class method: `from_directory()`

- Provides a convenient way to create a `Retriever` from a directory containing chunked JSONL files.
- Calls `load_chunks()` to load and validate the chunk records.
- Passes those records, the backend, and the cache path to the constructor.
- Uses `CHUNKED_DIR` and `DEFAULT_CACHE_PATH` by default, but both paths can be overridden.

#### search method: `search()`

- Accepts a query and the number of results to return (`top_k`, default `5`).
- Validates that:
    - `top_k` is an integer and not a boolean.
    - `top_k` is greater than zero.
    - The query is a string.
    - The query is not empty or whitespace-only.
- Returns an empty list if the corpus contains no records.
- Calls `embed_query()` to convert the query into a validated, normalized vector.
- Calls `self._index.search()` to rank the chunks by cosine similarity and return up to `top_k` results.

#### private attributes

- `self._backend`: The embedding backend used to generate query vectors.
- `self._records`: The original corpus chunk records.
- `self._index`: The initialized cosine-similarity index.

The leading underscore indicates that these attributes are intended for internal use.

#### cache status attributes

- `self.cache_hit`: Indicates whether the existing embedding cache was reused.
- `self.cache_rebuild_reason`: Explains why the cache was rebuilt, or why it did not already exist.

### overall flow

1. `from_directory()` loads the chunked JSONL corpus.
2. `__init__()` loads or builds the chunk embeddings.
3. `__init__()` constructs the `CosineIndex`.
4. `search()` embeds the user's query.
5. `CosineIndex.search()` ranks the chunks by cosine similarity.
6. The method returns ranked `SearchResult` objects containing the original chunk records and their similarity scores.

### overall takeaway

`retriever.py` is the orchestration layer for retrieval. It connects the chunk corpus, cached document embeddings, query embedding, and cosine index behind a simple interface: initialize a `Retriever`, then call `search()` whenever a query needs to be answered.