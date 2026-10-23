## embedding.py 

### overview 
where does it fall in the overall flow ? 

### imp notices
#### there are two normalizations ?  
- So yes, the vector is being normalized twice along this path.

- The second normalization generally doesn't change the vector meaningfully: a unit vector normalized again is still a unit vector, apart from tiny floating-point rounding differences.
Is that redundant?
- Potentially, yes — but there's a design trade-off.
    - validate_vector() protects the embedding interface.
    - _normalized_vector() protects the retrieval interface independently.
- The second function could be useful if someone later calls CosineIndex directly with vectors that did not come through embedding.py. The retrieval module would still protect itself.

### code sections 

#### error classes
- first two error classes are there only in case you want to take care of that error in a different way 
- Custom exception: Gives a specific name to an error condition so other code can recognize and handle it.
- Exception inheritance: InputTooLongError inherits from EmbeddingError, which inherits from ValueError. This creates an error hierarchy.

#### embedding identity class 
- so this has four data members which are 
    1. model_id : Identifies the embedding model (str)
    2. revision : Identifies the particular version of the model being used
    3. preprocessing : Identifies how text is prepared before embedding 
    4. dimension : Number of values in each embedding vector

#### backends
- a protocol defines what an object must provide, rather than how it implements it.
- An embedding backend must provide:
    - embed(text) — turn text into a vector.
    - identity — identify the model, revision, preprocessing and vector dimension.
- taskaware : is required if both are present and can distinguish the two roles because some models use different preprocessing for each. 

#### validation 
- text validation : checks that the input is a non-empty string.The context parameter helps produce an informative error
- vector validation : checks the output of an embedding backend. It verifies that the vector has the expected dimension, contains valid finite numeric values, and isn't a zero vector. It also normalizes the vector to unit length.


## index_embeddings.py

### overview

**Where does it fall in the overall flow?**

- This file manages the generation, validation, storage, and reuse of embedding vectors for corpus chunks.
- It sits between the embedding backend (`embedding.py`) and the retrieval pipeline (`retriever.py`).
- It prevents unnecessary embedding generation by reusing a valid cache when possible.
- The default cache is stored at `scraper/data/embeddings/chunk_vectors.json`.

### important notices

#### cache hit vs cache miss

- **Cache hit:** An existing cache is compatible with the current corpus and embedding configuration, and its vectors pass validation.
    - Returns the existing vectors.
    - Skips embedding generation and cache writing.
- **Cache miss:** The cache is missing, incompatible, or invalid.
    - Generates new embeddings.
    - Writes the new vectors to the cache.
    - Returns the newly generated vectors.
- Even on a cache hit, the function still validates the corpus, computes its fingerprint, and reads and validates the cache.

#### corpus fingerprint

- `_corpus_fingerprint()` creates a single SHA-256 fingerprint for the entire corpus using each chunk's `chunk_id` and `text`.
- Chunks are sorted by ID before hashing, so changing their input order alone doesn't change the fingerprint.
- If a chunk ID or text changes, the fingerprint changes and the existing cache is rebuilt.
- Changes to other fields, such as metadata, don't affect this fingerprint unless they also change the ID or text.

#### atomic cache replacement

- `_write_cache()` writes the new cache to a temporary file before replacing the existing cache with `os.replace()`.
- This helps prevent an incomplete write from corrupting a previously valid cache.
- If writing or replacement fails with an `OSError`, the function attempts to remove the temporary file and raises the error.

### code sections

#### cache version and default cache path

- `CACHE_VERSION`: Identifies the cache format version. It can be incremented when the cache structure or format changes.
- `DEFAULT_CACHE_PATH`: Defines the default location of the saved embedding cache relative to this Python file.

#### error class

- `EmbeddingCacheError`: A custom exception used when source chunks cannot safely map to cached vectors, such as when chunk IDs are invalid or duplicated.

#### embedding cache class

- `EmbeddingCache` is a frozen dataclass containing:
    1. `vectors_by_chunk_id`: A mapping where each chunk ID is the key and its embedding vector is the value.
    2. `cache_hit`: Indicates whether an existing compatible cache was reused.
    3. `rebuild_reason`: Explains why the cache could not be reused, or why it needed to be created.
- `rebuild_reason` describes the current cache decision; it is not a persistent history of previous rebuilds.

#### corpus fingerprint function

- `_corpus_fingerprint(records)`:
    - Sorts the records by `chunk_id`.
    - Serializes each chunk's ID and text into a consistent byte representation.
    - Feeds these bytes into a SHA-256 digest.
    - Returns one hexadecimal fingerprint for the entire corpus.

#### record validation function

- `_validate_records(records)` validates the corpus before loading or generating embeddings.
- It checks that:
    - Every record is a mapping.
    - Every chunk has a nonempty string ID and text.
    - `_validate_chunk()` performs the existing chunk-level validation.
    - Chunk IDs are unique.
- Unique IDs are necessary because the cache uses them as dictionary keys to associate chunks with their vectors.

#### read cache function

- `_read_cache()` attempts to load and validate an existing JSON cache.
- It checks:
    - Whether the file can be read and parsed.
    - Cache format version.
    - Embedding model identity and preprocessing configuration.
    - Corpus fingerprint.
    - Expected vector dimension.
    - Validity and completeness of cached vectors and chunk IDs.
- If all checks pass, it returns an `EmbeddingCache` with `cache_hit=True`.
- If a check fails, it returns a cache miss with a rebuild reason.
- It validates cached vectors using `validate_vector()` from `embedding.py`.
- It does **not** build the cosine-similarity index; it only loads and validates cached vectors.

#### write cache function

- `_write_cache()` serializes the embedding vectors and related metadata into JSON.
- The cache contains:
    - Cache version.
    - Embedding identity.
    - Vector dimension.
    - Corpus fingerprint.
    - Chunk IDs and their vectors.
- It writes to a temporary file and then replaces the destination cache atomically.

#### load or build embeddings function

- `load_or_build_embeddings()` coordinates the complete caching process.
- Its flow is:
    1. Validate the source records.
    2. Compute the corpus fingerprint.
    3. Attempt to load the existing cache.
    4. If the cache is valid, return it immediately.
    5. Otherwise, generate embeddings using `embed_documents()`.
    6. Map each vector to its corresponding chunk ID.
    7. Write the new cache and return the result.
- The line `if cached.cache_hit: return cached` is the key to avoiding unnecessary embedding generation.
- The function prepares and stores vectors for retrieval; it does not itself construct the cosine index.

### overall takeaway

`index_embeddings.py` is the persistence and cache-management layer for chunk embeddings. It reuses valid vectors when possible and rebuilds the cache when the corpus, embedding configuration, or cached data no longer matches expectations.
