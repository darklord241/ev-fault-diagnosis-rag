# Retrieval Pipeline Flow

## 1. Purpose

This document describes how the EV fault-diagnosis RAG retrieval pipeline accepts a user query, processes it, retrieves relevant chunks from the available corpus, and returns ranked results.

The current focus is to validate the retrieval pipeline independently before integrating its output into the final fault-diagnosis response.

## 2. High-Level Flow

```text
User Query
    |
    v
Retriever.search(query, top_k)
    |
    v
Validate Query and top_k
    |
    v
Generate Query Embedding
    |
    v
Load / Reuse Cached Chunk Embeddings
    |
    v
Cosine Similarity Search
    |
    v
Rank Matching Chunks
    |
    v
Return Top-k Search Results
    |
    v
Inspect Results and Evaluate Retrieval Quality
```

The chunk embeddings are generated and cached separately from query-time retrieval when the index is first built or needs rebuilding. Subsequent searches can reuse the cached vectors.

## 3. Main Components

| Component | Responsibility |
|---|---|
| `retrieval.py` | Loads and validates the existing chunked corpus and provides the cosine-similarity index and retrieval-related structures. |
| `embedding.py` | Defines the model-independent embedding interface, embedding identity, and vector validation. |
| `index_embeddings.py` | Validates corpus records, manages the embedding cache, generates missing vectors, and persists them. |
| `retriever.py` | Connects the embedding backend, cached corpus embeddings, and similarity index to expose the search interface. |
| `retrieval_eval.json` | Stores retrieval evaluation cases used to assess search quality. |

## 4. Input

The primary runtime input is a text query describing an EV problem, symptom, warning, or fault.

Example:

```text
The scooter loses power while accelerating.
```

The search interface also accepts `top_k`, which controls the maximum number of results requested.

The current corpus consists of previously chunked FAQ and PDF-derived records. The staged batch of 15 PDFs is a separate ingestion task and should not be considered part of the active retrieval corpus until it has been processed and validated.

## 5. Processing Flow

### Step 1 — Query validation

`Retriever.search()` validates the query and `top_k`. Invalid inputs should fail clearly rather than silently producing misleading results.

### Step 2 — Query embedding

The embedding backend converts the query into a numeric vector. The same embedding space and compatible preprocessing must be used for both query and corpus vectors.

A real embedding model has not yet been selected and validated. Until then, test backends are used for controlled tests.

### Step 3 — Corpus embedding preparation

`load_or_build_embeddings()` manages the chunk-vector cache.

- If the cache is valid for the current corpus and embedding identity, existing vectors can be reused.
- If the cache is missing or invalid, the required chunk embeddings are generated and the cache is rebuilt.
- Cache writes use an atomic replacement mechanism to protect the previous cache if replacement fails.

The corpus fingerprint is based on chunk IDs and text, so changes to those fields can invalidate the cache.

### Step 4 — Similarity search

The query vector is compared with the stored chunk vectors using cosine similarity. The index ranks the candidates by similarity.

### Step 5 — Result construction

The retriever returns ranked `SearchResult` objects for the requested number of matches. Each result should be inspected to determine which fields are available, such as chunk identity, text, similarity score, and source metadata.

The exact result schema must be confirmed from the implementation rather than assumed here.

## 6. Output

The pipeline's output is a ranked collection of retrieved chunks, not a final natural-language diagnosis.

For each result, inspect:

- The rank and similarity score.
- The chunk text and whether it answers the query.
- The source document and provenance metadata, where available.
- Whether the evidence actually supports a possible explanation or troubleshooting step.

A high similarity score is not, by itself, proof that a chunk is factually correct or relevant.

## 7. Current Scope and Limitations

- The retrieval implementation is separated from answer generation.
- A production embedding model has not yet been selected.
- The current chunked corpus is the retrieval source; the 15 staged PDFs still need to be ingested.
- Retrieval quality must be measured using representative queries and expected relevant evidence.
- Output formatting and downstream use have not yet been finalized.

## 8. Next Steps

### Phase A — Model setup and pipeline validation

1. Select a suitable embedding model and record its model identity, revision, dimension, and preprocessing configuration.
2. Implement or configure the real embedding backend.
3. Generate the corpus embeddings and verify that the cache is built correctly.
4. Run representative queries through the full pipeline.
5. Check result ordering, scores, source metadata, cache reuse, and failure handling.
6. Run the existing automated tests and the retrieval evaluation cases.

### Phase B — Output inspection and integration

1. Inspect the actual `SearchResult` schema.
2. Decide how retrieved chunks should be presented to a downstream component.
3. Preserve source identifiers and provenance when transforming the output.
4. Establish how the retrieved evidence will be used in the final fault-diagnosis workflow.

### Phase C — Retrieval experiments

1. Establish a baseline using the initial embedding model and current chunking strategy.
2. Experiment with different chunk lengths and overlap settings.
3. Compare alternative embedding models using the same evaluation queries.
4. Track retrieval quality, latency, and resource requirements.
5. Rebuild the embedding cache whenever the corpus text, chunking, or embedding identity changes.

Change one major variable at a time so that improvements can be attributed to a specific change.

### Phase D — Ingest the 15 staged PDFs

1. Inspect the documents and confirm their provenance.
2. Extract and validate their text.
3. Apply a consistent chunking strategy.
4. Preserve source-document identifiers and page or section information where available.
5. Add the new chunks to the corpus through the existing ingestion workflow.
6. Validate the generated chunks before rebuilding the index.
7. Rerun the retrieval evaluation against both the existing sources and the newly ingested documents.

## 9. Success Criteria

The pipeline is ready for downstream integration when:

- A real embedding backend works end to end.
- The corpus and query vectors are compatible.
- Repeated queries correctly reuse a valid embedding cache.
- Retrieved results contain inspectable text and traceable source information.
- Representative queries return useful evidence consistently.
- Retrieval quality can be compared reproducibly across models and chunking configurations.
