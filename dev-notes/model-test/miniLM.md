### just some facts 
- install sentence tranformer to be able to use this model 

### Text embeddings
- MiniLM converts text into a numerical vector containing 384 values.
- For example, these two chunks may be semantically similar even if their wording differs:
    - “The scooter loses power during acceleration.”
    - “Reduced vehicle performance when increasing throttle.”
- This helps retrieval find relevant chunks without requiring exact keyword matches

### Lightweight encoder
- The model uses a six-layer Transformer architecture. It is relatively small and suitable for experimenting with CPU-based embeddings in WSL.
- You don't need a hosted embedding API for ordinary inference once the model is downloaded.

### Maximum sequence length: 256 tokens
- This is one of the most important limitations for your project. Longer inputs are truncated by the model, so some information at the end of a long chunk may not influence its embedding.
- A token is not necessarily a word. We'll need to inspect the token lengths of your existing chunks before deciding whether chunking needs to change.

### Cosine similarity
- Once the query and corpus chunks have embeddings, cosine similarity compares their directions to rank likely matches.
- A higher score generally means greater semantic similarity, but it does not guarantee that a chunk contains the 
correct diagnosis or evidence.

### Local inference and caching

- After downloading the model, embeddings can be generated locally on your machine. You don't need to send queries to an external embedding API.
- There are two separate caches to understand:
    - Model cache: downloaded MiniLM model files.
    - Project embedding cache: vectors generated for your corpus chunks.
- Changing the embedding model requires regenerating the project vectors, even if the source chunks remain unchanged.