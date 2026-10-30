**This is for the `scrapper/retrieval.py`** 

### Function overview 

- `ChunkFormatError` and `_validate_chunk()` : Define an error for malformed chunks and validate their structure.
- `load_chunks()` : Read and validate all chunk records from JSONL files.
- `_normalized_vector()` : Validate vectors and normalize them to unit length.
- `CosineIndex` : Store records and vectors, then rank chunks against a query vector.
- `SearchResult` : Package each similarity score with the corresponding original chunk.

### libraries used 

- `dataclass` : automatically generate useful methods for a particular class
- `Real` : checks if a vector formed has real numeric values 
- `typing` : provides runtime support for type hints and annotations
    - These annotations help developers, IDEs, and static type checkers. They don't automatically validate the arguments at runtime. 
    - basically they tell the kind of input and output a function has 


### line by line details 

- `REQUIRED_CHUNK_FIELDS` : this is a set datatype in python 
- `ChunkFormatError` : it is similar to how there was a custom exception check in spring project, basically it takes the exception from python and gives a custom exception desc which makes sense according to the requirement not met. but i need to write out the defn of this class here na and why is it a class first of all, cuz spring had it as okay i forgot 
- `Embedder` : this class gives the embed function for creating the embedding vector of the chunks and queries. why is protocol passed as an argument here and wht does it do and also what is sequence here and this format of embed function is a type hint right ? 
- `SearchResult` : this class has the dataclass annotation which gives it the common methods without defining it and tis was there in spring/java also so give tht analogy and here only the initialiser method is used and what is mapping doing here

does record : mapping[str, object] mean that record is of map data type with the key being a string and an object being the value  

 - The leading underscore in `_validate_chunk` is a Python naming convention that signals the function is intended for internal use within this module. It isn't a security restriction. and this function does a number of checks on it 
- so i dont need to give the defn for the chunkformaterror cuz the calling functions themselves take care of it, kk got it 
- `load_chunks` : it basically does wht the name says that is loads all of them while adding some checks of its own which can exterior to the validate chunks 
- `_normalize_vector` : this is basically converting the vectors to unit vectors using the euclidean norm and this is done cuz cosine similarity uses angles and not the distance so its fine 
- `CosineIndex` : this is the main part which brings the chunks and vectors together and gives the most similar chunks. the init function here is to add the basic checks and initialise the chunks properly 

### search
- This is the core mathematical operation.
- Because both the query and chunk vectors have already been normalized, their dot product equals their cosine similarity
    `enumerate(self._vectors)`

- Loops through each stored vector while providing its index.
    `zip(query, vector)`

- Pairs the components at matching positions. For example, [0.6, 0.8] and [0.8, 0.6] produce the pairs (0.6, 0.8) and (0.8, 0.6).
    `query_value * vector_value`

- Multiplies each pair of components.
    `math.fsum(...)`

- Adds those products with improved floating-point summation accuracy.
    `max(-1.0, min(1.0, score))`

- Clamps the score to the theoretical cosine-similarity range of \([-1,1]\). This protects against tiny floating-point rounding errors outside that range.
    `scored.append((score, index))`

- Stores the similarity score alongside the vector's index. That index is how the code later finds the corresponding original chunk.

- For normalized vectors, a score near 1 indicates a small angle between them, a score near 0 indicates near-orthogonality, and a score near -1 indicates opposite directions. In real embedding spaces, the meaning of a useful score depends on the model and task.


