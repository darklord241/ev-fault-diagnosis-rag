# `scraper/retrieval.py` — Notes

## 1. Function and class overview

- **`ChunkFormatError`**: A custom exception used to indicate that a chunk record does not follow the expected format.
- **`_validate_chunk()`**: Checks whether a chunk record contains the required fields and whether those fields have valid types and values.
- **`load_chunks()`**: Reads chunk records from JSONL files, validates them, and returns them as a list.
- **`_normalized_vector()`**: Validates a numerical vector and converts it into a unit vector with a Euclidean norm of 1.
- **`Embedder`**: A protocol that defines the interface expected from an embedding implementation.
- **`SearchResult`**: Stores a similarity score together with its corresponding original chunk record.
- **`CosineIndex`**: Stores chunk records and their normalized vectors, then ranks the chunks according to their cosine similarity with a query vector.

---

## 2. Libraries and imports used

### `dataclass`

A decorator from Python's `dataclasses` module that automatically generates common methods for a class, such as `__init__()` and `__repr__()`.

It reduces boilerplate when a class mainly exists to store data.

**Java analogy:** Similar in purpose to using a Java `record` or generating constructors and other standard methods with Lombok. They are not exactly equivalent, but they solve similar problems.

### `Real`

Imported from `numbers`. It is an abstract numeric type used to check whether a value is a real-number type.

In this project, it helps reject values that should not be accepted as vector components. The code also separately rejects booleans and non-finite numbers such as `NaN` and infinity.

### `typing`

Provides tools for type annotations and interfaces, including `Mapping`, `Protocol`, and `Sequence`.

Type annotations describe the expected types of inputs, outputs, and variables. They help developers, IDEs, and static type checkers understand the code, but they do not automatically validate function arguments at runtime.

### `math`

Provides mathematical functions used by the retrieval implementation, including:

- `math.hypot()` — calculates a Euclidean norm in a numerically robust way.
- `math.fsum()` — adds floating-point values with improved summation accuracy.
- `math.isfinite()` — checks that a number is neither infinite nor `NaN`.

### `json` and `pathlib.Path`

- `json` parses JSON objects from JSONL files.
- `Path` represents filesystem paths and provides methods for locating and reading files.

---

## 3. `REQUIRED_CHUNK_FIELDS`

```python
REQUIRED_CHUNK_FIELDS = {
    "chunk_id",
    "record_id",
    "source_document",
    "source_type",
    "content_type",
    "title",
    "text",
    "provenance",
    "metadata",
}
```

This is a Python **set** containing the names of the fields expected in every chunk record.

A set stores unique elements and supports operations such as membership checks and set differences.

For example, set difference can identify which required fields are missing from a record.

The purpose is to ensure that the retrieval system receives chunks in the expected format rather than silently working with incomplete data.

---

## 4. `ChunkFormatError`

```python
class ChunkFormatError(ValueError):
    pass
```

**Definition:** `ChunkFormatError` is a user-defined exception class that represents an invalid chunk-record format.

### Why is it a class?

In Python, exceptions are represented by classes. By creating a custom exception, we can distinguish chunk-format problems from other errors.

`ValueError` is a built-in Python exception generally used when a value has the correct general type but is inappropriate or invalid for the operation.

By inheriting from `ValueError`, `ChunkFormatError` becomes a specialized form of it.

**Java/Spring analogy:** Similar in concept to defining a custom exception class that extends `RuntimeException` or another exception type in Java. The exact parent class differs, but the purpose is similar: to describe a specific failure meaningfully.

### Why define it if other functions already validate chunks?

Validation and exception definition are two different responsibilities:

- `_validate_chunk()` checks whether a record is valid.
- `ChunkFormatError` provides a specific error type to raise when the record is invalid.
- `load_chunks()` calls the validation logic while loading records.

For example, a chunk missing `chunk_id` can cause `ChunkFormatError` to be raised with a message explaining the problem.

**Important:** Defining the exception class does not perform any checks by itself. The validation code must explicitly raise it when a problem is found. The caller may allow the error to propagate or catch and handle it.

---

## 5. `Embedder` and `Protocol`

Conceptually, the code defines an interface like this:

```python
class Embedder(Protocol):
    def embed(self, text: str) -> Sequence[float]:
        ...
```

### What is `Protocol`?

`Protocol` comes from Python's `typing` module. It allows us to describe an interface that a class is expected to satisfy.

Here, any suitable embedder should provide an `embed()` method that accepts text and returns a sequence of numerical values.

A future implementation could use a local Sentence Transformers model or a hosted embedding service, provided it satisfies this interface.

### Why use `Protocol`?

It separates the retrieval engine from the particular embedding implementation.

For example, `CosineIndex` only needs numerical vectors. It does not need to know whether those vectors were generated by one model or another.

This makes it easier to replace the embedding model without redesigning the cosine-search logic.

**Java analogy:** Similar to defining an interface in Java and providing different classes that implement it.

One difference is that Python protocols generally support *structural typing*: a class can satisfy the protocol by having the required methods, without explicitly inheriting from it.

### What is `Sequence`?

`Sequence` represents an ordered collection of elements, such as a list or tuple.

In `Sequence[float]`, the `float` annotation describes the expected element type.

For example:

```python
[0.6, 0.8]
(0.6, 0.8)
```

Both are sequences of floating-point numbers.

### Is `embed(self, text: str) -> Sequence[float]` a type hint?

Yes. It describes the expected method signature:

- `self`: the current instance.
- `text: str`: the method expects text as a string.
- `-> Sequence[float]`: the method is expected to return an ordered collection of numerical values.

These annotations describe the expected interface; they do not themselves generate embeddings or guarantee runtime type correctness.

The `...` indicates that the protocol specifies the method signature without implementing its actual behavior.

---

## 6. `SearchResult` and `dataclass`

Conceptually, the class looks like this:

```python
@dataclass(frozen=True)
class SearchResult:
    score: float
    record: Mapping[str, object]
```

### What does `@dataclass` do?

The decorator automatically generates methods such as `__init__()` and `__repr__()` based on the declared fields.

Instead of manually writing a constructor to assign `score` and `record`, Python generates one.

The `frozen=True` option makes the dataclass fields read-only after initialization through normal attribute assignment.

**Java analogy:** Similar to using a Java `record` for a small data carrier. The intent is to represent data without manually writing repetitive boilerplate.

### What does `Mapping[str, object]` mean?

Yes, your understanding is basically correct.

- `Mapping` describes a mapping-like object, similar to a dictionary or Java `Map`.
- `str` means the keys are expected to be strings.
- `object` means the values can be objects of various types.

For example, a chunk record might contain:

```python
{
    "chunk_id": "example_chunk_0",
    "title": "Charging FAQ",
    "text": "Check the charger connection.",
    "metadata": {"page": 3}
}
```

The values have different types: strings and a nested dictionary, for example. Therefore, `object` is useful for describing the heterogeneous values in the record.

`Mapping` is a general interface for reading key-value data; it does not require the underlying object to be a mutable `dict`.

### Why use `Mapping` rather than `dict`?

Because the code only needs to access the record's fields, not necessarily modify the record itself. `Mapping` expresses that read-oriented requirement more generally.

One subtle point: `frozen=True` prevents reassignment of `SearchResult` fields, but it does **not** deeply freeze the original record or any dictionary stored inside it.

---

## 7. `_validate_chunk()`

The leading underscore is a Python naming convention indicating that a function is intended for internal use within its module. It is not a security restriction.

The validation function checks things such as:

- Whether the record is a dictionary.
- Whether all required fields are present.
- Whether fields that should contain strings actually contain strings.
- Whether required string values are non-empty, with the appropriate exception for an empty title.
- Whether `provenance` and `metadata` have the expected dictionary structure.

This prevents malformed data from entering the retrieval system and causing confusing errors later.

The function raises `ChunkFormatError` when a chunk violates the expected format.

---

## 8. `load_chunks()`

As the name suggests, this function loads the chunk records.

Its responsibilities include:

1. Locate the chunked-data directory.
2. Find JSONL files in a predictable, sorted order.
3. Read the files line by line.
4. Parse each line as JSON.
5. Validate each record.
6. Collect the valid records into a list and return it.

It also handles loading-related problems, such as a missing directory or malformed JSON.

The distinction is:

- `_validate_chunk()` checks the structure of an individual record.
- `load_chunks()` manages the process of reading records from files and invokes validation.

An existing directory with no JSONL records can produce an empty list, whereas a missing directory raises an error.

---

## 9. `_normalized_vector()`

This function validates a vector and converts it into a unit vector.

A unit vector has a Euclidean norm of 1.

For a vector \(v = [v_1,v_2,\ldots,v_n]\), its Euclidean norm is:

\[
\|v\|_2 = \sqrt{\sum_{i=1}^{n} v_i^2}
\]

The normalized vector is:

\[
\hat v = \frac{v}{\|v\|_2}
\]

For example:

\[
v = [3,4]
\]

Its norm is:

\[
\|v\|_2 = \sqrt{3^2+4^2}=5
\]

After normalization:

\[
\hat v = [0.6,0.8]
\]

### Why normalize vectors?

Cosine similarity compares the angle between two vectors rather than their magnitudes.

Normalizing the vectors makes their lengths equal to 1. The dot product of two normalized vectors is then equal to their cosine similarity.

The function also rejects invalid inputs, such as empty vectors, non-numeric values, non-finite values, and zero vectors, which cannot be normalized.

---

## 10. `CosineIndex`

This is the main class responsible for connecting chunk records to their vectors and retrieving the most similar chunks.

### Initialization: `__init__()`

The constructor performs setup and validation:

- Checks that the number of vectors matches the number of records.
- Validates the chunk records.
- Normalizes the vectors.
- Checks that all vectors have the same dimensionality.
- Stores the validated records and vectors for later searching.

The record-vector relationship is crucial: each vector must remain associated with the correct chunk.

For example, if `records[4]` describes a charging problem, `vectors[4]` must represent that same chunk. Otherwise, a similarity score could be returned with the wrong source text.

The class does not generate embeddings itself. It receives precomputed vectors.

---

## 11. The `search()` method

This is the core mathematical operation that ranks chunks by their similarity to a query vector.

### Step 1: Validate the query

The method validates `top_k`, normalizes the query vector, and checks that its dimensions match those of the stored vectors.

`top_k` controls how many results to return. For example, `top_k=5` requests the five highest-ranked results, or fewer if the index contains fewer records.

### Step 2: Iterate through the stored vectors

```python
enumerate(self._vectors)
```

`enumerate()` iterates through a collection while providing both the index and the element.

For example:

```python
for index, vector in enumerate(vectors):
    ...
```

This gives the current vector and its position. The position is used later to retrieve the corresponding original chunk record.

### Step 3: Pair matching vector components

```python
zip(query, vector)
```

`zip()` pairs elements at matching positions.

For example:

```python
query = [0.6, 0.8]
vector = [0.8, 0.6]
```

The pairs are:

```text
(0.6, 0.8)
(0.8, 0.6)
```

Both vectors must have matching dimensions for this operation to represent a valid comparison. The index checks dimensionality before calculating similarity.

### Step 4: Multiply matching components

```python
query_value * vector_value
```

Each pair of corresponding components is multiplied.

For the example above:

\[
(0.6 \times 0.8) + (0.8 \times 0.6)
= 0.48 + 0.48 = 0.96
\]

### Step 5: Sum the products

```python
math.fsum(...)
```

`math.fsum()` adds the component-wise products with improved floating-point summation accuracy.

For normalized vectors, this sum equals their cosine similarity, apart from possible tiny floating-point rounding errors.

### Step 6: Clamp the score

```python
max(-1.0, min(1.0, score))
```

The theoretical range of cosine similarity is:

\[
-1 \leq \cos(\theta) \leq 1
\]

Due to floating-point rounding, a calculated score can occasionally fall just outside that interval. This expression clamps it to the valid range.

### Step 7: Store the score and index

```python
scored.append((score, index))
```

This stores each similarity score alongside the corresponding vector's index.

The index is important because the program must later associate the score with the correct original chunk record.

### Step 8: Sort and return results

The results are sorted by descending similarity score. If two scores are equal, their original indices provide a deterministic tie-breaker.

The method then selects the top `k` entries and returns `SearchResult` objects containing each score and its corresponding original record.

---

## 12. Interpreting cosine similarity

For normalized vectors, the dot product equals cosine similarity.

| Score | Geometric interpretation |
|---|---|
| Near `1` | Vectors point in similar directions |
| Near `0` | Vectors are approximately orthogonal |
| Near `-1` | Vectors point in opposite directions |

For text embeddings, a higher score generally indicates greater similarity according to the embedding model.

However, the score is **not automatically a probability or a percentage of correctness**. Its practical usefulness depends on the embedding model, the text, and the retrieval task.

A result with the highest score is simply the closest match among the candidates according to the current similarity calculation. It is not guaranteed to be the correct answer.

---

## 13. Overall retrieval flow

```text
Chunked JSONL files
        |
        v
    load_chunks()
        |
        v
  Validate chunk records
        |
        v
  Chunk records + precomputed vectors
        |
        v
     CosineIndex
        |
        v
    search(query_vector)
        |
        v
 Calculate cosine similarity for each chunk
        |
        v
 Rank by score and select top-k
        |
        v
 SearchResult objects
 (score + original chunk record)
```

**Current limitation:** The retrieval engine expects vectors to be supplied to it. It does not yet convert the chunk text or the user's natural-language query into real embeddings. That is the next implementation stage.
