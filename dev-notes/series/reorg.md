### Retrieval Module Reorganization

- Grouped the retrieval-related modules under `scraper/vector_search/` to keep the codebase organized and separate vector-search functionality from the rest of the scraper.
- Added `__init__.py` to make `vector_search` an explicit Python package.
- Updated imports to use package-relative imports within `vector_search` and package-qualified imports in tests.
- Updated filesystem paths for the chunked corpus and embedding cache to account for the new module locations.
- Moved `retrieval_eval.json` into `vector_search/` alongside the retrieval implementation.
- Kept the existing corpus chunks, staged PDFs, and unrelated development notes unchanged.
- one more doubt was that why .embedding for sibling imports and not the usual ./embedding and that is cuz python's syntax for a relative module import.
- 

**Validation**
- Confirmed the full test suite passes: 63 tests.
- Verified corpus chunk and staged PDF SHA-256 checksums remained unchanged.
- Confirmed the updated paths resolve correctly.
- Excluded test-script changes from this commit.

**Scope:** Structural reorganization only; no changes to the retrieval algorithm or corpus data.