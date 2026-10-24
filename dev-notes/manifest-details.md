# Ingestion Manifest Schema — Version 1

## 1. Purpose

The manifest is the central registry for source documents and their ingestion state in the EV Fault Diagnosis RAG project.

It tracks:
- Every discovered file, including unsupported formats.
- Stable document identity and source-file integrity.
- Metadata, extraction, and chunking status.
- Generated artifact paths and counts.
- Archival state and processing errors.

The manifest is not a replacement for provenance sidecars or extracted content. It tracks lifecycle state and relationships between source documents and their artifacts.

## 2. Top-level structure

```json
{
  "schema_version": 1,
  "documents": {}
}
```

`documents` is a mapping from stable `document_id` to its manifest entry.

## 3. Document entry

```json
{
  "schema_version": 1,
  "documents": {
    "example_document_id": {
      "source": {
        "filename": "example.pdf",
        "path": "incoming/example.pdf",
        "input_type": "pdf",
        "source_url": null,
        "sha256": "verified_sha256_hash",
        "size_bytes": 12345,
        "status": "registered"
      },
      "metadata": {
        "status": "pending",
        "sidecar_path": null,
        "last_error": null
      },
      "extraction": {
        "status": "pending",
        "output_path": null,
        "record_count": null,
        "last_error": null
      },
      "chunking": {
        "status": "pending",
        "output_path": null,
        "chunk_count": null,
        "last_error": null
      },
      "archival": {
        "status": "pending",
        "archive_path": null,
        "last_error": null
      }
    }
  }
}
```

This is a schema example, not an actual document entry. Real paths, checksums, counts, and statuses must be populated from verified files and processing results.

## 4. Field definitions

### 4.1 Source

| Field | Meaning |
|---|---|
| `filename` | Original source filename |
| `path` | Current source path, relative to `data/` |
| `input_type` | Detected format, such as `html`, `pdf`, or `txt` |
| `source_url` | Known source URL, or `null` if unknown |
| `sha256` | SHA-256 checksum of the source file |
| `size_bytes` | Source file size in bytes |
| `status` | Discovery and source classification status |

Supported source statuses:
- `registered`: discovered and registered for ingestion.
- `unsupported_format`: format is not supported by the current pipeline.
- `needs_review`: source classification or identity needs human review.
- `duplicate`: identified as a duplicate of another registered source.
- `failed`: discovery or source validation failed.

Unsupported files must still be represented in the manifest. They should not be silently ignored.

### 4.2 Metadata

| Field | Meaning |
|---|---|
| `status` | Metadata generation and validation state |
| `sidecar_path` | Path to the provenance sidecar |
| `last_error` | Current metadata-stage error, or `null` |

### 4.3 Extraction

| Field | Meaning |
|---|---|
| `status` | Extraction state |
| `output_path` | Path to normalized extracted records |
| `record_count` | Number of validated records, or `null` if unknown |
| `last_error` | Current extraction error, or `null` |

### 4.4 Chunking

| Field | Meaning |
|---|---|
| `status` | Chunking state |
| `output_path` | Path to retrieval-ready chunks |
| `chunk_count` | Number of validated chunks, or `null` if unknown |
| `last_error` | Current chunking error, or `null` |

### 4.5 Archival

| Field | Meaning |
|---|---|
| `status` | Archival state |
| `archive_path` | Archived source path, or `null` |
| `last_error` | Current archival error, or `null` |

Archival must occur only after the required pipeline stages and validations have succeeded. When a source is archived, `source.path` must be updated to its new location as well.

## 5. Shared stage status values

Metadata, extraction, chunking, and archival use the following statuses:

- `pending`: not yet attempted or not yet verified.
- `in_progress`: currently being processed.
- `complete`: processing and required validation succeeded.
- `failed`: an attempted operation failed.
- `needs_review`: human review is required.
- `skipped`: deliberately excluded or not applicable.

The source status `unsupported_format` is separate from these stage statuses. For an unsupported file, applicable downstream stages should remain unprocessed or be marked `skipped`, rather than incorrectly reporting a processing failure.

## 6. Identity and integrity rules

1. Each source document has one stable `document_id`.
2. Existing IDs must be preserved during bootstrap.
3. New IDs may be derived from filenames, provided collisions are detected.
4. SHA-256 checksums identify source content independently of filenames.
5. A changed source checksum invalidates the assumption that existing outputs are current. Affected stages must be revalidated or marked pending.
6. Output paths and counts are populated only when verified.
7. Existing output files must not be assumed to belong to a source solely because their filenames match.

## 7. Manifest ownership

A shared `manifest.py` module owns manifest loading, schema validation, registration, status updates, and persistence.

Pipeline scripts report their own outcomes through this module. They must not independently implement JSON read-modify-write logic.

Discovery and reconciliation are responsible for identifying new files and detecting discrepancies between the manifest and filesystem. Pipeline stages are responsible for reporting their processing results.

Manifest updates should use safe persistence and prevent conflicting updates from silently overwriting one another.

## 8. Bootstrap rules

The initial bootstrap must:
1. Inventory all source files in `incoming/`.
2. Preserve established document IDs and sidecar mappings.
3. Match existing sources to processed and chunked outputs.
4. Validate artifact relationships before marking stages complete.
5. Populate checksums, sizes, and record counts where verified.
6. Leave unprocessed sources pending.
7. Flag ambiguous mappings, unsupported files, missing outputs, and duplicate identities.
8. Preview the proposed manifest before writing it.

The bootstrap must not modify original source files, sidecars, processed records, or chunks.

## 9. Scope

Version 1 tracks discovery, metadata, extraction, chunking, and archival.

Embedding generation, retrieval indexing, and evaluation remain outside this manifest until a later design decision.

Per-stage timestamps are intentionally omitted from version 1. They may be introduced later if auditing or operational debugging demonstrates a need for them.

