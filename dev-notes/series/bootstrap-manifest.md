# Understanding the Initial Ingestion Manifest

## 1. Purpose of the Manifest

The ingestion manifest is a central record of the documents handled by the EV Fault Diagnosis RAG project's ingestion pipeline.

It tracks:
- Which source documents exist in the dataset.
- Where each source document is stored.
- Whether metadata generation, extraction, and chunking have completed.
- The output artifacts associated with each document.
- Errors, unsupported formats, and documents requiring review.

The manifest represents the **current state of the pipeline**, not a history of every operation performed.

### Design decisions

- The manifest is stored at `scraper/data/manifest.json`.
- Document IDs remain stable across pipeline stages.
- Source paths are relative to `scraper/data/`.
- Detailed provenance remains in the existing metadata sidecars.
- The filesystem remains the source of truth for whether files actually exist.
- Embeddings, retrieval indexes, and evaluation results are outside the manifest's scope.
- Version 1 does not include timestamps for individual stages.

## 2. Manifest Structure

The manifest uses a versioned top-level structure:

```json
{
  "schema_version": 1,
  "documents": {
    "document_id": {
      "source": {},
      "metadata": {},
      "extraction": {},
      "chunking": {},
      "archival": {}
    }
  }
}
```

Each document is identified by a stable `document_id`.

### Source

The `source` section describes the original input.

| Field | Purpose |
|---|---|
| `filename` | Original filename |
| `path` | Source path relative to `data/` |
| `input_type` | Type of input, such as HTML, PDF, TXT, or archive |
| `source_url` | Original source URL, if known |
| `sha256` | Checksum used to identify source content |
| `size_bytes` | Source file size |
| `status` | Registration or support status of the source |

### Metadata

The `metadata` section tracks metadata generation.

| Field | Purpose |
|---|---|
| `status` | Current metadata stage status |
| `sidecar_path` | Path to the associated metadata sidecar |
| `last_error` | Most recent error, if any |

### Extraction

The `extraction` section tracks conversion of source content into processed records.

| Field | Purpose |
|---|---|
| `status` | Current extraction stage status |
| `output_path` | Path to the processed JSONL output |
| `record_count` | Number of extracted records |
| `last_error` | Most recent error, if any |

### Chunking

The `chunking` section tracks the creation of chunks for retrieval.

| Field | Purpose |
|---|---|
| `status` | Current chunking stage status |
| `output_path` | Path to the chunked JSONL output |
| `chunk_count` | Number of chunks produced |
| `last_error` | Most recent error, if any |

### Archival

The `archival` section tracks archival state.

| Field | Purpose |
|---|---|
| `status` | Current archival stage status |
| `archive_path` | Archive location, if applicable |
| `last_error` | Most recent error, if any |

## 3. Stage Statuses

The pipeline uses the following stage statuses:

- `pending` — the stage has not completed.
- `in_progress` — the stage is currently running.
- `complete` — the stage completed successfully.
- `failed` — the stage failed.
- `needs_review` — the result requires manual inspection.
- `skipped` — the stage does not apply or was intentionally skipped.

Source-level statuses additionally include:

- `registered`
- `unsupported_format`
- `needs_review`
- `duplicate`
- `failed`

These statuses distinguish an unprocessed document from one that the pipeline cannot currently process.

## 4. Existing Processed Documents

The initial manifest reconciles the source documents with the existing metadata sidecars, processed records, and chunks.

Eight existing document IDs were matched to their existing artifacts:

| Document ID | Source type | Processed records | Chunks |
|---|---|---:|---:|
| `ather_450_series_faqs___450x_vs_450s,_charging,_warranty` | HTML | 22 | 22 |
| `ather_charging_faqs` | HTML | 16 | 16 |
| `ather_ecw_warranty_faqs` | HTML | 1 | 1 |
| `ather_electric_scooter_faqs___support_&_help_center` | HTML | 335 | 335 |
| `ather_roadside_assistance_faqs` | HTML | 12 | 12 |
| `ather_subscription_terms_and_conditions` | PDF | 10 | 10 |
| `ather_warranty_policy_2024` | PDF | 6 | 6 |
| `battery_warranty_details_for_ather_scooters___faqs` | HTML | 5 | 5 |

**Existing artifact totals:**
- 8 matched documents.
- 407 processed records.
- 407 chunks.

The initial reconciliation preserved these existing artifacts rather than regenerating them.

## 5. Handling
