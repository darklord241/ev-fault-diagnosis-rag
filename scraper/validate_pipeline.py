import json
from pathlib import Path

from provenance import (
    PROVENANCE_FIELDS,
    SidecarMetadataError,
    load_sidecar_metadata,
    normalize_provenance,
)


BASE_DIR = Path(__file__).resolve().parent
METADATA_DIR = BASE_DIR / "data" / "metadata"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
CHUNKED_DIR = BASE_DIR / "data" / "chunked"
SUPPORTED_CONTENT_TYPES = {"faq", "pdf"}
REQUIRED_PROVENANCE_VALUES = {
    "document_id",
    "url",
    "source_type",
    "document_type",
    "manufacturer",
    "retrieved_at",
}


def read_jsonl_directory(directory, category, issues):
    rows = []
    if not directory.is_dir():
        issues[category].append(f"Directory does not exist: {directory}")
        return rows

    for file_path in sorted(directory.glob("*.jsonl")):
        try:
            with file_path.open("r", encoding="utf-8") as file:
                for line_number, line in enumerate(file, start=1):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as error:
                        issues[category].append(
                            f"{file_path.name}:{line_number}: invalid JSON: {error}"
                        )
                        continue
                    if not isinstance(record, dict):
                        issues[category].append(
                            f"{file_path.name}:{line_number}: record is not a JSON object"
                        )
                        continue
                    rows.append((record, f"{file_path.name}:{line_number}"))
        except OSError as error:
            issues[category].append(f"Could not read {file_path}: {error}")

    return rows


def validate_provenance(record, location, sidecars, normalized_sidecars, issues):
    document_id = record.get("source_document")
    if not isinstance(document_id, str) or not document_id:
        issues.append(f"{location}: missing source_document for sidecar matching")
        return

    sidecar = sidecars.get(document_id)
    if sidecar is None:
        issues.append(
            f"{location}: no sidecar has document_id {document_id!r}"
        )
        return
    expected = normalized_sidecars.get(document_id)
    if expected is None:
        return

    provenance = record.get("provenance")
    if not isinstance(provenance, dict):
        issues.append(f"{location}: provenance must be an object")
        return

    missing = [field for field in PROVENANCE_FIELDS if field not in provenance]
    if missing:
        issues.append(
            f"{location}: provenance is missing fields {', '.join(missing)}"
        )

    for field in REQUIRED_PROVENANCE_VALUES:
        value = provenance.get(field)
        if not isinstance(value, str) or not value.strip():
            issues.append(
                f"{location}: provenance.{field} must be a non-empty string"
            )

    for field in PROVENANCE_FIELDS:
        if field in provenance and provenance[field] != expected[field]:
            issues.append(
                f"{location}: provenance.{field} does not match sidecar "
                f"{sidecar['_sidecar_path']}"
            )

    if record.get("source_type") != expected["source_type"]:
        issues.append(
            f"{location}: top-level source_type does not match sidecar"
        )


def validate_records(rows, id_field, issues, seen_ids, check_faq_titles=True):
    records_by_id = {}
    for record, location in rows:
        record_id = record.get(id_field)
        if not isinstance(record_id, str) or not record_id:
            issues.append(f"{location}: missing or empty {id_field}")
        else:
            if record_id in seen_ids:
                issues.append(
                    f"{location}: duplicate {id_field} {record_id!r}; "
                    f"first seen at {seen_ids[record_id]}"
                )
            else:
                seen_ids[record_id] = location
            records_by_id.setdefault(record_id, []).append((record, location))

        content_type = record.get("content_type")
        if content_type not in SUPPORTED_CONTENT_TYPES:
            issues.append(
                f"{location}: unsupported content_type {content_type!r}"
            )

        text = record.get("text")
        if not isinstance(text, str) or not text.strip():
            issues.append(f"{location}: text must be a non-empty string")

        if content_type == "faq" and check_faq_titles:
            title = record.get("title")
            if not isinstance(title, str) or not title.strip():
                issues.append(f"{location}: FAQ title must be non-empty")

    return records_by_id


def validate_pipeline():
    issues = {
        "provenance": [],
        "data quality": [],
        "reconciliation": [],
    }

    try:
        sidecars = load_sidecar_metadata(METADATA_DIR)
    except SidecarMetadataError as error:
        sidecars = {}
        issues["provenance"].append(str(error))

    normalized_sidecars = {}
    for document_id, sidecar in sidecars.items():
        try:
            normalized_sidecars[document_id] = normalize_provenance(sidecar)
        except SidecarMetadataError as error:
            issues["provenance"].append(str(error))

    processed = read_jsonl_directory(
        PROCESSED_DIR, "data quality", issues
    )
    chunks = read_jsonl_directory(CHUNKED_DIR, "data quality", issues)

    processed_ids = {}
    chunk_ids = {}
    processed_by_id = validate_records(
        processed, "record_id", issues["data quality"], processed_ids
    )
    validate_records(
        chunks,
        "chunk_id",
        issues["data quality"],
        chunk_ids,
        check_faq_titles=False,
    )

    referenced_documents = set()
    for record, location in processed:
        document_id = record.get("source_document")
        if isinstance(document_id, str) and document_id:
            referenced_documents.add(document_id)
        validate_provenance(
            record,
            location,
            sidecars,
            normalized_sidecars,
            issues["provenance"],
        )

    for document_id in sorted(set(sidecars) - referenced_documents):
        issues["provenance"].append(
            f"Unmatched metadata sidecar document_id {document_id!r}"
        )

    for record, location in chunks:
        validate_provenance(
            record,
            location,
            sidecars,
            normalized_sidecars,
            issues["provenance"],
        )

        record_id = record.get("record_id")
        matches = processed_by_id.get(record_id, [])
        if len(matches) != 1:
            issues["reconciliation"].append(
                f"{location}: record_id {record_id!r} matches "
                f"{len(matches)} processed records"
            )
            continue

        processed_record, processed_location = matches[0]
        if record.get("provenance") != processed_record.get("provenance"):
            issues["provenance"].append(
                f"{location}: provenance differs from processed record "
                f"{processed_location}"
            )

    for record_id, matches in processed_by_id.items():
        chunk_matches = [
            (chunk, location)
            for chunk, location in chunks
            if chunk.get("record_id") == record_id
        ]
        if not chunk_matches:
            issues["reconciliation"].append(
                f"Processed record {record_id!r} has no chunk"
            )

    processed_record_ids = set(processed_by_id)
    for record, location in chunks:
        record_id = record.get("record_id")
        if record_id not in processed_record_ids:
            issues["reconciliation"].append(
                f"{location}: chunk references unknown record_id {record_id!r}"
            )

    processed_files = {path.name for path in PROCESSED_DIR.glob("*.jsonl")}
    chunk_files = {path.name for path in CHUNKED_DIR.glob("*.jsonl")}
    if processed_files != chunk_files:
        missing_chunks = sorted(processed_files - chunk_files)
        orphan_chunks = sorted(chunk_files - processed_files)
        issues["reconciliation"].append(
            f"JSONL file sets differ; missing chunk files={missing_chunks}, "
            f"chunk files without processed files={orphan_chunks}"
        )

    return issues, len(processed), len(chunks)


def main():
    issues, processed_count, chunk_count = validate_pipeline()
    print("Pipeline validation summary")
    print(f"Processed records: {processed_count}")
    print(f"Chunks: {chunk_count}")

    for category, messages in issues.items():
        status = "PASS" if not messages else "FAIL"
        print(f"{category.title()}: {status} ({len(messages)} issue(s))")
        for message in messages[:10]:
            print(f"  - {message}")
        if len(messages) > 10:
            print(f"  - ... and {len(messages) - 10} more")

    return 1 if any(issues.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
