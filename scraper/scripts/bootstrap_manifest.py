"""Inventory current data files and preview or write an initial manifest."""

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence


SCRAPER_DIR = Path(__file__).resolve().parents[1]
if str(SCRAPER_DIR) not in sys.path:
    sys.path.insert(0, str(SCRAPER_DIR))

from provenance import (  # noqa: E402
    SidecarMetadataError,
    load_sidecar_metadata,
    normalize_provenance,
)


SCHEMA_VERSION = 1
SOURCE_STATUSES = {
    "registered",
    "unsupported_format",
    "needs_review",
    "duplicate",
    "failed",
}
STAGE_STATUSES = {
    "pending",
    "in_progress",
    "complete",
    "failed",
    "needs_review",
    "skipped",
}
IGNORED_GENERATED_NAMES = {".ds_store"}
IGNORED_GENERATED_SUFFIXES = {".pyc"}
IGNORED_GENERATED_DIRECTORIES = {"__macosx", "__pycache__"}


class BootstrapError(ValueError):
    """Raised for fatal bootstrap input or manifest validation errors."""


@dataclass
class BootstrapReport:
    discovered: list[str] = field(default_factory=list)
    matched_documents: list[str] = field(default_factory=list)
    complete_stages: list[str] = field(default_factory=list)
    pending_stages: list[str] = field(default_factory=list)
    review_stages: list[str] = field(default_factory=list)
    unsupported_documents: list[str] = field(default_factory=list)
    containers: list[str] = field(default_factory=list)
    duplicates: list[str] = field(default_factory=list)
    collisions: list[str] = field(default_factory=list)
    id_overrides: list[str] = field(default_factory=list)
    ambiguous_mappings: list[str] = field(default_factory=list)
    unmatched_sidecars: list[str] = field(default_factory=list)
    orphan_outputs: list[str] = field(default_factory=list)
    missing_artifacts: list[str] = field(default_factory=list)
    stale_sidecar_paths: list[str] = field(default_factory=list)
    ignored_generated: list[str] = field(default_factory=list)
    unscanned_directories: list[str] = field(default_factory=list)
    fatal_errors: list[str] = field(default_factory=list)


@dataclass
class BootstrapResult:
    manifest: dict[str, object]
    report: BootstrapReport
    source_paths: list[str]


def _relative(path: Path, data_dir: Path) -> str:
    return path.resolve().relative_to(data_dir.resolve()).as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _input_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".html":
        return "html"
    if suffix == ".pdf":
        return "pdf"
    if suffix == ".txt":
        return "txt"
    if suffix in {".zip", ".7z", ".rar", ".tar", ".gz"}:
        return "archive"
    return suffix.lstrip(".") or "unknown"


def _is_ignored_generated(path: Path) -> bool:
    return (
        path.name.casefold() in IGNORED_GENERATED_NAMES
        or path.name.startswith("._")
        or path.suffix.casefold() in IGNORED_GENERATED_SUFFIXES
    )


def _discover(data_dir: Path, report: BootstrapReport) -> list[Path]:
    incoming_dir = data_dir / "incoming"
    if not incoming_dir.is_dir():
        raise BootstrapError(f"Incoming directory does not exist: {incoming_dir}")

    files = []
    for entry in sorted(incoming_dir.iterdir(), key=lambda path: path.name.casefold()):
        if entry.is_file():
            candidates = [entry]
        elif entry.is_dir():
            if entry.name.casefold() in IGNORED_GENERATED_DIRECTORIES:
                report.ignored_generated.append(_relative(entry, data_dir))
                continue
            candidates = sorted(
                entry.iterdir(), key=lambda path: path.name.casefold()
            )
            for child in candidates:
                if child.is_dir():
                    if child.name.casefold() in IGNORED_GENERATED_DIRECTORIES:
                        report.ignored_generated.append(_relative(child, data_dir))
                        continue
                    report.unscanned_directories.append(_relative(child, data_dir))
            candidates = [
                child
                for child in candidates
                if child.is_file()
                and child.parent.name.casefold()
                not in IGNORED_GENERATED_DIRECTORIES
            ]
        else:
            continue

        for path in candidates:
            relative = _relative(path, data_dir)
            if _is_ignored_generated(path):
                report.ignored_generated.append(relative)
                continue
            files.append(path)
            report.discovered.append(relative)
            input_type = _input_type(path)
            if input_type == "archive":
                report.containers.append(f"{relative} ({input_type})")
            elif input_type not in {"html", "pdf"}:
                report.unsupported_documents.append(f"{relative} ({input_type})")
    return files


def _candidate_id(path: Path) -> str:
    if _input_type(path) == "html":
        return path.stem.lower().replace(" ", "_")
    return path.stem


def _load_jsonl(path: Path, label: str) -> tuple[list[dict], str | None]:
    try:
        with path.open("r", encoding="utf-8") as file:
            records = []
            for line_number, line in enumerate(file, start=1):
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    return [], (
                        f"{label}:{line_number}: invalid JSON: {error.msg}"
                    )
                if not isinstance(value, dict):
                    return [], f"{label}:{line_number}: expected a JSON object"
                records.append(value)
    except (OSError, UnicodeDecodeError) as error:
        return [], f"{label}: could not read JSONL: {error}"
    return records, None


def _read_json_documents(directory: Path, report: BootstrapReport) -> dict[str, dict]:
    try:
        sidecars = load_sidecar_metadata(directory)
    except SidecarMetadataError as error:
        report.fatal_errors.append(str(error))
        return {}
    return sidecars


def _sidecar_candidates(
    document_id: str,
    sidecar: Mapping[str, object],
    sources: Sequence[Path],
) -> list[Path]:
    exact_original = sidecar.get("original_filename")
    file_path = sidecar.get("file_path")
    file_basename = Path(file_path).name if isinstance(file_path, str) else None
    content_type = sidecar.get("content_type")
    matches = []
    for source in sources:
        kind = _input_type(source)
        if exact_original == source.name:
            matches.append(source)
            continue
        if file_basename == source.name and (
            (content_type == "application/pdf" and kind == "pdf")
            or (content_type == "text/html" and kind == "html")
        ):
            matches.append(source)
            continue
        if kind == "pdf" and source.stem == document_id:
            matches.append(source)
            continue
        if kind == "html" and _candidate_id(source) == document_id:
            matches.append(source)
    return list(dict.fromkeys(matches))


def _validate_records(
    document_id: str,
    source_type: str,
    expected_content_type: str,
    expected_provenance: Mapping[str, object] | None,
    records: Sequence[Mapping[str, object]],
    label: str,
) -> tuple[bool, str | None]:
    seen_record_ids = set()
    for index, record in enumerate(records, start=1):
        location = f"{label}:{index}"
        record_id = record.get("record_id")
        if not isinstance(record_id, str) or not record_id.startswith(
            f"{document_id}_"
        ):
            return False, f"{location}: record_id does not match {document_id!r}"
        if record_id in seen_record_ids:
            return False, f"{location}: duplicate record_id {record_id!r}"
        seen_record_ids.add(record_id)
        if record.get("source_document") != document_id:
            return False, f"{location}: source_document does not match document_id"
        if record.get("content_type") != expected_content_type:
            return False, f"{location}: content_type does not match source format"
        if not isinstance(record.get("text"), str) or not record["text"].strip():
            return False, f"{location}: text is missing or empty"
        if not isinstance(record.get("title"), str):
            return False, f"{location}: title must be a string"
        if record.get("source_type") != source_type:
            return False, f"{location}: source_type does not match sidecar"
        provenance = record.get("provenance")
        if not isinstance(provenance, dict):
            return False, f"{location}: provenance must be an object"
        if provenance.get("document_id") != document_id:
            return False, f"{location}: provenance document_id does not match"
        if expected_provenance is not None and provenance != expected_provenance:
            return False, f"{location}: provenance does not match sidecar"
        if not isinstance(record.get("metadata"), dict):
            return False, f"{location}: metadata must be an object"
    return True, None


def _validate_chunk_records(
    document_id: str,
    source_type: str,
    content_type: str,
    provenance: Mapping[str, object] | None,
    records: Sequence[Mapping[str, object]],
    processed_records: Sequence[Mapping[str, object]],
    label: str,
) -> tuple[bool, str | None]:
    source_by_record = {record["record_id"]: record for record in processed_records}
    seen_chunk_ids = set()
    chunk_texts: dict[str, list[str]] = defaultdict(list)
    for index, chunk in enumerate(records, start=1):
        location = f"{label}:{index}"
        record_id = chunk.get("record_id")
        chunk_id = chunk.get("chunk_id")
        if not isinstance(record_id, str):
            return False, f"{location}: record_id must be a string"
        if record_id not in source_by_record:
            return False, f"{location}: record_id is absent from processed output"
        if not isinstance(chunk_id, str) or not chunk_id.startswith(
            f"{record_id}_chunk_"
        ):
            return False, f"{location}: chunk_id does not match record_id"
        match = re.fullmatch(re.escape(f"{record_id}_chunk_") + r"(\d+)", chunk_id)
        if match is None:
            return False, f"{location}: chunk_id has an invalid chunk index"
        if chunk_id in seen_chunk_ids:
            return False, f"{location}: duplicate chunk_id {chunk_id!r}"
        seen_chunk_ids.add(chunk_id)
        source_record = source_by_record[record_id]
        for field_name, expected in (
            ("source_document", document_id),
            ("source_type", source_type),
            ("content_type", content_type),
            ("title", source_record["title"]),
            ("provenance", provenance or source_record["provenance"]),
            ("metadata", source_record["metadata"]),
        ):
            if chunk.get(field_name) != expected:
                return False, f"{location}: {field_name} does not match processed output"
        text = chunk.get("text")
        if not isinstance(text, str) or not text.strip():
            return False, f"{location}: text is missing or empty"
        chunk_texts[record_id].append(text)

    if len(chunk_texts) != len(source_by_record):
        return False, "chunk output is missing chunks for one or more processed records"
    for record_id, source_record in source_by_record.items():
        texts = chunk_texts[record_id]
        if content_type == "faq":
            if texts != [source_record["text"]]:
                return False, f"FAQ chunks do not preserve text for {record_id!r}"
        else:
            source_words = source_record["text"].split()
            start = 0
            for index, text in enumerate(texts):
                end = min(start + 600, len(source_words))
                words = text.split()
                if source_words[start:end] != words:
                    return False, f"PDF chunk text is not contiguous for {record_id!r}"
                if end == len(source_words):
                    if index != len(texts) - 1:
                        return False, f"unexpected extra PDF chunk for {record_id!r}"
                    break
                start = end - 100
            else:
                return False, f"PDF chunks do not cover all text for {record_id!r}"
    return True, None


def _stage(status: str, **values: object) -> dict[str, object]:
    return {"status": status, **values}


def _entry_for_source(
    source_path: Path,
    document_id: str,
    data_dir: Path,
    sidecar: Mapping[str, object] | None,
    report: BootstrapReport,
) -> dict[str, object]:
    relative = _relative(source_path, data_dir)
    input_type = _input_type(source_path)
    digest = _sha256(source_path)
    size = source_path.stat().st_size
    is_container = input_type == "archive"
    supported = input_type in {"html", "pdf"}
    source_status = "registered" if supported else "unsupported_format"
    expected_content_type = "faq" if input_type == "html" else "pdf"

    source_url = None
    provenance = None
    sidecar_path = None
    metadata_status = "pending" if supported else "skipped"
    metadata_error = None
    if sidecar is not None and supported:
        sidecar_path = f"metadata/{Path(str(sidecar['_sidecar_path'])).name}"
        try:
            provenance = normalize_provenance(sidecar)
            source_url = provenance["url"]
            expected_sidecar_type = (
                "text/html" if input_type == "html" else "application/pdf"
            )
            if sidecar.get("content_type") not in (None, expected_sidecar_type):
                raise SidecarMetadataError("sidecar content_type is unsupported")
            metadata_status = "complete"
        except SidecarMetadataError as error:
            metadata_status = "needs_review"
            metadata_error = str(error)
            provenance = None
            source_url = None
            report.ambiguous_mappings.append(
                f"{relative}: sidecar provenance is invalid: {error}"
            )

        saved_path = sidecar.get("file_path")
        if isinstance(saved_path, str):
            saved_candidate = data_dir / saved_path
            if not saved_candidate.exists():
                report.stale_sidecar_paths.append(
                    f"{relative}: sidecar file_path {saved_path!r} does not exist"
                )
    elif supported:
        report.missing_artifacts.append(
            f"{relative}: metadata/{document_id}.json"
        )

    if not supported:
        extraction = _stage("skipped", output_path=None, record_count=None, last_error=None)
        chunking = _stage("skipped", output_path=None, chunk_count=None, last_error=None)
    else:
        processed_path = data_dir / "processed" / f"{document_id}.jsonl"
        chunked_path = data_dir / "chunked" / f"{document_id}.jsonl"
        if not processed_path.is_file():
            extraction = _stage(
                "pending", output_path=None, record_count=None, last_error=None
            )
            report.missing_artifacts.append(
                f"{relative}: processed/{document_id}.jsonl"
            )
            processed_records = []
        else:
            processed_records, error = _load_jsonl(
                processed_path, f"processed/{processed_path.name}"
            )
            if error is None:
                if sidecar is None or provenance is None:
                    error = "no valid provenance sidecar is available to verify records"
                else:
                    valid, error = _validate_records(
                        document_id,
                        provenance["source_type"],
                        expected_content_type,
                        provenance,
                        processed_records,
                        f"processed/{processed_path.name}",
                    )
            if error is None:
                extraction = _stage(
                    "complete",
                    output_path=f"processed/{document_id}.jsonl",
                    record_count=len(processed_records),
                    last_error=None,
                )
            else:
                extraction = _stage(
                    "needs_review",
                    output_path=None,
                    record_count=None,
                    last_error=error,
                )
                report.ambiguous_mappings.append(f"{relative}: {error}")
                processed_records = []

        if not chunked_path.is_file():
            chunking = _stage(
                "pending", output_path=None, chunk_count=None, last_error=None
            )
            report.missing_artifacts.append(f"{relative}: chunked/{document_id}.jsonl")
        else:
            chunk_records, error = _load_jsonl(
                chunked_path, f"chunked/{chunked_path.name}"
            )
            if error is None:
                if extraction["status"] != "complete":
                    error = "processed records are not verified"
                elif sidecar is None or provenance is None:
                    error = "no valid provenance sidecar is available to verify chunks"
                else:
                    valid, error = _validate_chunk_records(
                        document_id,
                        provenance["source_type"],
                        expected_content_type,
                        provenance,
                        chunk_records,
                        processed_records,
                        f"chunked/{chunked_path.name}",
                    )
            if error is None:
                chunking = _stage(
                    "complete",
                    output_path=f"chunked/{document_id}.jsonl",
                    chunk_count=len(chunk_records),
                    last_error=None,
                )
            else:
                chunking = _stage(
                    "needs_review",
                    output_path=None,
                    chunk_count=None,
                    last_error=error,
                )
                report.ambiguous_mappings.append(f"{relative}: {error}")

    if is_container:
        source_status = "unsupported_format"
        metadata_status = "skipped"
        metadata_error = None
        metadata_stage = _stage(
            metadata_status, sidecar_path=None, last_error=metadata_error
        )
        extraction = _stage(
            "skipped", output_path=None, record_count=None, last_error=None
        )
        chunking = _stage(
            "skipped", output_path=None, chunk_count=None, last_error=None
        )
    else:
        metadata_stage = _stage(
            metadata_status,
            sidecar_path=sidecar_path,
            last_error=metadata_error,
        )

    return {
        "source": {
            "filename": source_path.name,
            "path": relative,
            "input_type": input_type,
            "source_url": source_url,
            "sha256": digest,
            "size_bytes": size,
            "status": source_status,
        },
        "metadata": metadata_stage,
        "extraction": extraction,
        "chunking": chunking,
        "archival": _stage(
            "pending", archive_path=None, last_error=None
        ),
    }


def _read_id_overrides(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BootstrapError(f"Could not read ID override file {path}: {error}") from error
    if not isinstance(payload, dict) or not all(
        isinstance(key, str)
        and isinstance(value, str)
        and value.strip()
        for key, value in payload.items()
    ):
        raise BootstrapError(
            "ID override file must map exact incoming filenames to non-empty IDs"
        )
    return {key: value.strip() for key, value in payload.items()}


def build_manifest(
    data_dir: Path,
    id_overrides: Mapping[str, str] | None = None,
) -> BootstrapResult:
    data_dir = Path(data_dir).resolve()
    report = BootstrapReport()
    sources = _discover(data_dir, report)
    overrides = dict(id_overrides or {})
    by_filename = {path.name: path for path in sources}
    for filename in overrides:
        if filename not in by_filename:
            report.fatal_errors.append(
                f"ID override names an undiscovered file: {filename}"
            )

    sidecars = _read_json_documents(data_dir / "metadata", report)
    source_to_sidecar: dict[Path, tuple[str, dict]] = {}
    sidecar_to_source: dict[str, Path] = {}
    for document_id, sidecar in sidecars.items():
        candidates = _sidecar_candidates(document_id, sidecar, sources)
        if len(candidates) > 1:
            report.ambiguous_mappings.append(
                f"Sidecar {document_id!r} matches multiple sources: "
                + ", ".join(_relative(path, data_dir) for path in candidates)
            )
            report.unmatched_sidecars.append(document_id)
            continue
        if not candidates:
            report.unmatched_sidecars.append(document_id)
            continue
        source_path = candidates[0]
        if source_path in source_to_sidecar:
            report.ambiguous_mappings.append(
                f"Multiple sidecars match {_relative(source_path, data_dir)}"
            )
            report.unmatched_sidecars.append(document_id)
            continue
        source_to_sidecar[source_path] = (document_id, sidecar)
        sidecar_to_source[document_id] = source_path
        report.matched_documents.append(
            f"{document_id} <- {_relative(source_path, data_dir)}"
        )

    candidate_ids: dict[Path, str] = {}
    for source_path in sources:
        mapped = source_to_sidecar.get(source_path)
        candidate_ids[source_path] = (
            mapped[0]
            if mapped is not None
            else overrides.get(source_path.name, _candidate_id(source_path))
        )
        if mapped is None and source_path.name in overrides:
            report.id_overrides.append(
                f"{source_path.name}: {overrides[source_path.name]}"
            )
        if mapped is not None and source_path.name in overrides:
            report.fatal_errors.append(
                f"Cannot override established document_id for {source_path.name}"
            )

    ids_by_folded: dict[str, list[Path]] = defaultdict(list)
    for source_path, document_id in candidate_ids.items():
        ids_by_folded[document_id.casefold()].append(source_path)
    for folded_group in ids_by_folded.values():
        if len(folded_group) < 2:
            continue
        group_ids = {candidate_ids[path] for path in folded_group}
        label = (
            "Document ID collision"
            if len(group_ids) == 1
            else "Case-insensitive document ID collision"
        )
        paths = ", ".join(
            _relative(path, data_dir) for path in folded_group
        )
        established = [
            f"{_relative(path, data_dir)} -> {source_to_sidecar[path][0]!r}"
            for path in folded_group
            if path in source_to_sidecar
        ]
        detail = (
            f"; established sidecar mapping: {', '.join(established)}"
            if established
            else ""
        )
        report.collisions.append(
            f"{label} for {', '.join(sorted(group_ids))!r}: {paths}{detail}"
        )

    duplicate_hashes: dict[str, list[Path]] = defaultdict(list)
    for source_path in sources:
        duplicate_hashes[_sha256(source_path)].append(source_path)
    duplicate_source_paths = set()
    for group in duplicate_hashes.values():
        if len(group) > 1:
            for path in group[1:]:
                duplicate_source_paths.add(path)
            label = " = ".join(_relative(path, data_dir) for path in group)
            report.duplicates.append(label)

    documents: dict[str, object] = {}
    expected_sources = {_relative(path, data_dir) for path in sources}
    represented_sources = set()
    for source_path in sources:
        document_id = candidate_ids[source_path]
        folded_group = ids_by_folded[document_id.casefold()]
        exact_group = [
            path for path in folded_group
            if candidate_ids[path] == document_id
        ]
        if len(exact_group) > 1:
            mapped_members = [
                path for path in exact_group if path in source_to_sidecar
            ]
            if source_path not in mapped_members and mapped_members:
                continue

        sidecar_info = source_to_sidecar.get(source_path)
        sidecar = sidecar_info[1] if sidecar_info else None
        entry = _entry_for_source(
            source_path, document_id, data_dir, sidecar, report
        )
        if source_path in duplicate_source_paths and entry["source"]["status"] != (
            "unsupported_format"
        ):
            entry["source"]["status"] = "duplicate"
        if len(folded_group) > 1 and _input_type(source_path) in {"html", "pdf"}:
            entry["source"]["status"] = "needs_review"

        if document_id in documents:
            report.collisions.append(
                f"Duplicate manifest key {document_id!r}; "
                f"{_relative(source_path, data_dir)} was not represented"
            )
            continue
        documents[document_id] = entry
        represented_sources.add(_relative(source_path, data_dir))

        for stage_name in ("metadata", "extraction", "chunking", "archival"):
            status = entry[stage_name]["status"]
            label = f"{document_id}: {stage_name}={status}"
            if status == "complete":
                report.complete_stages.append(label)
            elif status == "pending":
                report.pending_stages.append(label)
            elif status == "needs_review":
                report.review_stages.append(label)

    for path in sorted((data_dir / "processed").glob("*.jsonl")):
        if path.stem not in documents:
            report.orphan_outputs.append(f"processed/{path.name}")
    for path in sorted((data_dir / "chunked").glob("*.jsonl")):
        if path.stem not in documents:
            report.orphan_outputs.append(f"chunked/{path.name}")

    missing_sources = expected_sources - represented_sources
    for relative in sorted(missing_sources):
        report.fatal_errors.append(
            f"Source cannot be uniquely represented in documents: {relative}"
        )

    manifest: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "documents": documents,
    }
    validation_errors = validate_manifest(manifest, expected_sources)
    report.fatal_errors.extend(validation_errors)
    return BootstrapResult(
        manifest=manifest,
        report=report,
        source_paths=sorted(expected_sources),
    )


def validate_manifest(
    manifest: Mapping[str, object],
    expected_source_paths: set[str] | None = None,
) -> list[str]:
    errors = []
    if set(manifest) != {"schema_version", "documents"}:
        errors.append("top-level fields do not match the manifest schema")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        errors.append("schema_version must equal 1")
    documents = manifest.get("documents")
    if not isinstance(documents, dict):
        return errors + ["documents must be an object"]

    represented_paths = set()
    source_fields = {
        "filename",
        "path",
        "input_type",
        "source_url",
        "sha256",
        "size_bytes",
        "status",
    }
    stage_fields = {
        "metadata": {"status", "sidecar_path", "last_error"},
        "extraction": {"status", "output_path", "record_count", "last_error"},
        "chunking": {"status", "output_path", "chunk_count", "last_error"},
        "archival": {"status", "archive_path", "last_error"},
    }
    for document_id, entry in documents.items():
        if not isinstance(document_id, str) or not document_id.strip():
            errors.append("document IDs must be non-empty strings")
            continue
        if not isinstance(entry, dict) or set(entry) != {
            "source",
            "metadata",
            "extraction",
            "chunking",
            "archival",
        }:
            errors.append(f"{document_id}: entry fields do not match schema")
            continue
        source = entry["source"]
        if not isinstance(source, dict) or set(source) != source_fields:
            errors.append(f"{document_id}: source fields do not match schema")
            continue
        if source.get("status") not in SOURCE_STATUSES:
            errors.append(f"{document_id}: invalid source status")
        if not isinstance(source.get("filename"), str):
            errors.append(f"{document_id}: source filename must be a string")
        path = source.get("path")
        if not isinstance(path, str) or path.startswith("/") or ".." in Path(path).parts:
            errors.append(f"{document_id}: source path must be relative to data/")
        else:
            represented_paths.add(path)
        if not isinstance(source.get("input_type"), str):
            errors.append(f"{document_id}: input_type must be a string")
        source_url = source.get("source_url")
        if source_url is not None and not isinstance(source_url, str):
            errors.append(f"{document_id}: source_url must be a string or null")
        digest = source.get("sha256")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            errors.append(f"{document_id}: sha256 must be a lowercase SHA-256 digest")
        size = source.get("size_bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            errors.append(f"{document_id}: size_bytes must be a non-negative integer")

        for stage_name, expected_fields in stage_fields.items():
            stage = entry[stage_name]
            if not isinstance(stage, dict) or set(stage) != expected_fields:
                errors.append(f"{document_id}: {stage_name} fields do not match schema")
                continue
            if stage.get("status") not in STAGE_STATUSES:
                errors.append(f"{document_id}: invalid {stage_name} status")
            if stage.get("last_error") is not None and not isinstance(
                stage["last_error"], str
            ):
                errors.append(f"{document_id}: {stage_name}.last_error must be string or null")
            path_key = {
                "metadata": "sidecar_path",
                "extraction": "output_path",
                "chunking": "output_path",
                "archival": "archive_path",
            }[stage_name]
            stage_path = stage.get(path_key)
            if stage_path is not None and not isinstance(stage_path, str):
                errors.append(
                    f"{document_id}: {stage_name}.{path_key} must be string or null"
                )
            if isinstance(stage_path, str) and not stage_path.strip():
                errors.append(f"{document_id}: {stage_name}.{path_key} must not be empty")
            count_key = "record_count" if stage_name == "extraction" else "chunk_count"
            if count_key in stage:
                count = stage[count_key]
                if count is not None and (
                    isinstance(count, bool) or not isinstance(count, int) or count < 0
                ):
                    errors.append(f"{document_id}: {stage_name}.{count_key} is invalid")
            if stage.get("status") == "complete" and stage_path is None:
                errors.append(f"{document_id}: complete {stage_name} requires {path_key}")
            if (
                stage_name in {"extraction", "chunking"}
                and stage.get("status") == "complete"
                and stage.get(count_key) is None
            ):
                errors.append(
                    f"{document_id}: complete {stage_name} requires {count_key}"
                )
            if stage.get("status") != "complete" and stage_name in {
                "extraction",
                "chunking",
            } and stage.get(count_key) is not None:
                errors.append(
                    f"{document_id}: non-complete {stage_name} must not declare a count"
                )
        if source.get("status") == "unsupported_format":
            for stage_name in ("metadata", "extraction", "chunking"):
                if entry[stage_name].get("status") != "skipped":
                    errors.append(
                        f"{document_id}: unsupported source must skip {stage_name}"
                    )

    if expected_source_paths is not None and represented_paths != expected_source_paths:
        missing = sorted(expected_source_paths - represented_paths)
        extra = sorted(represented_paths - expected_source_paths)
        if missing:
            errors.append(f"manifest does not represent sources: {', '.join(missing)}")
        if extra:
            errors.append(f"manifest contains unknown source paths: {', '.join(extra)}")
    ids = [document_id.casefold() for document_id in documents]
    if len(ids) != len(set(ids)):
        errors.append("document IDs are not unique case-insensitively")
    return errors


def write_manifest_atomic(path: Path, manifest: Mapping[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary_path = Path(file.name)
            json.dump(manifest, file, ensure_ascii=False, indent=2)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)
    except OSError:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def format_report(result: BootstrapResult) -> str:
    report = result.report
    lines = [
        "Ingestion manifest bootstrap preview",
        f"Discovered files: {len(report.discovered)}",
        f"Matched existing documents: {len(report.matched_documents)}",
        f"Complete stages: {len(report.complete_stages)}",
        f"Pending stages: {len(report.pending_stages)}",
        f"Stages needing review: {len(report.review_stages)}",
        f"Unsupported document formats: {len(report.unsupported_documents)}",
        f"Collection containers: {len(report.containers)}",
        f"Duplicate-content groups: {len(report.duplicates)}",
        f"ID collisions: {len(report.collisions)}",
        f"Ambiguous mappings: {len(report.ambiguous_mappings)}",
        f"Unmatched sidecars: {len(report.unmatched_sidecars)}",
        f"Orphan outputs: {len(report.orphan_outputs)}",
        f"Missing artifacts: {len(report.missing_artifacts)}",
        f"Stale sidecar paths: {len(report.stale_sidecar_paths)}",
        f"Fatal validation errors: {len(report.fatal_errors)}",
        "",
        "Discovered files:",
    ]
    lines.extend(f"- {item}" for item in report.discovered)
    for title, items in (
        ("Matched existing documents", report.matched_documents),
        ("Complete stages", report.complete_stages),
        ("Pending stages", report.pending_stages),
        ("Stages needing review", report.review_stages),
        ("Unsupported document formats", report.unsupported_documents),
        ("Collection containers (not extracted)", report.containers),
        ("Duplicate content", report.duplicates),
        ("ID collisions", report.collisions),
        ("Explicit ID overrides", report.id_overrides),
        ("Ambiguous mappings", report.ambiguous_mappings),
        ("Unmatched sidecars", report.unmatched_sidecars),
        ("Orphan outputs", report.orphan_outputs),
        ("Missing artifacts", report.missing_artifacts),
        ("Stale sidecar paths", report.stale_sidecar_paths),
        ("Ignored generated files", report.ignored_generated),
        ("Directories not scanned below the one-level limit", report.unscanned_directories),
        ("Fatal validation errors", report.fatal_errors),
    ):
        if items:
            lines.extend(("", f"{title}:"))
            lines.extend(f"- {item}" for item in items)
    lines.extend(("", "Proposed manifest:"))
    lines.append(json.dumps(result.manifest, ensure_ascii=False, indent=2))
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Preview or bootstrap scraper/data/manifest.json."
    )
    parser.add_argument(
        "--data-dir", type=Path, default=SCRAPER_DIR / "data"
    )
    parser.add_argument(
        "--id-overrides",
        type=Path,
        help="JSON mapping exact incoming filenames to explicit document IDs",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="write the validated manifest; default is preview only",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="allow replacing a nonempty manifest (requires --write)",
    )
    args = parser.parse_args(argv)
    if args.overwrite and not args.write:
        parser.error("--overwrite requires --write")

    manifest_path = args.data_dir / "manifest.json"
    if args.write and manifest_path.exists() and manifest_path.stat().st_size > 0:
        if not args.overwrite:
            parser.error(
                f"Refusing to overwrite nonempty manifest {manifest_path}; "
                "pass --overwrite explicitly"
            )

    try:
        overrides = _read_id_overrides(args.id_overrides)
        result = build_manifest(args.data_dir, overrides)
    except (BootstrapError, OSError, SidecarMetadataError) as error:
        parser.error(str(error))

    sys.stdout.write(format_report(result))
    if result.report.fatal_errors:
        return 1
    if args.write:
        try:
            write_manifest_atomic(manifest_path, result.manifest)
        except OSError as error:
            sys.stderr.write(f"Could not write manifest atomically: {error}\n")
            return 1
        sys.stdout.write(f"\nWrote manifest atomically to {manifest_path}\n")
    else:
        sys.stdout.write("\nPreview only; no files were written.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
