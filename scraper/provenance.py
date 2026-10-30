import json
from pathlib import Path


PROVENANCE_FIELDS = (
    "document_id",
    "url",
    "source_type",
    "document_type",
    "manufacturer",
    "vehicle_model",
    "retrieved_at",
)


class SidecarMetadataError(ValueError):
    pass


def load_sidecar_metadata(metadata_dir: Path):
    if not metadata_dir.is_dir():
        raise SidecarMetadataError(
            f"Metadata directory does not exist: {metadata_dir}"
        )

    sidecars = {}
    for metadata_path in sorted(metadata_dir.glob("*.json")):
        try:
            with metadata_path.open("r", encoding="utf-8") as file:
                metadata = json.load(file)
        except (OSError, json.JSONDecodeError) as error:
            raise SidecarMetadataError(
                f"Could not read sidecar {metadata_path}: {error}"
            ) from error

        if not isinstance(metadata, dict):
            raise SidecarMetadataError(
                f"Sidecar must contain a JSON object: {metadata_path}"
            )

        document_id = metadata.get("document_id")
        if not isinstance(document_id, str) or not document_id.strip():
            raise SidecarMetadataError(
                f"Sidecar has no non-empty document_id: {metadata_path}"
            )

        if document_id in sidecars:
            previous_path = sidecars[document_id]["_sidecar_path"]
            raise SidecarMetadataError(
                f"Duplicate sidecar document_id {document_id!r}: "
                f"{previous_path} and {metadata_path}"
            )

        sidecars[document_id] = {
            **metadata,
            "_sidecar_path": str(metadata_path),
        }

    return sidecars


def normalize_provenance(metadata):
    final_url = metadata.get("final_url")
    source_url = metadata.get("url")
    url = (
        final_url
        if isinstance(final_url, str) and final_url.strip()
        else source_url
    )

    required_values = {
        "document_id": metadata.get("document_id"),
        "url": url,
        "source_type": metadata.get("source_type"),
        "document_type": metadata.get("document_type"),
        "manufacturer": metadata.get("manufacturer"),
        "retrieved_at": metadata.get("retrieved_at"),
    }
    for field, value in required_values.items():
        if not isinstance(value, str) or not value.strip():
            raise SidecarMetadataError(
                f"Sidecar {metadata.get('document_id', '<unknown>')!r} "
                f"requires {field} to be a non-empty string"
            )

    vehicle_model = metadata.get("vehicle_model")
    if vehicle_model is not None and not isinstance(vehicle_model, str):
        raise SidecarMetadataError(
            f"Sidecar {metadata.get('document_id', '<unknown>')!r} "
            "requires vehicle_model to be a string or null"
        )

    return {
        "document_id": required_values["document_id"],
        "url": url,
        "source_type": required_values["source_type"],
        "document_type": required_values["document_type"],
        "manufacturer": required_values["manufacturer"],
        "vehicle_model": vehicle_model,
        "retrieved_at": required_values["retrieved_at"],
    }


def get_provenance(sidecars, document_id):
    metadata = sidecars.get(document_id)
    if metadata is None:
        raise SidecarMetadataError(
            f"No metadata sidecar matches document_id {document_id!r}"
        )

    return normalize_provenance(metadata)
