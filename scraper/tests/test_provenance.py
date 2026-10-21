import json
import tempfile
import unittest
from pathlib import Path

from chunker import create_chunks
from provenance import (
    SidecarMetadataError,
    get_provenance,
    load_sidecar_metadata,
    normalize_provenance,
)


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.metadata_dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def write_sidecar(self, filename, metadata):
        path = self.metadata_dir / filename
        path.write_text(json.dumps(metadata), encoding="utf-8")
        return path

    @staticmethod
    def valid_metadata(**updates):
        metadata = {
            "document_id": "doc-1",
            "url": "https://example.test/source",
            "source_type": "official",
            "document_type": "faq",
            "manufacturer": "Ather Energy",
            "vehicle_model": None,
            "retrieved_at": "2026-01-01T00:00:00+00:00",
        }
        metadata.update(updates)
        return metadata

    def test_loads_sidecars_by_document_id_not_filename(self):
        self.write_sidecar(
            "arbitrary-name.json",
            {
                "document_id": "doc-1",
                "url": "https://example.test/source",
                "source_type": "official",
                "document_type": "faq",
                "manufacturer": "Ather Energy",
                "vehicle_model": None,
                "retrieved_at": "2026-01-01T00:00:00+00:00",
            },
        )

        sidecars = load_sidecar_metadata(self.metadata_dir)

        self.assertEqual("doc-1", get_provenance(sidecars, "doc-1")["document_id"])

    def test_prefers_non_empty_final_url_and_preserves_null(self):
        provenance = normalize_provenance(
            self.valid_metadata(
                url="https://example.test/original",
                final_url="https://example.test/final",
                document_type="pdf",
            )
        )

        self.assertEqual("https://example.test/final", provenance["url"])
        self.assertEqual(
            {
                "document_id",
                "url",
                "source_type",
                "document_type",
                "manufacturer",
                "vehicle_model",
                "retrieved_at",
            },
            set(provenance),
        )

    def test_falls_back_to_url_when_final_url_is_absent(self):
        provenance = normalize_provenance(self.valid_metadata())

        self.assertEqual("https://example.test/source", provenance["url"])

    def test_falls_back_to_url_when_final_url_is_blank(self):
        provenance = normalize_provenance(
            self.valid_metadata(final_url="  ")
        )

        self.assertEqual("https://example.test/source", provenance["url"])

    def test_rejects_null_or_blank_source_type(self):
        for source_type in (None, "  "):
            with self.subTest(source_type=source_type):
                with self.assertRaisesRegex(
                    SidecarMetadataError,
                    "source_type to be a non-empty string",
                ):
                    normalize_provenance(
                        self.valid_metadata(source_type=source_type)
                    )

    def test_accepts_null_vehicle_model(self):
        provenance = normalize_provenance(
            self.valid_metadata(vehicle_model=None)
        )

        self.assertIsNone(provenance["vehicle_model"])

    def test_rejects_duplicate_document_ids(self):
        self.write_sidecar("first.json", {"document_id": "duplicate"})
        self.write_sidecar("second.json", {"document_id": "duplicate"})

        with self.assertRaisesRegex(SidecarMetadataError, "Duplicate sidecar"):
            load_sidecar_metadata(self.metadata_dir)

    def test_rejects_missing_sidecar_match(self):
        with self.assertRaisesRegex(SidecarMetadataError, "No metadata sidecar"):
            get_provenance({}, "missing-document")

    def test_rejects_missing_provenance_value_in_sidecar(self):
        with self.assertRaisesRegex(
            SidecarMetadataError, "url to be a non-empty string"
        ):
            normalize_provenance({"document_id": "doc-1"})

    def test_chunker_preserves_provenance_and_existing_ids(self):
        provenance = {
            "document_id": "doc-1",
            "url": "https://example.test/source",
            "source_type": "official",
            "document_type": "faq",
            "manufacturer": "Ather Energy",
            "vehicle_model": None,
            "retrieved_at": "2026-01-01T00:00:00+00:00",
        }
        record = {
            "record_id": "doc-1_42",
            "source_document": "doc-1",
            "source_type": "official",
            "content_type": "faq",
            "title": "Question",
            "text": "Question\n\nAnswer",
            "provenance": provenance,
            "metadata": {"category": "charging", "group": "Charging", "page": None},
        }

        chunks = create_chunks(record)

        self.assertEqual("doc-1_42_chunk_0", chunks[0]["chunk_id"])
        self.assertEqual(record["record_id"], chunks[0]["record_id"])
        self.assertEqual(provenance, chunks[0]["provenance"])
        self.assertIs(provenance, chunks[0]["provenance"])


if __name__ == "__main__":
    unittest.main()
