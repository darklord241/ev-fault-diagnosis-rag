import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import call, patch

from tools.audit_pdf_batch import (
    SPECIFICATION_REVIEW_TARGETS,
    audit_pdf,
    build_report,
    page_text_counts,
    parse_pdfinfo,
)


class PdfBatchAuditTests(unittest.TestCase):
    def test_parse_pdfinfo_preserves_values_with_colons(self):
        metadata = parse_pdfinfo(
            "Title: Example PDF\nCreationDate: D:20260101000000Z\nPages: 2"
        )

        self.assertEqual("Example PDF", metadata["Title"])
        self.assertEqual("D:20260101000000Z", metadata["CreationDate"])
        self.assertEqual("2", metadata["Pages"])

    def test_page_text_counts_keep_original_numbers_and_flag_zero_text(self):
        pages, extraction_page_count = page_text_counts(
            " page one \f   \fpage three\f", expected_page_count=3
        )

        self.assertEqual(3, extraction_page_count)
        self.assertEqual([1, 2, 3], [page["page_number"] for page in pages])
        self.assertEqual(
            [7, 0, 9],
            [page["non_whitespace_chars"] for page in pages],
        )
        self.assertEqual("zero_text_human_review", pages[1]["status"])
        self.assertTrue(pages[1]["human_review_required"])

    def test_trailing_form_feed_matches_processor_split_contract(self):
        single_trailing, single_count = page_text_counts(
            "page one\f", expected_page_count=1
        )
        multiple_trailing, multiple_count = page_text_counts(
            "page one\f\f", expected_page_count=2
        )

        self.assertEqual(1, single_count)
        self.assertEqual(1, len(single_trailing))
        self.assertEqual(2, multiple_count)
        self.assertEqual(2, len(multiple_trailing))
        self.assertEqual(
            "zero_text_human_review", multiple_trailing[1]["status"]
        )

    def test_missing_extraction_page_is_not_called_blank(self):
        pages, extraction_page_count = page_text_counts(
            "page one", expected_page_count=2
        )

        self.assertEqual(1, extraction_page_count)
        self.assertEqual("missing_from_extraction", pages[1]["status"])
        self.assertIsNone(pages[1]["non_whitespace_chars"])

    def test_tool_failure_is_reported_and_other_tool_still_runs(self):
        failed_info = {
            "returncode": 127,
            "stdout": "",
            "stderr": "pdfinfo unavailable",
            "error": None,
        }
        successful_extraction = {
            "returncode": 0,
            "stdout": "one page",
            "stderr": "",
            "error": None,
        }

        with TemporaryDirectory() as temporary_directory:
            pdf_path = Path(temporary_directory) / "example.pdf"
            pdf_path.write_bytes(b"not parsed by the mocked tools")
            with patch(
                "tools.audit_pdf_batch.run_tool",
                side_effect=[failed_info, successful_extraction],
            ) as run_tool:
                report = audit_pdf(pdf_path)

        self.assertEqual(2, run_tool.call_count)
        self.assertEqual("pdfinfo", report["errors"][0]["stage"])
        self.assertEqual("text_extracted", report["pages"][0]["status"])
        run_tool.assert_has_calls(
            [
                call(["pdfinfo", str(pdf_path)]),
                call(["pdftotext", "-layout", str(pdf_path), "-"]),
            ]
        )

    def test_missing_pdfinfo_page_count_and_failed_extraction_are_explicit(self):
        failed_info = {
            "returncode": None,
            "stdout": "",
            "stderr": "No such file or directory",
            "error": "No such file or directory: 'pdfinfo'",
        }
        failed_extraction = {
            "returncode": 1,
            "stdout": "",
            "stderr": "Could not open file",
            "error": None,
        }

        with TemporaryDirectory() as temporary_directory:
            pdf_path = Path(temporary_directory) / "example.pdf"
            pdf_path.write_bytes(b"mock PDF")
            with patch(
                "tools.audit_pdf_batch.run_tool",
                side_effect=[failed_info, failed_extraction],
            ):
                report = audit_pdf(pdf_path)

        self.assertIsNone(report["page_count"])
        self.assertEqual(
            ["pdfinfo", "pdftotext"],
            [error["stage"] for error in report["errors"]],
        )
        self.assertEqual(
            [
                {
                    "page_number": None,
                    "non_whitespace_chars": None,
                    "status": "extraction_failed_page_count_unknown",
                    "human_review_required": True,
                }
            ],
            report["pages"],
        )

    def test_successful_pdfinfo_without_pages_is_reported(self):
        successful_info_without_pages = {
            "returncode": 0,
            "stdout": "Title: Example",
            "stderr": "",
            "error": None,
        }
        successful_extraction = {
            "returncode": 0,
            "stdout": "page one\f",
            "stderr": "",
            "error": None,
        }

        with TemporaryDirectory() as temporary_directory:
            pdf_path = Path(temporary_directory) / "example.pdf"
            pdf_path.write_bytes(b"mock PDF")
            with patch(
                "tools.audit_pdf_batch.run_tool",
                side_effect=[
                    successful_info_without_pages,
                    successful_extraction,
                ],
            ):
                report = audit_pdf(pdf_path)

        self.assertIsNone(report["page_count"])
        self.assertEqual(1, len(report["pages"]))
        self.assertEqual("text_extracted", report["pages"][0]["status"])
        self.assertEqual("pdfinfo", report["errors"][0]["stage"])

    def test_nonzero_pdftotext_marks_expected_pages_as_failed(self):
        successful_info = {
            "returncode": 0,
            "stdout": "Pages: 2",
            "stderr": "",
            "error": None,
        }
        failed_extraction = {
            "returncode": 1,
            "stdout": "",
            "stderr": "Text extraction failed",
            "error": None,
        }

        with TemporaryDirectory() as temporary_directory:
            pdf_path = Path(temporary_directory) / "example.pdf"
            pdf_path.write_bytes(b"mock PDF")
            with patch(
                "tools.audit_pdf_batch.run_tool",
                side_effect=[successful_info, failed_extraction],
            ):
                report = audit_pdf(pdf_path)

        self.assertEqual(2, report["page_count"])
        self.assertEqual(
            [1, 2], [page["page_number"] for page in report["pages"]]
        )
        self.assertTrue(
            all(page["status"] == "extraction_failed" for page in report["pages"])
        )
        self.assertTrue(
            all(page["non_whitespace_chars"] is None for page in report["pages"])
        )
        self.assertEqual("pdftotext", report["errors"][0]["stage"])

    def test_page_count_mismatch_is_reported_using_processor_split_count(self):
        successful_info = {
            "returncode": 0,
            "stdout": "Pages: 2",
            "stderr": "",
            "error": None,
        }
        successful_extraction = {
            "returncode": 0,
            "stdout": "page one\f",
            "stderr": "",
            "error": None,
        }

        with TemporaryDirectory() as temporary_directory:
            pdf_path = Path(temporary_directory) / "example.pdf"
            pdf_path.write_bytes(b"mock PDF")
            with patch(
                "tools.audit_pdf_batch.run_tool",
                side_effect=[successful_info, successful_extraction],
            ):
                report = audit_pdf(pdf_path)

        self.assertEqual(1, report["extraction_page_count"])
        self.assertEqual("missing_from_extraction", report["pages"][1]["status"])
        self.assertEqual(["page_count"], [error["stage"] for error in report["errors"]])

    def test_pdf_and_unique_hash_counts_have_distinct_semantics(self):
        with TemporaryDirectory() as temporary_directory:
            input_dir = Path(temporary_directory)
            (input_dir / "first.pdf").write_bytes(b"same content")
            (input_dir / "second.pdf").write_bytes(b"same content")
            mock_report = {
                "filename": "",
                "sha256": "shared-hash",
                "errors": [],
            }
            with patch(
                "tools.audit_pdf_batch.audit_pdf",
                side_effect=lambda path: {
                    **mock_report,
                    "filename": path.name,
                },
            ):
                report = build_report(input_dir)

        self.assertEqual(2, report["pdf_count"])
        self.assertEqual(1, report["unique_sha256_count"])
        self.assertEqual("first.pdf", report["documents"][1]["duplicate_of"])
        self.assertIn("including duplicate content", report["count_semantics"]["pdf_count"])
        self.assertIn("distinct SHA-256 hashes", report["count_semantics"]["unique_sha256_count"])

    def test_specification_tables_are_review_targets(self):
        self.assertEqual(
            {
                "Ather-450-Spec-Sheet-Oct-2024.pdf",
                "Ather-450-specsheet.pdf",
                "Rizta_specifications.pdf",
            },
            SPECIFICATION_REVIEW_TARGETS,
        )


if __name__ == "__main__":
    unittest.main()
