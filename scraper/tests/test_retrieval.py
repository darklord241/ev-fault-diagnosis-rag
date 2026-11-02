import json
import tempfile
import unittest
from pathlib import Path

from retrieval import ChunkFormatError, CosineIndex, load_chunks


def make_chunk(chunk_id):
    return {
        "chunk_id": chunk_id,
        "record_id": f"{chunk_id}_record",
        "source_document": "example-document",
        "source_type": "official",
        "content_type": "pdf",
        "title": "Example title",
        "text": f"Text for {chunk_id}",
        "provenance": {
            "document_id": "example-document",
            "url": "https://example.test/document",
        },
        "metadata": {"page": 7},
    }


class LoadChunksTests(unittest.TestCase):
    def test_loads_jsonl_files_in_sorted_order(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            (directory / "b.jsonl").write_text(
                json.dumps(make_chunk("second")) + "\n", encoding="utf-8"
            )
            (directory / "a.jsonl").write_text(
                json.dumps(make_chunk("first")) + "\n", encoding="utf-8"
            )

            records = load_chunks(directory)

        self.assertEqual(["first", "second"], [row["chunk_id"] for row in records])

    def test_empty_directory_returns_no_chunks(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            self.assertEqual([], load_chunks(Path(temporary_directory)))

    def test_missing_directory_is_an_explicit_error(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            missing_directory = Path(temporary_directory) / "missing"
            with self.assertRaises(FileNotFoundError):
                load_chunks(missing_directory)

    def test_malformed_json_reports_file_and_line(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "chunks.jsonl"
            path.write_text(
                json.dumps(make_chunk("valid")) + "\nnot-json\n", encoding="utf-8"
            )

            with self.assertRaisesRegex(
                ChunkFormatError, r"chunks\.jsonl:2: invalid JSON"
            ):
                load_chunks(path.parent)

    def test_malformed_record_reports_missing_fields(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "chunks.jsonl"
            path.write_text(json.dumps({"chunk_id": "only-id"}) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ChunkFormatError, "missing required fields"):
                load_chunks(path.parent)

    def test_empty_title_is_preserved(self):
        record = make_chunk("empty-title")
        record["title"] = ""
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "chunks.jsonl"
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")

            records = load_chunks(path.parent)

        self.assertEqual("", records[0]["title"])


class CosineIndexTests(unittest.TestCase):
    def setUp(self):
        self.records = [
            make_chunk("perpendicular"),
            make_chunk("diagonal"),
            make_chunk("aligned"),
        ]
        self.index = CosineIndex(
            self.records,
            [[0, 1], [3, 4], [1, 0]],
        )

    def test_ranks_by_cosine_similarity(self):
        results = self.index.search([1, 0], top_k=3)

        self.assertEqual(
            ["aligned", "diagonal", "perpendicular"],
            [result.record["chunk_id"] for result in results],
        )
        self.assertAlmostEqual(1.0, results[0].score)
        self.assertAlmostEqual(0.6, results[1].score)
        self.assertAlmostEqual(0.0, results[2].score)

    def test_equal_scores_keep_input_order(self):
        records = [make_chunk("first"), make_chunk("second")]
        index = CosineIndex(records, [[0, 1], [0, -1]])

        results = index.search([1, 0], top_k=2)

        self.assertEqual(["first", "second"], [r.record["chunk_id"] for r in results])
        self.assertEqual(results[0].score, results[1].score)

    def test_top_k_limits_results_and_larger_value_returns_all(self):
        self.assertEqual(2, len(self.index.search([1, 0], top_k=2)))
        self.assertEqual(3, len(self.index.search([1, 0], top_k=20)))

    def test_record_filter_applies_before_top_k(self):
        results = self.index.search(
            [1, 0],
            top_k=2,
            record_filter=lambda record: record["chunk_id"] != "aligned",
        )

        self.assertEqual(
            ["diagonal", "perpendicular"],
            [result.record["chunk_id"] for result in results],
        )

    def test_empty_index_returns_no_results(self):
        index = CosineIndex([], [])

        self.assertEqual([], index.search([1, 0]))

    def test_rejects_invalid_top_k(self):
        invalid_values = (
            (0, ValueError),
            (-1, ValueError),
            (1.5, TypeError),
            (True, TypeError),
        )
        for top_k, error in invalid_values:
            with self.subTest(top_k=top_k):
                with self.assertRaises(error):
                    self.index.search([1, 0], top_k=top_k)

    def test_rejects_zero_empty_non_numeric_and_non_finite_vectors(self):
        invalid_vectors = (
            ([], "must not be empty"),
            ([0, 0], "zero norm"),
            ([1, "bad"], "finite number"),
            ([1, float("inf")], "finite number"),
            ([10**10000], "finite number"),
        )
        for vector, message in invalid_vectors:
            with self.subTest(vector=vector):
                with self.assertRaisesRegex(ValueError, message):
                    self.index.search(vector)

    def test_rejects_invalid_stored_vectors_and_dimensions(self):
        record = make_chunk("record")
        for records, vectors, message in (
            ([record], [[0, 0]], "zero norm"),
            ([record], [[1, 0], [0, 1]], "one vector per chunk"),
            (
                [record, make_chunk("second")],
                [[1, 0], [1, 0, 0]],
                "same dimensions",
            ),
        ):
            with self.subTest(vectors=vectors):
                with self.assertRaisesRegex(ValueError, message):
                    CosineIndex(records, vectors)

    def test_rejects_query_dimension_mismatch(self):
        with self.assertRaisesRegex(ValueError, "expected 2"):
            self.index.search([1, 0, 0])

    def test_preserves_original_chunk_payload(self):
        results = self.index.search([1, 0], top_k=1)
        result = results[0]

        self.assertIs(self.records[2], result.record)
        self.assertEqual("aligned", result.record["chunk_id"])
        self.assertEqual("aligned_record", result.record["record_id"])
        self.assertEqual(
            self.records[2]["provenance"], result.record["provenance"]
        )
        self.assertEqual({"page": 7}, result.record["metadata"])
        self.assertNotIn("score", result.record)


if __name__ == "__main__":
    unittest.main()
