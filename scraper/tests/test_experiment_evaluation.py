import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.evaluation import (
    EvaluationInputError,
    fuse_scores,
    load_evaluation_inputs,
    min_max_normalize,
    normalize_text,
    ranking_metrics,
    tokenize_bm25,
)
from experiments.runners.compare_retrieval import (
    _per_query_row,
    build_token_chunks,
    exclude_empty_processed_records,
    split_record_by_tokens,
)
from scraper.vector_search.retrieval import load_chunks


class OffsetTokenizer:
    def num_special_tokens_to_add(self, pair=False):
        return 2

    def __call__(
        self,
        text,
        *,
        add_special_tokens=False,
        return_offsets_mapping=False,
        truncation=False,
        verbose=True,
    ):
        import re

        offsets = [
            match.span()
            for match in re.finditer(r"\S+", text)
        ]
        return {
            "input_ids": list(range(len(offsets))),
            "offset_mapping": offsets,
        }

    def encode(self, text, *, add_special_tokens=True, truncation=False):
        import re

        token_count = len(re.findall(r"\S+", text))
        return list(range(token_count + (2 if add_special_tokens else 0)))


class BoundarySensitiveTokenizer(OffsetTokenizer):
    def encode(self, text, *, add_special_tokens=True, truncation=False):
        extra_token = 1 if text.startswith("one ") else 0
        token_count = super().encode(
            text,
            add_special_tokens=add_special_tokens,
            truncation=truncation,
        )
        return token_count + [0] * extra_token


def make_processed_record(text="one two three four five six"):
    return {
        "record_id": "doc_page_1",
        "source_document": "doc",
        "source_type": "official",
        "content_type": "pdf",
        "title": "Doc",
        "text": text,
        "provenance": {},
        "metadata": {"page": 1},
    }


class MetricTests(unittest.TestCase):
    def test_gold_id_metrics_use_binary_relevance(self):
        metrics = ranking_metrics(
            ["irrelevant", "gold-b", "gold-a"],
            gold_ids={"gold-a", "gold-b"},
        )

        self.assertEqual(0.0, metrics["hit_at_1"])
        self.assertEqual(1.0, metrics["hit_at_3"])
        self.assertAlmostEqual(0.5, metrics["mrr"])
        self.assertAlmostEqual(
            (1 / math.log2(3) + 1 / 2) / (1 + 1 / math.log2(3)),
            metrics["ndcg_at_5"],
        )

    def test_fact_metrics_use_total_relevant_chunks_for_ideal_dcg(self):
        metrics = ranking_metrics(
            ["miss", "fact-hit"],
            fact_phrases=["70% battery health"],
            text_by_id={"miss": "no", "fact-hit": "Guaranteed 70% battery health"},
            total_relevant=1,
        )

        self.assertAlmostEqual(0.5, metrics["mrr"])
        self.assertAlmostEqual(1 / math.log2(3), metrics["ndcg_at_5"])

    def test_mrr_and_first_relevant_rank_are_limited_to_top_five(self):
        metrics = ranking_metrics(
            ["miss-1", "miss-2", "miss-3", "miss-4", "miss-5", "gold"],
            gold_ids={"gold"},
        )

        self.assertEqual(0.0, metrics["mrr"])
        self.assertIsNone(metrics["first_relevant_rank"])
        self.assertEqual(0.0, metrics["hit_at_5"])

    def test_per_query_row_contains_requested_metrics_and_latency(self):
        case = {"id": "case-1"}
        config = {"config_name": "dense"}
        metrics = ranking_metrics(["miss", "gold"], gold_ids={"gold"})

        row = _per_query_row(
            case=case,
            config=config,
            query_type="exact_term",
            metrics=metrics,
            latency=0.25,
            status="complete",
        )

        self.assertEqual("case-1", row["query_id"])
        self.assertEqual("dense", row["config"])
        self.assertEqual(2, row["first_relevant_rank"])
        self.assertEqual(0.5, row["reciprocal_rank"])
        self.assertEqual(0.25, row["latency_seconds"])


class FusionTests(unittest.TestCase):
    def test_min_max_normalization_handles_constant_scores(self):
        self.assertEqual([0.0, 0.0], min_max_normalize([3.0, 3.0]))
        self.assertEqual([0.0, 0.5, 1.0], min_max_normalize([1, 2, 3]))

    def test_hybrid_fusion_applies_alpha_to_normalized_scores(self):
        scores = fuse_scores([0, 1], [1, 0], alpha=0.3)
        self.assertAlmostEqual(0.7, scores[0])
        self.assertAlmostEqual(0.3, scores[1])

    def test_bm25_tokenizer_preserves_vehicle_and_power_identifiers(self):
        self.assertEqual(
            ["ather", "700w", "450x", "p0a80"],
            tokenize_bm25("Ather 700W 450X P0A80"),
        )


class RechunkingTests(unittest.TestCase):
    def test_split_uses_token_offsets_and_stays_within_inclusive_limit(self):
        chunks, counts = build_token_chunks(
            [make_processed_record()],
            OffsetTokenizer(),
            token_limit=5,
        )

        self.assertEqual(
            ["one two three", "four five six"],
            [chunk["text"] for chunk in chunks],
        )
        self.assertEqual([5, 5], counts)
        self.assertEqual(
            ["doc_page_1_chunk_0", "doc_page_1_chunk_1"],
            [chunk["chunk_id"] for chunk in chunks],
        )
        self.assertEqual({"page": 1}, chunks[0]["metadata"])

    def test_split_checks_actual_token_count_after_slicing(self):
        chunks, counts = build_token_chunks(
            [make_processed_record("one two three four five")],
            BoundarySensitiveTokenizer(),
            token_limit=5,
        )

        self.assertEqual(["one two", "three four five"], [c["text"] for c in chunks])
        self.assertEqual([5, 5], counts)

    def test_chunker_rejects_empty_processed_text(self):
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            split_record_by_tokens(
                make_processed_record("  "),
                OffsetTokenizer(),
                token_limit=5,
            )

    def test_empty_processed_records_are_excluded_and_reported(self):
        empty_record = make_processed_record("")
        empty_record["record_id"] = "empty_page"

        usable, excluded_ids = exclude_empty_processed_records(
            [make_processed_record(), empty_record]
        )

        self.assertEqual(["doc_page_1"], [record["record_id"] for record in usable])
        self.assertEqual(["empty_page"], excluded_ids)

    def test_non_string_processed_text_is_rejected(self):
        record = make_processed_record()
        record["text"] = None
        with self.assertRaisesRegex(EvaluationInputError, "non-string text"):
            exclude_empty_processed_records([record])

    def test_text_normalization_ignores_punctuation_and_case(self):
        self.assertEqual(
            normalize_text("30 PSI front & 28 PSI back"),
            normalize_text("30 psi FRONT, 28 psi BACK"),
        )


class EvaluationInputTests(unittest.TestCase):
    def test_restored_dataset_and_key_facts_match_the_corpus(self):
        cases, annotations = load_evaluation_inputs(
            ROOT / "experiments/datasets/retrieval_eval_40.json",
            ROOT / "experiments/datasets/evaluation_annotations.json",
            load_chunks(ROOT / "scraper/data/chunked"),
        )

        self.assertEqual(40, len(cases))
        self.assertEqual(40, len(annotations))
        self.assertEqual(
            39,
            sum(case["gold_chunk_ids"] is not None for case in cases),
        )
        self.assertEqual(
            20,
            sum(
                annotations[case["id"]]["query_type"] == "exact_term"
                for case in cases
            ),
        )
        self.assertEqual(
            20,
            sum(
                annotations[case["id"]]["query_type"] == "paraphrased"
                for case in cases
            ),
        )

    def test_loader_rejects_key_facts_not_supported_by_gold_chunks(self):
        corpus = [
            {
                "chunk_id": "gold",
                "text": "The charger supports 700W.",
            }
        ]
        with tempfile.TemporaryDirectory() as temporary:
            cases_path = Path(temporary) / "retrieval_eval.json"
            annotations_path = Path(temporary) / "annotations.json"
            cases_path.write_text(
                json.dumps(
                    {
                        "cases": [
                            {
                                "id": "case",
                                "query": "What is supported?",
                                "gold_chunk_ids": ["gold"],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            annotations_path.write_text(
                json.dumps(
                    {
                        "cases": {
                            "case": {
                                "query_type": "exact_term",
                                "key_facts": ["not supported"],
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(EvaluationInputError, "not supported"):
                load_evaluation_inputs(
                    cases_path,
                    annotations_path,
                    [
                        {
                            **corpus[0],
                            "record_id": "record",
                            "source_document": "doc",
                            "source_type": "official",
                            "content_type": "faq",
                            "title": "title",
                            "provenance": {},
                            "metadata": {},
                        }
                    ],
                )


if __name__ == "__main__":
    unittest.main()
