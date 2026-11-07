import tempfile
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.evaluation import ranking_metrics
from experiments.runners.compare_embeddings import (
    _aggregate_results,
    _per_query_result,
    load_model_specs,
)
from scraper.vector_search.embedding import (
    InputTooLongError,
    SentenceTransformerTaskEmbedder,
)
from scraper.vector_search.index_embeddings import load_or_build_embeddings
from scraper.vector_search.retrieval import SearchResult


MODEL_CONFIG_SPECS = load_model_specs(
    ROOT / "experiments/configs/embedding_models.json"
)


class WhitespaceTokenizer:
    def encode(self, text, *, add_special_tokens=True, truncation=False):
        return text.split()


class FakeSentenceTransformer:
    max_seq_length = 8
    tokenizer = WhitespaceTokenizer()

    def __init__(self):
        self.encoded_texts = []

    @staticmethod
    def get_embedding_dimension():
        return 2

    def encode(self, texts, **kwargs):
        self.encoded_texts.extend(texts)
        return [[1.0, 0.0] for _ in texts]


def make_chunk(text="document text"):
    return {
        "chunk_id": "doc_chunk_0",
        "record_id": "doc_record",
        "source_document": "doc",
        "source_type": "official",
        "content_type": "pdf",
        "title": "Doc",
        "text": text,
        "provenance": {},
        "metadata": {},
    }


class SentenceTransformerTaskEmbedderTests(unittest.TestCase):
    def test_e5_formats_query_and_document_with_required_prefixes(self):
        model = FakeSentenceTransformer()
        backend = SentenceTransformerTaskEmbedder(
            "intfloat/e5-base-v2",
            query_prefix="query: ",
            document_prefix="passage: ",
            model=model,
        )

        backend.embed_query("battery warning")
        backend.embed_documents(["charger fault"])

        self.assertEqual(
            ["query: battery warning", "passage: charger fault"],
            model.encoded_texts,
        )
        self.assertIn("query_prefix='query: '", backend.identity.preprocessing)
        self.assertIn("document_prefix='passage: '", backend.identity.preprocessing)

    def test_backend_without_prefix_keeps_input_unchanged(self):
        model = FakeSentenceTransformer()
        backend = SentenceTransformerTaskEmbedder(
            "BAAI/bge-base-en-v1.5",
            model=model,
        )

        backend.embed_query("a query")
        backend.embed_documents(["a passage"])

        self.assertEqual(["a query", "a passage"], model.encoded_texts)

    def test_prefix_tokens_count_toward_limit_without_truncation(self):
        backend = SentenceTransformerTaskEmbedder(
            "intfloat/e5-base-v2",
            query_prefix="query: ",
            model=FakeSentenceTransformer(),
        )

        with self.assertRaisesRegex(InputTooLongError, "was not truncated"):
            backend.embed_query("one two three four five six seven eight")

    def test_model_config_contains_only_requested_encoder_models(self):
        self.assertEqual(
            {
                "current": "sentence-transformers/all-MiniLM-L6-v2",
                "bge-base": "BAAI/bge-base-en-v1.5",
                "e5-base": "intfloat/e5-base-v2",
            },
            {key: spec.model_id for key, spec in MODEL_CONFIG_SPECS.items()},
        )
        self.assertNotIn("query: ", MODEL_CONFIG_SPECS["bge-base"].query_prefix)
        self.assertEqual("query: ", MODEL_CONFIG_SPECS["e5-base"].query_prefix)

    def test_cache_identity_prevents_reuse_across_models(self):
        chunk = make_chunk()
        with tempfile.TemporaryDirectory() as temporary:
            cache_path = Path(temporary) / "vectors.json"
            original_model = FakeSentenceTransformer()
            original_backend = SentenceTransformerTaskEmbedder(
                "model-a", model=original_model
            )
            first = load_or_build_embeddings(
                [chunk], original_backend, cache_path
            )
            second = load_or_build_embeddings(
                [chunk],
                SentenceTransformerTaskEmbedder(
                    "model-a", model=FakeSentenceTransformer()
                ),
                cache_path,
            )
            third_model = FakeSentenceTransformer()
            third = load_or_build_embeddings(
                [chunk],
                SentenceTransformerTaskEmbedder(
                    "model-b", model=third_model
                ),
                cache_path,
            )

        self.assertFalse(first.cache_hit)
        self.assertTrue(second.cache_hit)
        self.assertFalse(third.cache_hit)
        self.assertEqual("embedding model or preprocessing differs", third.rebuild_reason)
        self.assertEqual(["document text"], third_model.encoded_texts)

    def test_model_revision_uses_loaded_commit_hash_when_available(self):
        model = FakeSentenceTransformer()
        model._modules = {
            "transformer": type(
                "TransformerModule",
                (),
                {
                    "auto_model": type(
                        "AutoModel",
                        (),
                        {
                            "config": type(
                                "Config",
                                (),
                                {"_commit_hash": "resolved-commit"},
                            )()
                        },
                    )()
                },
            )()
        }

        backend = SentenceTransformerTaskEmbedder(
            "model-a",
            revision="main",
            model=model,
        )

        self.assertEqual("resolved-commit", backend.identity.revision)

    def test_per_query_output_keeps_query_rank_scores_and_gold_ids(self):
        backend = SentenceTransformerTaskEmbedder(
            "model-a", model=FakeSentenceTransformer()
        )
        case = {
            "id": "case-1",
            "query": "Where is the charger?",
            "gold_chunk_ids": ["doc_chunk_0"],
        }
        results = [
            SearchResult(score=0.9, record=make_chunk()),
        ]
        metrics = ranking_metrics(
            ["doc_chunk_0"], gold_ids={"doc_chunk_0"}
        )

        row = _per_query_result(
            case,
            "exact_term",
            MODEL_CONFIG_SPECS["e5-base"],
            backend,
            results,
            metrics,
            0.012,
        )

        self.assertEqual("case-1", row["query_id"])
        self.assertEqual("Where is the charger?", row["query_text"])
        self.assertEqual('["doc_chunk_0"]', row["retrieved_chunk_ids"])
        self.assertEqual("[0.9]", row["retrieval_scores"])
        self.assertEqual('["doc_chunk_0"]', row["gold_chunk_ids"])
        self.assertEqual(1.0, row["mrr_at_5"])
        self.assertEqual(0.012, row["latency_seconds"])

    def test_aggregate_maps_mrr_to_per_query_mrr_at_5(self):
        backend = SentenceTransformerTaskEmbedder(
            "intfloat/e5-base-v2",
            query_prefix="query: ",
            document_prefix="passage: ",
            model=FakeSentenceTransformer(),
        )
        case = {
            "id": "case-1",
            "query": "Where is the charger?",
            "gold_chunk_ids": ["doc_chunk_0"],
        }
        per_query_row = _per_query_result(
            case,
            "exact_term",
            MODEL_CONFIG_SPECS["e5-base"],
            backend,
            [SearchResult(score=0.9, record=make_chunk())],
            ranking_metrics(["doc_chunk_0"], gold_ids={"doc_chunk_0"}),
            0.012,
        )

        rows = _aggregate_results(
            [per_query_row],
            [MODEL_CONFIG_SPECS["e5-base"]],
            {
                "e5-base": {
                    "model_id": backend.identity.model_id,
                    "revision": backend.identity.revision,
                }
            },
            total_cases=1,
            exact_case_count=1,
            paraphrased_case_count=0,
            bootstrap_samples=20,
            seed=1,
        )
        overall = next(row for row in rows if row["query_type"] == "overall")

        self.assertEqual(1.0, overall["mrr"])
        self.assertEqual(1.0, overall["hit_at_3"])


if __name__ == "__main__":
    unittest.main()
