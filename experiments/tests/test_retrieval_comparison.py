import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.evaluation import retrieval_method_metrics
from experiments.runners.retrieval_comparison import (
    BM25_B,
    BM25_K1,
    CANDIDATE_POOL_SIZE,
    DEFAULT_ANNOTATIONS_PATH,
    DEFAULT_CASES_PATH,
    FINAL_TOP_K,
    METHODS,
    METHOD_BY_ID,
    RERANKER_MODEL_ID,
    RRF_K,
    _bm25_rank,
    _dense_rank,
    _evidence_rows,
    _query_rankings,
    build_bm25_index,
    final_results,
    reciprocal_rank_fusion,
    rerank_candidates,
)
from scraper.vector_search.retrieval import SearchResult, load_chunks


def make_record(chunk_id, text=None):
    return {
        "chunk_id": chunk_id,
        "record_id": f"{chunk_id}-record",
        "source_document": f"{chunk_id}.pdf",
        "source_type": "official",
        "content_type": "pdf",
        "title": chunk_id,
        "text": text or f"manual battery {chunk_id}",
        "provenance": {},
        "metadata": {},
    }


def results(*chunk_ids):
    return [
        SearchResult(score=float(len(chunk_ids) - index), record=make_record(chunk_id))
        for index, chunk_id in enumerate(chunk_ids)
    ]


class RecordingRetriever:
    def __init__(self, ranked):
        self.ranked = ranked
        self.requested_top_k = []

    def search(self, _, *, top_k):
        self.requested_top_k.append(top_k)
        return self.ranked[:top_k]


class FakeReranker:
    def __init__(self, scores):
        self.scores = scores
        self.inputs = None
        self.arguments = None

    def predict(self, inputs, **kwargs):
        self.inputs = inputs
        self.arguments = kwargs
        return self.scores


class RetrievalMethodTests(unittest.TestCase):
    def test_rrf_combines_one_based_ranks_with_fixed_reciprocal_formula(self):
        corpus = [make_record(chunk_id) for chunk_id in ("a", "b", "c", "d")]

        ranked = reciprocal_rank_fusion(
            [results("a", "b", "c"), results("c", "a", "d")],
            corpus,
            k=1,
        )

        self.assertEqual(["a", "c", "b", "d"], [
            result.record["chunk_id"] for result in ranked
        ])
        self.assertAlmostEqual(1 / 2 + 1 / 3, ranked[0].score)
        self.assertAlmostEqual(1 / 4 + 1 / 2, ranked[1].score)
        self.assertEqual(60, RRF_K)
        self.assertEqual(60, METHOD_BY_ID["R4"]["rrf_k"])

    def test_bm25_uses_fixed_okapi_parameters(self):
        records = [
            make_record("a", "battery charging instructions"),
            make_record("b", "brake inspection instructions"),
        ]

        index = build_bm25_index(records)
        ranked = _bm25_rank("battery instructions", records, index)

        self.assertEqual(BM25_K1, index.k1)
        self.assertEqual(BM25_B, index.b)
        self.assertEqual("a", ranked[0].record["chunk_id"])
        self.assertEqual(1.2, METHOD_BY_ID["R3"]["k1"])
        self.assertEqual(0.75, METHOD_BY_ID["R3"]["b"])

    def test_dense_methods_share_production_dense_cosine_top20(self):
        ranked = results(*[f"chunk-{index}" for index in range(25)])
        retriever = RecordingRetriever(ranked)
        r1 = _dense_rank(retriever, "query")
        r2 = _dense_rank(retriever, "query")

        self.assertEqual([CANDIDATE_POOL_SIZE, CANDIDATE_POOL_SIZE], retriever.requested_top_k)
        self.assertEqual(
            [(result.record["chunk_id"], result.score) for result in r1],
            [(result.record["chunk_id"], result.score) for result in r2],
        )
        self.assertEqual(
            METHOD_BY_ID["R1"]["retrieval"],
            "production_dense_cosine",
        )
        self.assertEqual(METHOD_BY_ID["R2"]["retrieval"], "dense_cosine")

    def test_rrf_receives_only_top20_dense_and_bm25_candidates(self):
        records = [make_record(f"chunk-{index}") for index in range(891)]
        dense_results = results(*[f"chunk-{index}" for index in range(891)])
        retriever = RecordingRetriever(dense_results)
        bm25_results = results(*[f"chunk-{index}" for index in range(891)])
        received_rankings = []

        def capture_rrf(rankings, corpus):
            received_rankings.append(rankings)
            self.assertEqual(records, corpus)
            return rankings[0]

        with (
            patch(
                "experiments.runners.retrieval_comparison._bm25_rank",
                return_value=bm25_results,
            ),
            patch(
                "experiments.runners.retrieval_comparison.reciprocal_rank_fusion",
                side_effect=capture_rrf,
            ),
        ):
            _query_rankings(
                "battery query",
                retriever=retriever,
                records=records,
                bm25_index=object(),
                reranker=None,
                repeats=1,
            )

        self.assertEqual([20, 20], [len(ranking) for ranking in received_rankings[0]])
        self.assertEqual(
            [CANDIDATE_POOL_SIZE] * 3,
            retriever.requested_top_k,
        )

    def test_reranker_scores_top20_then_keeps_full_ranking_for_recall20(self):
        candidates = results(*[f"chunk-{index}" for index in range(25)])
        reranker = FakeReranker(list(range(CANDIDATE_POOL_SIZE)))

        ranked = rerank_candidates("battery query", candidates, reranker)

        self.assertEqual(CANDIDATE_POOL_SIZE, len(reranker.inputs))
        self.assertEqual(CANDIDATE_POOL_SIZE, len(ranked))
        self.assertEqual(
            f"chunk-{CANDIDATE_POOL_SIZE - 1}",
            ranked[0].record["chunk_id"],
        )
        self.assertEqual(16, reranker.arguments["batch_size"])
        self.assertFalse(reranker.arguments["apply_softmax"])
        self.assertEqual(FINAL_TOP_K, METHOD_BY_ID["R5"]["final_top_k"])
        self.assertEqual(RERANKER_MODEL_ID, METHOD_BY_ID["R6"]["reranker_model_id"])
        self.assertEqual(5, len(final_results(ranked)))
        self.assertEqual(
            [result.record["chunk_id"] for result in ranked[:5]],
            [result.record["chunk_id"] for result in final_results(ranked)],
        )

    def test_evidence_contains_ranked_chunk_text_source_and_final_top5_flag(self):
        case = {"id": "q1"}
        evidence = _evidence_rows(
            case,
            "R5",
            results("c1", "c2", "c3", "c4", "c5", "c6"),
        )

        self.assertEqual(6, len(evidence))
        self.assertEqual(
            {
                "query_id",
                "method",
                "rank",
                "chunk_id",
                "score",
                "source_document",
                "chunk_text",
                "is_final_top5",
            },
            set(evidence[0]),
        )
        self.assertEqual(1, evidence[0]["rank"])
        self.assertEqual("c1.pdf", evidence[0]["source_document"])
        self.assertEqual("manual battery c1", evidence[0]["chunk_text"])
        self.assertTrue(evidence[4]["is_final_top5"])
        self.assertFalse(evidence[5]["is_final_top5"])

    def test_recall20_and_ndcg10_use_the_untruncated_candidate_ranking(self):
        ranked_ids = [f"miss-{index}" for index in range(19)] + ["gold"]

        metrics = retrieval_method_metrics(
            ranked_ids,
            gold_ids={"gold"},
        )

        self.assertEqual(0.0, metrics["recall_at_5"])
        self.assertEqual(0.0, metrics["recall_at_10"])
        self.assertEqual(1.0, metrics["recall_at_20"])
        self.assertEqual(0.0, metrics["ndcg_at_10"])

    def test_experiment_inputs_and_corpus_match_fixed_invariants(self):
        records = load_chunks(ROOT / "data/chunked")
        self.assertEqual(891, len(records))
        cases = json.loads(DEFAULT_CASES_PATH.read_text())["cases"]
        annotations = json.loads(DEFAULT_ANNOTATIONS_PATH.read_text())["cases"]
        self.assertEqual(40, len(annotations))
        self.assertEqual(40, len(cases))
        self.assertEqual(39, sum(case["gold_chunk_ids"] is not None for case in cases))
        self.assertEqual(
            ["R1", "R2", "R3", "R4", "R5", "R6"],
            [method["method"] for method in METHODS],
        )
        self.assertEqual(20, CANDIDATE_POOL_SIZE)
        self.assertEqual(5, FINAL_TOP_K)


if __name__ == "__main__":
    unittest.main()
