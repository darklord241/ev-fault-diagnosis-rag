import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.evaluation import METRICS, TOP_K
from experiments.runners.baseline import (
    BASELINE_MODEL_ID,
    DEFAULT_OUTPUT_DIR,
    _baseline_model,
    build_parser,
)


class BaselineRunnerTests(unittest.TestCase):
    def test_baseline_uses_only_the_current_production_embedding_model(self):
        model = _baseline_model()

        self.assertEqual("current", model.key)
        self.assertEqual(BASELINE_MODEL_ID, model.model_id)
        self.assertEqual("", model.query_prefix)
        self.assertEqual("", model.document_prefix)

    def test_baseline_parser_has_one_fixed_configuration(self):
        args = build_parser().parse_args([])

        self.assertEqual(DEFAULT_OUTPUT_DIR, args.output_dir)
        self.assertEqual(5, TOP_K)
        self.assertEqual(
            ("hit_at_1", "hit_at_3", "hit_at_5", "mrr", "ndcg_at_5"),
            METRICS,
        )
        self.assertFalse(hasattr(args, "models"))


if __name__ == "__main__":
    unittest.main()
