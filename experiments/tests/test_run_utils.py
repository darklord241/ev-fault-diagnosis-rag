import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.run_utils import (
    DEFAULT_RESULTS_DIR,
    create_run_directory,
    file_sha256,
    write_run_metadata,
)


class RunUtilityTests(unittest.TestCase):
    def test_default_results_root_is_repository_results_directory(self):
        self.assertEqual(ROOT / "results", DEFAULT_RESULTS_DIR)

    def test_run_directories_use_sequential_numeric_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "01").mkdir()
            (root / "02").mkdir()
            (root / "historical-run").mkdir()

            run_id, output_dir = create_run_directory(root)

        self.assertEqual("03", run_id)
        self.assertEqual(root / "03", output_dir)

    def test_run_metadata_separates_provenance_from_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_dir = root / "01"
            output_dir.mkdir()
            config_path = output_dir / "config.json"
            config_path.write_text('{"experiment": "baseline"}\n', encoding="utf-8")
            dataset_path = root / "dataset.json"
            dataset_path.write_text("[]\n", encoding="utf-8")
            config_hash = file_sha256(config_path)
            dataset_hash = file_sha256(dataset_path)

            metadata = write_run_metadata(
                output_dir,
                run_id="01",
                config_path=config_path,
                input_files={"dataset": dataset_path},
                corpus={"chunk_count": 891},
                benchmark={"scored_query_count": 39},
            )
            stored = json.loads(
                (output_dir / "metadata.json").read_text(encoding="utf-8")
            )

        self.assertEqual(config_hash, metadata["config_sha256"])
        self.assertEqual(
            dataset_hash,
            stored["input_files"]["dataset"]["sha256"],
        )
        self.assertEqual(891, stored["corpus"]["chunk_count"])
        self.assertEqual(39, stored["benchmark"]["scored_query_count"])
        self.assertIn("git_commit", stored)
        self.assertIn("runtime", stored)


if __name__ == "__main__":
    unittest.main()
