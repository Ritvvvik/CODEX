import json
import tempfile
import unittest
from pathlib import Path

from app.data_pipeline_agent import DatasetToWeightsAgent, LLMBuildOrchestrator
from app.enterprise_agent import ValidationError


class DataPipelineAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = DatasetToWeightsAgent()

    def test_profile_jsonl_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_path = Path(tmp_dir) / "kb.jsonl"
            dataset_path.write_text(
                "\n".join(
                    [
                        json.dumps({"text": "policy one", "label": "finance"}),
                        json.dumps({"text": "policy two", "label": "legal"}),
                    ]
                ),
                encoding="utf-8",
            )
            profile = self.agent.profile_dataset(str(dataset_path))

        self.assertEqual(profile.samples, 2)
        self.assertTrue(profile.has_labels)
        self.assertEqual(profile.source, "jsonl")

    def test_profile_rejects_empty_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_path = Path(tmp_dir) / "kb.jsonl"
            dataset_path.write_text("", encoding="utf-8")
            with self.assertRaises(ValidationError):
                self.agent.profile_dataset(str(dataset_path))

    def test_orchestrator_runs_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_path = Path(tmp_dir) / "kb.csv"
            dataset_path.write_text(
                "text,label\nretention policy document,policy\non-call runbook,ops\n",
                encoding="utf-8",
            )
            result = LLMBuildOrchestrator().run(str(dataset_path), total_layers=32)

        self.assertIn("dataset_profile", result)
        self.assertIn("weight_bundle", result)
        self.assertIn("llm_assessment", result)
        self.assertIn("architecture_match", result["weight_bundle"])
        self.assertGreaterEqual(result["weight_bundle"]["provided_layers"], 1)


if __name__ == "__main__":
    unittest.main()
