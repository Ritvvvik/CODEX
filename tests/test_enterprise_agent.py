import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from app.enterprise_agent import (
    EnterpriseLLMAgent,
    ValidationError,
    WeightBundle,
    _read_payloads,
)


class EnterpriseAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = EnterpriseLLMAgent()

    def test_high_feasibility_when_full_compatible_checkpoint(self) -> None:
        bundle = WeightBundle(
            has_full_checkpoint=True,
            architecture_match=True,
            tokenizer_included=True,
            provided_layers=32,
            total_layers=32,
        )
        result = self.agent.assess(bundle)
        self.assertEqual(result["feasibility_tier"], "high")
        self.assertEqual(result["mode"], "Deploy-and-adapt")
        self.assertEqual(result["risk_level"], "low")

    def test_medium_feasibility_for_partial_compatible_weights(self) -> None:
        bundle = WeightBundle(
            has_full_checkpoint=False,
            architecture_match=True,
            tokenizer_included=False,
            provided_layers=20,
            total_layers=40,
        )
        result = self.agent.assess(bundle)
        self.assertEqual(result["feasibility_tier"], "medium")

    def test_low_feasibility_for_sparse_or_incompatible_weights(self) -> None:
        bundle = WeightBundle(
            has_full_checkpoint=False,
            architecture_match=False,
            tokenizer_included=False,
            provided_layers=3,
            total_layers=40,
        )
        result = self.agent.assess(bundle)
        self.assertEqual(result["feasibility_tier"], "low")
        self.assertEqual(result["mode"], "Base-model-first")

    def test_weight_validation_rejects_invalid_layers(self) -> None:
        with self.assertRaises(ValidationError):
            WeightBundle.from_dict({"provided_layers": 50, "total_layers": 40})


    def test_weight_validation_rejects_non_boolean_flags(self) -> None:
        with self.assertRaises(ValidationError):
            WeightBundle.from_dict(
                {
                    "has_full_checkpoint": "maybe",
                    "architecture_match": True,
                    "tokenizer_included": False,
                    "provided_layers": 4,
                    "total_layers": 8,
                }
            )

    def test_weight_validation_accepts_boolean_strings(self) -> None:
        bundle = WeightBundle.from_dict(
            {
                "has_full_checkpoint": "true",
                "architecture_match": "1",
                "tokenizer_included": "no",
                "provided_layers": 4,
                "total_layers": 8,
            }
        )
        self.assertTrue(bundle.has_full_checkpoint)
        self.assertTrue(bundle.architecture_match)
        self.assertFalse(bundle.tokenizer_included)

    def test_read_payloads_from_json_file_list(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "weights.json"
            file_path.write_text(json.dumps([{"provided_layers": 4, "total_layers": 8}]), encoding="utf-8")
            payloads = _read_payloads(Namespace(weights="", weights_file=str(file_path)))
        self.assertEqual(len(payloads), 1)
        self.assertEqual(payloads[0]["provided_layers"], 4)


if __name__ == "__main__":
    unittest.main()
