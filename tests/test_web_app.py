import unittest

from app.enterprise_agent import ValidationError
from app.web_app import assess_weights, orchestrate_dataset


class WebAppTests(unittest.TestCase):
    def test_assess_weights_returns_assessment(self) -> None:
        result = assess_weights(
            {
                "has_full_checkpoint": "true",
                "architecture_match": "true",
                "tokenizer_included": "true",
                "provided_layers": 32,
                "total_layers": 32,
            }
        )
        self.assertEqual(result["feasibility_tier"], "high")

    def test_orchestrate_dataset_accepts_jsonl_content(self) -> None:
        result = orchestrate_dataset(
            {
                "dataset_format": "jsonl",
                "dataset_content": '{"text":"alpha policy","label":"ops"}\n{"text":"beta runbook","label":"ops"}',
                "total_layers": 32,
            }
        )
        self.assertIn("dataset_profile", result)
        self.assertIn("llm_assessment", result)

    def test_orchestrate_dataset_rejects_empty_content(self) -> None:
        with self.assertRaises(ValidationError):
            orchestrate_dataset(
                {
                    "dataset_format": "jsonl",
                    "dataset_content": "",
                    "total_layers": 32,
                }
            )


if __name__ == "__main__":
    unittest.main()
