import unittest

from app.enterprise_agent import EnterpriseLLMAgent, WeightBundle


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


if __name__ == "__main__":
    unittest.main()
