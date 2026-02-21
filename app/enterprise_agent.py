from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class WeightBundle:
    """Describes what model assets an enterprise can provide."""

    has_full_checkpoint: bool
    architecture_match: bool
    tokenizer_included: bool
    provided_layers: int
    total_layers: int

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "WeightBundle":
        return cls(
            has_full_checkpoint=bool(payload.get("has_full_checkpoint", False)),
            architecture_match=bool(payload.get("architecture_match", False)),
            tokenizer_included=bool(payload.get("tokenizer_included", False)),
            provided_layers=int(payload.get("provided_layers", 0)),
            total_layers=max(int(payload.get("total_layers", 0)), 1),
        )

    @property
    def layer_coverage(self) -> float:
        return min(max(self.provided_layers / self.total_layers, 0.0), 1.0)


class EnterpriseLLMAgent:
    """Agent that evaluates feasibility and returns an implementation plan."""

    def assess(self, bundle: WeightBundle) -> dict[str, Any]:
        coverage = bundle.layer_coverage

        if bundle.has_full_checkpoint and bundle.architecture_match and bundle.tokenizer_included:
            tier = "high"
            feasibility = "Deploy-and-adapt"
            recommendation = [
                "Load the checkpoint directly into a compatible serving stack.",
                "Run a short supervised fine-tune for enterprise tasks.",
                "Add RAG and policy controls before production rollout.",
            ]
        elif coverage >= 0.4 and bundle.architecture_match:
            tier = "medium"
            feasibility = "Partial-merge"
            recommendation = [
                "Merge compatible partial weights or adapters.",
                "Continue fine-tuning from a known base model.",
                "Validate with regression and hallucination benchmarks.",
            ]
        else:
            tier = "low"
            feasibility = "Base-model-first"
            recommendation = [
                "Start from a stable open/commercial base model.",
                "Use LoRA/QLoRA with enterprise data to specialize quickly.",
                "Use retrieved enterprise knowledge instead of relying on sparse weights.",
            ]

        return {
            "feasibility_tier": tier,
            "mode": feasibility,
            "coverage": round(coverage, 3),
            "next_steps": recommendation,
        }

    def build_plan_markdown(self, assessment: dict[str, Any]) -> str:
        steps = "\n".join(f"- {step}" for step in assessment["next_steps"])
        return (
            "# Enterprise LLM Agent Result\n\n"
            f"- **Feasibility tier:** {assessment['feasibility_tier']}\n"
            f"- **Mode:** {assessment['mode']}\n"
            f"- **Layer coverage:** {assessment['coverage']}\n\n"
            "## Recommended next steps\n"
            f"{steps}\n"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enterprise LLM feasibility agent")
    parser.add_argument(
        "--weights",
        required=True,
        help=(
            "JSON describing weights, e.g. "
            "'{\"has_full_checkpoint\":true,\"architecture_match\":true,"
            "\"tokenizer_included\":true,\"provided_layers\":32,\"total_layers\":32}'"
        ),
    )
    parser.add_argument("--output", default="", help="Optional markdown output file path")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload = json.loads(args.weights)
    bundle = WeightBundle.from_dict(payload)
    agent = EnterpriseLLMAgent()
    assessment = agent.assess(bundle)
    rendered = agent.build_plan_markdown(assessment)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as file:
            file.write(rendered)

    print(json.dumps(assessment, indent=2))


if __name__ == "__main__":
    main()
