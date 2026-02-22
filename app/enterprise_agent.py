from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ValidationError(ValueError):
    """Raised when incoming enterprise weight metadata is invalid."""


class FeasibilityTier(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class DeploymentMode(str, Enum):
    DEPLOY_AND_ADAPT = "Deploy-and-adapt"
    PARTIAL_MERGE = "Partial-merge"
    BASE_MODEL_FIRST = "Base-model-first"


@dataclass(frozen=True)
class WeightBundle:
    """Describes enterprise-provided model artifacts for feasibility assessment."""

    has_full_checkpoint: bool
    architecture_match: bool
    tokenizer_included: bool
    provided_layers: int
    total_layers: int

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "WeightBundle":
        if not isinstance(payload, dict):
            raise ValidationError("weight payload must be a JSON object")

        try:
            provided_layers = int(payload.get("provided_layers", 0))
            total_layers = int(payload.get("total_layers", 0))
        except (TypeError, ValueError) as error:
            raise ValidationError("provided_layers and total_layers must be integers") from error

        bundle = cls(
            has_full_checkpoint=_parse_bool(payload.get("has_full_checkpoint", False), "has_full_checkpoint"),
            architecture_match=_parse_bool(payload.get("architecture_match", False), "architecture_match"),
            tokenizer_included=_parse_bool(payload.get("tokenizer_included", False), "tokenizer_included"),
            provided_layers=provided_layers,
            total_layers=total_layers,
        )
        bundle.validate()
        return bundle

    def validate(self) -> None:
        if self.total_layers <= 0:
            raise ValidationError("total_layers must be > 0")
        if self.provided_layers < 0:
            raise ValidationError("provided_layers must be >= 0")
        if self.provided_layers > self.total_layers:
            raise ValidationError("provided_layers cannot exceed total_layers")

    @property
    def layer_coverage(self) -> float:
        return min(max(self.provided_layers / self.total_layers, 0.0), 1.0)


class EnterpriseLLMAgent:
    """Production-oriented feasibility assessor for enterprise LLM onboarding."""

    def assess(self, bundle: WeightBundle) -> dict[str, Any]:
        coverage = bundle.layer_coverage

        if bundle.has_full_checkpoint and bundle.architecture_match and bundle.tokenizer_included:
            tier = FeasibilityTier.HIGH
            mode = DeploymentMode.DEPLOY_AND_ADAPT
            recommendations = [
                "Load checkpoint in a compatibility-verified serving stack.",
                "Fine-tune on enterprise tasks with guarded rollout.",
                "Enable retrieval and policy controls before full launch.",
            ]
            risk_level = "low"
        elif coverage >= 0.4 and bundle.architecture_match:
            tier = FeasibilityTier.MEDIUM
            mode = DeploymentMode.PARTIAL_MERGE
            recommendations = [
                "Merge compatible partial weights/adapters with shape checks.",
                "Continue fine-tuning from a stable base model.",
                "Gate releases using regression and safety evaluation suites.",
            ]
            risk_level = "medium"
        else:
            tier = FeasibilityTier.LOW
            mode = DeploymentMode.BASE_MODEL_FIRST
            recommendations = [
                "Start from a licensed stable base model.",
                "Use LoRA/QLoRA with enterprise data for specialization.",
                "Back responses with secure retrieval for freshness and control.",
            ]
            risk_level = "high"

        return {
            "feasibility_tier": tier.value,
            "mode": mode.value,
            "coverage": round(coverage, 3),
            "risk_level": risk_level,
            "next_steps": recommendations,
            "production_controls": [
                "versioned model registry",
                "offline + online eval gates",
                "observability and audit logging",
                "rollback playbook",
            ],
        }

    def build_plan_markdown(self, assessment: dict[str, Any]) -> str:
        steps = "\n".join(f"- {step}" for step in assessment["next_steps"])
        controls = "\n".join(f"- {control}" for control in assessment["production_controls"])
        return (
            "# Enterprise LLM Agent Result\n\n"
            f"- **Feasibility tier:** {assessment['feasibility_tier']}\n"
            f"- **Mode:** {assessment['mode']}\n"
            f"- **Layer coverage:** {assessment['coverage']}\n"
            f"- **Risk level:** {assessment['risk_level']}\n\n"
            "## Recommended next steps\n"
            f"{steps}\n\n"
            "## Production controls\n"
            f"{controls}\n"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enterprise LLM feasibility agent")
    parser.add_argument("--weights", default="", help="Single JSON payload describing weights")
    parser.add_argument("--weights-file", default="", help="Path to JSON file with one payload or a list")
    parser.add_argument("--output", default="", help="Optional markdown output file path")
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    return parser.parse_args()


def _read_payloads(args: argparse.Namespace) -> list[dict[str, Any]]:
    if bool(args.weights) == bool(args.weights_file):
        raise ValidationError("Provide exactly one of --weights or --weights-file")

    if args.weights:
        return [json.loads(args.weights)]

    file_path = Path(args.weights_file)
    data = json.loads(file_path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list) and all(isinstance(item, dict) for item in data):
        return data
    raise ValidationError("weights file must contain either a JSON object or a list of JSON objects")


def _parse_bool(value: Any, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    raise ValidationError(f"{field_name} must be a boolean")


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    payloads = _read_payloads(args)
    agent = EnterpriseLLMAgent()

    assessments: list[dict[str, Any]] = []
    for index, payload in enumerate(payloads, start=1):
        bundle = WeightBundle.from_dict(payload)
        assessment = agent.assess(bundle)
        assessments.append(assessment)
        logger.info("Processed payload %s with tier=%s", index, assessment["feasibility_tier"])

    if args.output:
        report = "\n\n".join(agent.build_plan_markdown(assessment) for assessment in assessments)
        Path(args.output).write_text(report, encoding="utf-8")

    print(json.dumps(assessments[0] if len(assessments) == 1 else assessments, indent=2))


if __name__ == "__main__":
    main()
