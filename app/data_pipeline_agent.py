from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .enterprise_agent import EnterpriseLLMAgent, ValidationError, WeightBundle


@dataclass(frozen=True)
class DatasetProfile:
    samples: int
    avg_text_length: float
    has_labels: bool
    source: str


class DatasetToWeightsAgent:
    """Converts enterprise datasets into training-ready weight bundle metadata."""

    def profile_dataset(self, dataset_path: str, text_field: str = "text", label_field: str = "label") -> DatasetProfile:
        path = Path(dataset_path)
        suffix = path.suffix.lower()

        if suffix == ".jsonl":
            rows = self._read_jsonl(path)
            source = "jsonl"
        elif suffix == ".csv":
            rows = self._read_csv(path)
            source = "csv"
        else:
            raise ValidationError("Unsupported dataset format. Use .csv or .jsonl")

        if not rows:
            raise ValidationError("Dataset is empty")

        texts = [str(row.get(text_field, "")) for row in rows if str(row.get(text_field, "")).strip()]
        if not texts:
            raise ValidationError(f"No usable '{text_field}' field values found")

        labeled_count = sum(1 for row in rows if str(row.get(label_field, "")).strip())
        avg_text_length = sum(len(text) for text in texts) / len(texts)

        return DatasetProfile(
            samples=len(rows),
            avg_text_length=round(avg_text_length, 2),
            has_labels=labeled_count > 0,
            source=source,
        )

    def synthesize_weight_bundle(self, profile: DatasetProfile, total_layers: int = 40) -> WeightBundle:
        if total_layers <= 0:
            raise ValidationError("total_layers must be > 0")

        richness_score = min(profile.avg_text_length / 500, 1.0)
        sample_score = min(profile.samples / 20000, 1.0)
        supervision_bonus = 0.15 if profile.has_labels else 0.0

        coverage_score = min(0.05 + (0.5 * sample_score) + (0.3 * richness_score) + supervision_bonus, 1.0)
        provided_layers = max(1, int(round(total_layers * coverage_score)))

        return WeightBundle(
            has_full_checkpoint=False,
            architecture_match=True,
            tokenizer_included=False,
            provided_layers=min(provided_layers, total_layers),
            total_layers=total_layers,
        )

    def generate_training_plan(self, profile: DatasetProfile, bundle: WeightBundle) -> dict[str, Any]:
        strategy = "supervised-finetuning" if profile.has_labels else "instruction-tuning"
        return {
            "strategy": strategy,
            "epochs": 2 if profile.samples > 10000 else 4,
            "batch_size": 32 if profile.samples > 5000 else 16,
            "target_layers": bundle.provided_layers,
            "notes": [
                "Use LoRA/QLoRA adapters for cost-efficient training.",
                "Run eval gates for regression, safety, and hallucination checks.",
                "Register model artifacts and maintain rollback checkpoints.",
            ],
        }

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if isinstance(item, dict):
                rows.append(item)
        return rows

    @staticmethod
    def _read_csv(path: Path) -> list[dict[str, Any]]:
        with path.open("r", encoding="utf-8", newline="") as file:
            return list(csv.DictReader(file))


class LLMBuildOrchestrator:
    """Runs the full pipeline from dataset profiling to LLM feasibility assessment."""

    def __init__(self) -> None:
        self.data_agent = DatasetToWeightsAgent()
        self.main_agent = EnterpriseLLMAgent()

    def run(self, dataset_path: str, total_layers: int = 40) -> dict[str, Any]:
        profile = self.data_agent.profile_dataset(dataset_path)
        bundle = self.data_agent.synthesize_weight_bundle(profile=profile, total_layers=total_layers)
        assessment = self.main_agent.assess(bundle)
        plan = self.data_agent.generate_training_plan(profile, bundle)

        return {
            "dataset_profile": {
                "samples": profile.samples,
                "avg_text_length": profile.avg_text_length,
                "has_labels": profile.has_labels,
                "source": profile.source,
            },
            "weight_bundle": {
                "provided_layers": bundle.provided_layers,
                "total_layers": bundle.total_layers,
                "coverage": round(bundle.layer_coverage, 3),
            },
            "training_plan": plan,
            "llm_assessment": assessment,
        }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dataset-to-LLM orchestration agent")
    parser.add_argument("--dataset", required=True, help="Path to enterprise dataset (.csv or .jsonl)")
    parser.add_argument("--total-layers", type=int, default=40, help="Target model total layer count")
    parser.add_argument("--output", default="", help="Optional JSON file output path")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = LLMBuildOrchestrator().run(dataset_path=args.dataset, total_layers=args.total_layers)
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
