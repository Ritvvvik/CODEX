# Enterprise LLM Agent App

Production-oriented Python toolkit for enterprise LLM onboarding with two agent flows:

1. **Weight-first flow**: evaluate enterprise-provided model weights/checkpoints.
2. **Data-first flow**: ingest enterprise knowledge base/datasets, synthesize weight metadata, and pass into the main LLM assessor.

## Repository components

- `app/enterprise_agent.py`: main feasibility and deployment-mode assessor.
- `app/data_pipeline_agent.py`: dataset profiling + weight synthesis + orchestration.
- `docs/enterprise-llm-plan.md`: architecture and rollout guidance.
- `tests/`: unit tests for both flows.

## Run: weight-first assessment

```bash
python -m app.enterprise_agent --weights '{"has_full_checkpoint":true,"architecture_match":true,"tokenizer_included":true,"provided_layers":32,"total_layers":32}'
```

## Run: data-first orchestration (dataset -> weight metadata -> LLM assessor)

```bash
python -m app.data_pipeline_agent --dataset enterprise_kb.jsonl --total-layers 40 --output build_report.json
```

The data-first pipeline performs:
- dataset profiling (sample count, average text length, supervision availability),
- weight metadata synthesis,
- training-plan proposal,
- final feasibility assessment by the main enterprise agent.


## Run: web frontend

```bash
python -m app.web_app --host 0.0.0.0 --port 8000
```

Then open `http://localhost:8000` for:
- weight-first feasibility form,
- data-first dataset orchestration form (paste JSONL/CSV directly),
- JSON result viewer.

## Test

```bash
python -m unittest discover -s tests -v
```
