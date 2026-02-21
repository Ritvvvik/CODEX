# Enterprise LLM Agent App

Production-oriented Python agents for enterprise onboarding:
1. Weight feasibility assessment.
2. Dataset/knowledge-base to weight-bundle orchestration.

## Features

- Strict validation for weight metadata.
- Feasibility classification (`high` / `medium` / `low`) with deployment mode.
- Production controls in output (eval gates, observability, rollback, model registry).
- Dataset pipeline that profiles `.csv`/`.jsonl`, synthesizes weight-bundle metadata, and feeds it to the main LLM assessment agent.

## Run main feasibility agent

```bash
python -m app.enterprise_agent --weights '{"has_full_checkpoint":true,"architecture_match":true,"tokenizer_included":true,"provided_layers":32,"total_layers":32}'
```

## Run dataset → weights → LLM orchestration

```bash
python -m app.data_pipeline_agent --dataset enterprise_kb.jsonl --total-layers 40 --output build_report.json
```

This orchestrator:
- Reads enterprise dataset/knowledge-base datapoints,
- Profiles data quality/scale,
- Synthesizes training-ready weight bundle metadata,
- Passes that bundle into the main LLM feasibility engine,
- Emits training plan + assessment output.

## Test

```bash
python -m unittest discover -s tests -v
```
