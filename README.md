# Enterprise LLM Agent App

Production-oriented Python agent that evaluates whether enterprise-provided model weights are sufficient for adaptation and returns rollout guidance for small-enterprise scale deployments.

## Features

- Strict input validation for weight metadata.
- Feasibility classification (`high` / `medium` / `low`) with deployment mode.
- Production controls in output (eval gates, observability, rollback, model registry).
- Supports single payload (`--weights`) or batch processing via JSON file (`--weights-file`).

## Run (single payload)

```bash
python -m app.enterprise_agent --weights '{"has_full_checkpoint":true,"architecture_match":true,"tokenizer_included":true,"provided_layers":32,"total_layers":32}'
```

## Run (batch file)

```bash
python -m app.enterprise_agent --weights-file weights.json --output plans.md
```

`weights.json` can be either a single JSON object or an array of objects.

## Test

```bash
python -m unittest discover -s tests -v
```
