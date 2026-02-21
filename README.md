# Enterprise LLM Agent App

A small runnable Python agent that evaluates whether enterprise-provided model weights are enough to create/adapt an LLM, then returns practical next steps.

## Run

```bash
python -m app.enterprise_agent --weights '{"has_full_checkpoint":true,"architecture_match":true,"tokenizer_included":true,"provided_layers":32,"total_layers":32}'
```

Optional markdown output:

```bash
python -m app.enterprise_agent --weights '{"has_full_checkpoint":false,"architecture_match":true,"tokenizer_included":false,"provided_layers":20,"total_layers":40}' --output plan.md
```

## Test

```bash
python -m unittest discover -s tests -v
```
