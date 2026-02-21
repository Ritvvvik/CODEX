# Enterprise Weight-to-LLM Plan

## Short answer
You generally cannot produce a high-quality brand-new foundation LLM from only a few enterprise weights.

A practical production path is:
1. Start from a compatible base model,
2. Merge/use enterprise checkpoints where valid,
3. Fine-tune with enterprise data,
4. Add retrieval, safety controls, and evaluation gates.

## Two onboarding paths

### A) Weight-first path
Use this when the enterprise provides model assets/checkpoints.

- **Full compatible checkpoint** (architecture + tokenizer + all layers): deploy and adapt quickly.
- **Partial weights/adapters**: merge only with shape/format compatibility and continue fine-tuning.
- **Sparse vectors only**: insufficient for foundation-model creation; use base-model-first strategy.

### B) Data-first path
Use this when the enterprise provides datasets/knowledge bases.

- Profile dataset quality and scale.
- Convert profile into weight-bundle metadata heuristics.
- Generate a training plan (LoRA/QLoRA-first for cost efficiency).
- Feed synthesized bundle into the main feasibility agent.

## Production controls
- Policy and PII guardrails,
- Prompt-injection and retrieval hardening,
- Offline/online eval gates,
- Model registry + audit logs,
- Rollback strategy for releases.

## Minimal checklist
- Legally approved training/validation datasets,
- Serving/inference budget and latency targets,
- Evaluation suite and acceptance criteria,
- Versioned deployment strategy.

## Next inputs needed
- Preferred model family and deployment environment,
- What assets you have (weights/checkpoints vs datasets),
- Task type and latency/SLA needs,
- Compliance requirements.
