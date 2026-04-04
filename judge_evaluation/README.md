# Judge Evaluation Archive

This directory preserves the judge-migration reference set and the supporting
artifacts we decided were worth keeping.

It is separate from the main `analysis/` directory:

- `analysis/` remains the canonical destination for full judged model runs.
- `judge_evaluation/` is an archival workspace for evaluating judge models on a
  frozen sample, reviewing gold-label problems, and documenting the migration
  from the original GPT-4o-based gold set to the revised consensus-backed one.

## What Was Kept

### 1. Frozen Sample Inputs

These define the fixed sample used to compare alternate judges.

- `us_hard_sample_responses.jsonl`
  The frozen `ModelResponse` sample that all candidate judges were run on.
- `us_hard_sample_manifest.jsonl`
  The original frozen gold manifest. This is the original GPT-4o-era expected
  label set.
- `us_hard_sample_summary.json`
  Selection summary for the frozen sample.

### 2. Revised Consensus Gold

These capture the final reviewed replacement for the original gold labels.

- `reports/gold_audit.summary.json`
  Consensus-candidate summary showing where multiple judges converged against
  the original gold labels.
- `reports/gold_audit.candidates.jsonl`
  The 60 candidate rows flagged for manual review.
- `reports/gold_audit_review_v4.jsonl`
  Final reviewed decisions. This is the canonical review pass we kept.
- `reports/gold_audit_review_v4.summary.json`
  Final review summary for `v4`.
- `us_hard_sample_manifest_consensus_v4.jsonl`
  The materialized revised manifest built from the original manifest plus the
  final `v4` review decisions.
- `us_hard_sample_manifest_consensus_v4.summary.json`
  Summary of the revised manifest build. This records that 53 of 400 rows were
  relabeled, shifting the sample totals from:
  - original: `200 COMPLETE / 100 DENIAL / 100 EVASIVE`
  - revised: `219 COMPLETE / 96 DENIAL / 85 EVASIVE`

If you need one "final revised gold" artifact, use
`us_hard_sample_manifest_consensus_v4.jsonl`.

### 3. Judge Result Sets

These are the actual `judge_compliance.py` outputs for the frozen sample under
the judge models we wanted to preserve.

- `results/openai_gpt-5.4_no_reasoning/`
- `results/openai_gpt-5.4_reasoning_medium/`
- `results/openai_gpt-5.4-mini_no_reasoning/`
- `results/openai_gpt-5.4-mini_reasoning/`
- `results/openai_gpt-5.4-nano_no_reasoning/`
- `results/openai_gpt-5.4-nano_reasoning/`
- `results/openai_gpt-5-mini/`
- `results/openai_gpt-5-nano/`
- `results/qwen_qwen3.5-9b_together_no_reasoning/`
- `results/qwen_qwen3.5-9b_together_reasoning/`
- `results/x-ai_grok-4.1-fast_no_reasoning/`
- `results/x-ai_grok-4.1-fast_reasoning_medium/`
- `results/x-ai_grok-4.1-fast_no_reasoning_grok_v2/`
- `results/x-ai_grok-4.1-fast_no_reasoning_grok_v3/`
- `results/x-ai_grok-4.1-fast_no_reasoning_grok_v4a/`
- `results/x-ai_grok-4.1-fast_no_reasoning_grok_v4b/`
- `results/x-ai_grok-4.1-fast_no_reasoning_grok_v4c/`
- `results/x-ai_grok-4.1-fast_no_reasoning_grok_v5/`
- `results/x-ai_grok-4.1-fast_no_reasoning_grok_v6/`

The `grok_v2` through `grok_v6` runs are prompt-variant experiments for the
same Grok judge line.

### 4. Comparison Reports

For each kept judge run, the corresponding files in `reports/` contain:

- `<judge_key>__compliance_us_hard_sample_responses.summary.json`
- `<judge_key>__compliance_us_hard_sample_responses.disagreements.jsonl`

These score each judge result against the original frozen manifest and list the
rows where it disagreed.

### 5. Charts

- `charts/gpt4o_gold/`

This contains the chart bundle built from the kept reports and the final `v4`
review. `chart_data.json` is the machine-readable aggregate source for the
plots.

### 6. Supporting Code

- `build_sample.py`
  Builds the frozen sample and original manifest from judged `us_hard` runs.
- `compare_judges.py`
  Scores one or more judge outputs against the frozen manifest.
- `audit_gold.py`
  Finds rows where multiple judges converge against the original gold.
- `build_revised_manifest.py`
  Builds the revised manifest from the original manifest and the final `v4`
  review file.
- `compare_analysis_dirs.py`
  Compares two full analysis directories model by model.
- `make_gpt4o_gold_charts.py`
  Generates the chart bundle in `charts/gpt4o_gold/`.
- `select_us_hard_pilot.py`
  A retained helper for selecting a stratified `us_hard` pilot set.

### 7. Prompt Variants

- `prompts/`

These are the saved Grok prompt variants used during judge-prompt iteration.
They are useful only if we want to revisit why the Grok no-reasoning line was
tested with multiple prompt styles.

## What Was Removed

To keep this archive focused, the following were intentionally dropped:

- queue runtime logs
- duplicate full-run Grok baseline dumps in `real_world_compare/`
- cache files
- lock files
- old Grok rollout manifests and cost/repair/queue reports
- intermediate gold-review passes `v1` through `v3`

The retained directory is meant to preserve the actual decision record, not the
entire operational trail.

## Practical Starting Points

### Revisit the original frozen gold

Use:

- `us_hard_sample_responses.jsonl`
- `us_hard_sample_manifest.jsonl`

### Revisit the final revised gold

Use:

- `us_hard_sample_responses.jsonl`
- `us_hard_sample_manifest_consensus_v4.jsonl`
- `reports/gold_audit_review_v4.jsonl`

### Re-score a kept judge result against original gold

```bash
uv run python judge_evaluation/compare_judges.py \
  judge_evaluation/results/openai_gpt-5.4_no_reasoning/compliance_us_hard_sample_responses.jsonl
```

### Rebuild the revised manifest from the final review file

```bash
uv run python judge_evaluation/build_revised_manifest.py
```

### Rebuild the chart bundle

```bash
uv run python judge_evaluation/make_gpt4o_gold_charts.py
```

## Recommended Mental Model

If you come back to this later, treat the artifacts as:

1. `us_hard_sample_manifest.jsonl`
   The original frozen benchmark labels.
2. `reports/gold_audit_review_v4.jsonl`
   The final human review decisions on disputed rows.
3. `us_hard_sample_manifest_consensus_v4.jsonl`
   The final revised benchmark labels.
4. `results/`
   The actual judge outputs being compared.
5. `reports/` and `charts/`
   The summarized evidence explaining how the migration decision was made.
