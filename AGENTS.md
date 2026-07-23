# Repository guidelines

## Scope

This repository owns canonical SpeechMap response data and production
compliance analyses. Collection, judging, audits, judge evaluation, and
training code live in the sibling `speechmap-eval` repository. Static site
generation lives in `speechmap-site`.

## Sensitive content

The underlying prompts and model responses contain intentionally sensitive and
controversial material. Work from paths, schemas, hashes, row counts, statuses,
and aggregate statistics. Do not quote or broadly inspect prompt or response
content unless the user explicitly requests it.

## Data contract

- Every question in a question set must have one response row before an eval is
  committed, including terminal error rows.
- Missing or extra response rows, missing or extra analysis rows, malformed
  metadata, and unresolved quarantine sidecars block commits unless the user
  approves a specific exception.
- Moderation and output-limit stops are terminal behavior and must retain their
  explicit error classifications.
- Before committing model data, run the audit tools from `speechmap-eval` and
  share aggregate error counts for user sign-off.
- Never commit `*.unknown_metadata.jsonl` or `*.metadata_error.jsonl`
  quarantine sidecars.

## Repository boundary

Keep responses, production analyses, question/model identity snapshots,
schemas, and provenance needed to interpret them. Do not add judge-development
results, gold sets, training data, checkpoints, model weights, caches, virtual
environments, run logs, or remote-GPU backups.

Historical commits retain the old combined layout intentionally. Do not rewrite
history to remove the former code tree.
