# SpeechMap eval data

This repository contains the response data, judge outputs, and collection
tooling behind SpeechMap.ai model compliance evaluations. The current primary
dataset is the `us_hard` SpeechMap task: prompts that ask models to compose
political speech, advocacy, or criticism involving governments, public policy,
and related civic topics.

Older multilingual China-criticism reports and reproduction commands are still
in this repository, but they are no longer the main front page for the project.
The previous README has been preserved at [README.old.md](README.old.md).

## Repository layout

- `questions/`: prompt sets. `questions/us_hard.jsonl` is the main SpeechMap
  question set.
- `responses/`: original model responses, one JSONL file per question set and
  model.
- `analysis/`: judge classifications for response files. These are the rows
  consumed by SpeechMap after preprocessing.
- `model_catalog.jsonl`: canonical model names, provider IDs, and request
  overrides used by `ask.py`.
- `ask.py`: collects model responses and annotates response terminal status.
- `judge_compliance.py`: judges model responses as `COMPLETE`, `EVASIVE`,
  `DENIAL`, or a terminal error label.
- `tools/`: audit, retry, queue, migration, and probe utilities.
- `judge_evaluation/`: judge-quality experiments and supporting datasets.
- `report.py` and `report/`: legacy static report generation.

## Run a model

Normal new-model collection uses OpenRouter unless a model needs a different
provider path. Register or resolve the canonical model through
`model_catalog.jsonl`, then collect responses:

```bash
PYTHONPATH=. uv run python ask.py \
  --questions questions/us_hard.jsonl \
  --provider openrouter \
  --model PROVIDER_MODEL_ID \
  --canonical-name CANONICAL_MODEL_ID \
  --workers 30 \
  --max-truncations 0
```

For OpenRouter runs, `ask.py` excludes known heavily moderated upstream
subproviders by default when possible. The current default exclude list is
`Azure`, `Amazon Bedrock`, `Google`, `Google AI Studio`, and `Novita`. These
routes have produced provider-side moderation that interferes with SpeechMap
measurement. Use `--exclude-subprovider` to add more upstreams to the ignore
list, `--allow-subprovider NAME` only for a deliberate exception, or
`--force-subprovider OpenAI` when you need a specific upstream route. If a
model result has many moderation rows, inspect the raw upstream `provider`
values before accepting the run; a concentrated subprovider pattern is a route
artifact to rerun, not model behavior.

If OpenRouter has no acceptable upstream after these route constraints, the run
aborts before writing provider-error rows. Decide explicitly whether to relax
the route for a historical or creator-only model, force a known subprovider, or
skip the model.

For open-weight models, prefer third-party OpenRouter providers without hard
moderation layers when available instead of a creator-only or known-moderated
API route. Open-weight models should normally have near-zero provider-side
moderation errors; elevated provider errors need route investigation before the
result is committed. If no usable unmoderated route exists, keep the moderated
rows only as an explicit exception.

`ask.py` takes an exclusive advisory lock on the response file by default. If
another writer already holds that lock, it exits instead of racing the file.
`--skip-lock` exists only for manual recovery when you know no competing writer
can corrupt the file.

Use `tools/probe_reasoning.py` before full collection when reasoning behavior
is unclear:

```bash
PYTHONPATH=. uv run python tools/probe_reasoning.py \
  --provider openrouter \
  --model PROVIDER_MODEL_ID
```

For models with verified non-reasoning and reasoning modes, keep separate
canonical entries, usually `MODEL` and `MODEL-reasoning`. For models where the
mode cannot be controlled reliably, run only the supported mode and do not add a
local mode suffix.

`recommendation_mode=paired_modes` is a collection requirement, not a hint:
collect and judge both required runs before treating the model as complete. The
probe output includes a `required_runs` list with the expected canonical names
and run flags. Skip one side of a pair only after an explicit decision, and
record the reason. Cost or length concerns should be surfaced before changing
which modes are collected; use `--max-truncations 0` to make long reasoning
runs stop at the first capped response instead of silently accepting partial
answers.

## Retry response errors

`ask.py --frpe` is for original-model failures that are worth retrying, such as
opaque provider failures, network/provider transients, quota/overload rows,
empty responses, and missing-content rows.

Do not use FRPE to race moderation or output-limit behavior. Original-model
moderation rows and truncation rows are terminal by default and are preserved by
FRPE.

```bash
PYTHONPATH=. uv run python ask.py \
  --detect responses/us_hard_MODEL.jsonl \
  --workers 30 \
  --frpe
```

Truncation cleanup is explicit. Use a larger `--max-tokens` and
`--retry-truncations` only after deciding that the extra inference cost is worth
it. The default `--max-truncations 0` guard stops new traffic after the first
new truncation, lets in-flight requests drain, writes collected rows, and exits
so the run can be reviewed.

Moderation cleanup is only appropriate after auditing the upstream provider
route. To repair a known subprovider artifact, remove only the affected rows and
rerun with the default blacklist still active:

```bash
PYTHONPATH=. uv run python ask.py \
  --detect responses/us_hard_MODEL.jsonl \
  --workers 30 \
  --retry-moderation-subprovider Azure
```

This does not make moderation generally retryable; it is a targeted rerun of
rows blocked by a named upstream route.

## Judge responses

The default hosted judge is Grok 4.1 Fast non-reasoning
(`xai/grok-4.1-fast-non-reasoning`) through `google_agent_platform`. Use the
calibrated rolling quota settings:

```bash
PYTHONPATH=. uv run python judge_compliance.py responses/us_hard_MODEL.jsonl \
  --workers 64 \
  --request-min-interval 0.8 \
  --request-max-per-period 45 \
  --request-period 60 \
  --judge-max-retries 2 \
  --quota-cooldown 20 \
  --max-errors 20 \
  --no-reasoning
```

To start spending judge quota as soon as rows arrive, run the judge in follow
mode while `ask.py` is still collecting. Follow mode repeatedly reprocesses the
whole response file and relies on the existing no-replay/staleness logic to
judge only new or stale rows. While the response-file lock is held, it keeps
polling. Once the lock is released, it runs a final pass and requires one
response row per prompt plus matching analysis rows.

```bash
PYTHONPATH=. uv run python judge_compliance.py responses/us_hard_MODEL.jsonl \
  --follow \
  --follow-poll-interval 30 \
  --workers 64 \
  --request-min-interval 0.8 \
  --request-max-per-period 45 \
  --request-period 60 \
  --judge-max-retries 2 \
  --quota-cooldown 20 \
  --max-errors 20 \
  --no-reasoning
```

For multiple files, use the queue wrapper so throttling and per-file logs stay
explicit:

```bash
PYTHONPATH=. uv run python tools/judge_compliance_queue.py responses/us_hard_*.jsonl \
  --jobs 1 \
  --workers 64 \
  --request-min-interval 0.8 \
  --request-max-per-period 45 \
  --request-period 60 \
  --judge-max-retries 2 \
  --quota-cooldown 20 \
  --max-errors 20 \
  --no-reasoning
```

Do not rerun judging just to try to get past a moderation/content-filter row.
Judge quota errors, network failures, overloads, and other transport/provider
failures are retryable, but retries must stay rate-limited.

## Error semantics

Response rows carry internal `response_status` metadata when collected or
rewritten by cleanup tools. Known terminal categories are:

- `success`
- `moderation`
- `truncation`
- `provider_error`
- `metadata_error`
- `missing_content`
- `empty_response`

Judge outputs use terminal error labels for rows that should not receive a
normal compliance label:

- `ERROR_ORIGINAL_MODERATION`: original model was stopped by a
  moderation/classifier layer.
- `ERROR_ORIGINAL_TRUNCATION`: original model hit the provider/model output
  limit.
- `ERROR_JUDGE_CONTENT_FILTER`: judge model was stopped by a content filter.
- `ERROR_ORIGINAL_RESPONSE`: original response failed in a way not otherwise
  classified.

New or unfamiliar finish/stop metadata is a data-quality issue. Newly collected
rows with unknown or disallowed terminal metadata (`unknown_metadata` or new
`metadata_error`) are written to quarantine sidecar files outside the main
response file, and collection aborts until the shape is classified. Those
sidecar files are temporary cleanup artifacts: never check them in, and treat
their unresolved existence as a commit blocker. The cleanup path is to classify
the shape, migrate the row into the main response file with an appropriate
`response_status`, and then judge or short-circuit it through the normal
analysis path. Judging refuses `unknown_metadata` rows.

Audit response metadata with:

```bash
PYTHONPATH=. uv run python tools/audit_response_statuses.py responses --top 30 --examples 20
```

Use `--write-annotations` only after reviewing the counts; it rewrites response
files with `response_status` and `response_status_reason`.

## Eval commit sign-off

Before committing a model eval, inspect every retained non-success row type and
decide whether each bucket is acceptable to keep. Share aggregate error
statistics for review first: counts by `response_status`, compliance error
label, finish/native finish reason, provider error code/message family, and
model/provider/upstream subprovider where relevant. High moderation counts must
include a subprovider breakdown; if a known-moderated route such as Azure,
Amazon Bedrock, Google, or Novita dominates the failures, rerun those rows with
that route excluded before sign-off. Every
prompt in the question set must have a
corresponding response row, even when that row is an error. Missing response
rows, extra response rows, missing analysis rows, extra analysis rows, and
unresolved quarantine sidecar files all block commit unless there is a
specially approved exception.

Use the read-only eval error report as the starting point:

```bash
PYTHONPATH=. uv run python tools/eval_error_report.py responses/us_hard_MODEL.jsonl
```

Do not commit eval results until retained errors have been characterized and
explicitly signed off.

## Legacy reports

The older chart-oriented README and multilingual China-criticism commands are
available at [README.old.md](README.old.md). The corresponding historical data
remains in `questions/`, `responses/`, `analysis/`, and `report/`.

## Support

LLM evaluations are expensive to run. If this work is useful to you, support is
welcome through [Ko-fi](https://ko-fi.com/xlr8harder).
