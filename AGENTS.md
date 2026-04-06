# LLM Compliance Runbook (New Models)

This is the end-to-end process for running new models, pushing error counts
as close to zero as practical, judging compliance, and publishing to Speechmap.

## Content caution
- Inputs and outputs can include content that triggers strict API filtering.
- Avoid viewing large amounts of raw output unless necessary; sample only small
  snippets for diagnostics.

## Prereqs
- Set `OPENROUTER_API_KEY` in the environment.
- Use `uv run python` for all scripts.
- Decide the question set (default to `questions/us_hard.jsonl` unless specified).
- Pick workers (default to `--workers 30`).
- Ensure the model is resolvable via `model_catalog.jsonl` or pass
  `--canonical-name`, `--provider`, and `--model` so `ask.py` can extend it.
- Use longer timeouts for large operations in this repo:
  - `ask.py` and `judge_compliance.py`: allow at least 60 minutes.
  - `preprocess.py` in `../speechmap`: allow at least 30 minutes.
  - `git` operations (add/commit/push): allow longer timeouts due to large files.
- If not provided, ask:
  - Whether to use `--reasoning` or `--no-reasoning`.
  - The canonical model name (use the API name unless a more specific label is needed).
  - If unsure, probe with the standardized tool before the full run:
    - `uv run python tools/probe_reasoning.py --provider openrouter --model <model_id>`
    - This runs Probe A/B automatically and Probe C only when needed, then
      recommends canonical naming plus the run flags to use.
  - Default policy: if reasoning traces are returned in a probe, use `--reasoning`.
    `ask.py` then treats missing reasoning as an error, which helps catch
    misconfigured OpenRouter subproviders.

## Reasoning mode policy
- Goal: for models that support both operational modes, maintain a pair:
  - base mode: `<model>` (non-reasoning)
  - reasoning mode: `<model>-reasoning`
- Important naming distinction:
  - In this repo, `-reasoning` is usually a local canonical-name suffix, not an
    API model ID.
  - For OpenRouter retries/tests/probes, use the actual provider model ID from
    `model_catalog.jsonl` (or from prior response rows such as `api_model` /
    `response.model`) rather than assuming the canonical name is callable.
- Do a cheap mode probe before full runs when behavior is unclear.
  - Standard command:
    `uv run python tools/probe_reasoning.py --provider openrouter --model <model_id>`
  - Probe A (default behavior): no reasoning flags.
  - Probe B (reasoning enabled): `--reasoning` with no effort override.
  - Probe C (only if needed): `--reasoning --reasoning-effort medium`.
- Prefer provider/model defaults when possible.
  - If reasoning can be enabled without explicit effort, use that default for
    `<model>-reasoning`.
  - Use explicit `--reasoning-effort medium` only when required for reliable
    reasoning behavior or when defaults are inconsistent.
- How to verify probe outcomes:
  - Check response payload for `usage.completion_tokens_details.reasoning_tokens`
    and/or `message.reasoning` / `message.reasoning_details`.
  - If `--reasoning` is enabled and no reasoning appears across probes, treat as
    likely unsupported/misconfigured and do not label as reasoning mode.
- If the provider exposes only one mode (cannot reliably toggle):
  - Run only the supported mode.
  - Keep the canonical/public model name as the base identifier `<model>`.
  - Do not append `-reasoning` or other mode suffixes when there is no
    corresponding alternate mode to distinguish it from.
  - Add a note in Speechmap metadata when useful.
- Keep naming synchronized everywhere after any rename:
  - response filename, analysis filename, JSONL `model` field values,
    `model_catalog.jsonl`, and `../speechmap/model_metadata.json`.
  - For tracked files, use `git mv` (not plain `mv`) to preserve rename history.

## Run a new model (responses)
1) Run the model:
```bash
uv run python ask.py \
  --questions questions/us_hard.jsonl \
  --provider openrouter \
  --model <provider_model_id> \
  --canonical-name <canonical_model_id> \
  --workers 30 \
  --reasoning
```
Notes:
- Do not pass `--out` for normal runs. Let `ask.py` auto-name files from the
  question set and canonical model; use the printed output path in follow-up
  commands.
- If capacity errors are high, reduce `--workers` and retry.
- Errors are recorded in the output JSONL and can be inspected for diagnostics.
- Use `--force-subprovider` only when a particular subprovider is problematic
  (or if you observe one subprovider causing most errors).
2) Sanity check the output:
   - `ask.py` prints `SUMMARY total=... apparent_errors=...` and the output path.
     Copy that exact path (auto-named) for retries and judging.
   - Optional exact count:
```bash
uv run python - <<'PY'
from pathlib import Path
from compliance.data import JSONLHandler, ModelResponse
path = Path("responses/<file>.jsonl")
rows = JSONLHandler.load_jsonl(path, ModelResponse)
errors = sum(r.is_permanent_error() for r in rows)
print(f"rows={len(rows)} errors={errors}")
PY
```
3) Retry errors until near-zero (or until the count stabilizes):
```bash
uv run python ask.py --detect responses/<file>.jsonl --workers 30 --frpe
```
If the error count keeps decreasing across retries, keep going until it reaches
a stable point. After that, any remaining errors are likely moderation-layer
failures and can be accepted. Note: moderation layers can be racy (some models
start answering while moderation is still processing), so it is normal for a
few to slip through even when the only failures are moderation-related.
- Useful interpretation hints for stable residuals:
  - Moderation-layer failures can be either explicit or implicit.
  - Explicit cases may show up as clear moderation/provider block payloads or
    other obvious policy-enforcement errors.
  - Implicit cases have to be inferred from the response shape.
  - If residual rows cluster on sensitive prompts and the payload shows
    `message.content` missing/empty with `finish_reason: "stop"` and no
    explicit provider error, treat that as likely moderation-layer suppression.
  - If residual rows remain as top-level API errors, `choices[0].error`, or
    obvious 429/5xx-style upstream failures on non-sensitive prompts, treat
    that as provider instability rather than hidden moderation.

## Judge compliance
1) Run the judge:
```bash
uv run python judge_compliance.py --workers 30 responses/<file>.jsonl
```
Note: judge capacity errors are rare since we use a large commercial model.
2) Check in with the user after judging:
   - Summarize compliance counts and error rate.
   - Confirm results look plausible before proceeding to reports or publishing.
3) Check error counts in the analysis file:
```bash
uv run python - <<'PY'
from pathlib import Path
from collections import Counter
from compliance.data import JSONLHandler, ComplianceAnalysis
path = Path("analysis/compliance_<file_stem>.jsonl")
rows = JSONLHandler.load_jsonl(path, ComplianceAnalysis)
counts = Counter(r.compliance for r in rows)
print(counts)
PY
```
4) If errors remain:
   - Re-run the same command; the judge updates error rows automatically.
   - If the run aborts early, increase `--max-errors`.
   - If the file is corrupted or badly skewed, re-run with `--force-restart`.
   - If the error count keeps decreasing, keep retrying until it stabilizes.
   - After a few attempts at a stable point, accept any residual
     moderation-style errors.

## Commit in llm-compliance
- Stage only the relevant files: `responses/`, `analysis/`, and
  `model_catalog.jsonl` (if it changed).
- Never commit a run with a very large error count. If errors are still high,
  treat the run as failed and non-informative, then continue retries (`--frpe`
  and/or judge retries) until errors are near-zero or clearly stabilized at a
  small residual level.
- Check file sizes before pushing:
```bash
find responses analysis -type f -size +50M
```
- Commit and push with a concise message (user-specified if provided).

## Update Speechmap website
1) In `../speechmap`, update `model_metadata.json`:
   - `creator`, `model_name`, `model_family`, `release_date` (YYYY-MM-DD),
     `reasoning_model` (true/false), `notes`.
   - Confirm `reasoning_model` by checking a response line for reasoning traces.
2) Regenerate site:
```bash
uv run python preprocess.py
```
3) Optional preview:
```bash
python3 -m http.server 8001
```
4) Follow `../speechmap/AGENTS.md` for staging/commit rules
   (summarize changes and get approval before `git add`/`git commit`).
   Do not commit `.cache/` or `.venv/`.
5) Commit and `git push` to deploy.
