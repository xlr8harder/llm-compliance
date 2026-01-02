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
- If not provided, ask:
  - Whether to use `--reasoning` or `--no-reasoning`.
  - The canonical model name (use the API name unless a more specific label is needed).
  - If unsure, probe by running a small test or by checking whether reasoning
    traces appear in the response when `--reasoning` is requested.
    - Quick probe (prints `reasoning` if provided):
      `uv run mq test --provider openrouter speechmap-test <model_id> --json "test query"`
  - Default policy: if reasoning traces are returned in a probe, use `--reasoning`.
    `ask.py` then treats missing reasoning as an error, which helps catch
    misconfigured OpenRouter subproviders.

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
- If capacity errors are high, reduce `--workers` and retry.
- Errors are recorded in the output JSONL and can be inspected for diagnostics.
- Use `--force-subprovider` only when a particular subprovider is problematic
  (or if you observe one subprovider causing most errors).
2) Sanity check the output:
   - `ask.py` prints `SUMMARY total=... apparent_errors=...` and the output path.
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
