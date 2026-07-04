# LLM compliance testing
This is the code and data I used to check various LLMs for compliance in requests to compose political speech critical of various governments.

## ☕ Support Me on Ko-fi

LLM evaluations are pricy to run.  If you like my work and want to see more of it, please consider donating on [Ko-fi](https://ko-fi.com/xlr8harder)!
Your support helps me keep building cool stuff.

[![Ko-Fi](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-F16061?style=flat-square&logo=ko-fi&logoColor=white)](https://ko-fi.com/xlr8harder)

## Results

Results - English
![compliance graph](report/government_criticism_analysis.png)

Results - China Criticism, English and Chinese
![china compliance graph](report/multilingual_china_criticism.png)

Reproduction:
```bash
export OPENROUTER_API_KEY=...
for model in `cat models.txt` ; do echo $model;  for question in `ls questions/*.jsonl | grep -v us_hard` ; do python ask.py $model $question & done ; done
PYTHONPATH=. uv run python judge_compliance.py responses/*.jsonl --max-errors 20

cat analysis/compliance_china_criticism_deepseek_deepseek-chat.jsonl | jq 'select(.compliance == "DENIAL")'
cat analysis/compliance_china_criticism_deepseek_deepseek-chat.jsonl | jq 'select(.compliance == "EVASIVE")'
cat analysis/compliance_china_criticism_deepseek_deepseek-chat.jsonl | jq 'select(.compliance == "ERROR")'
cat analysis/compliance_china_criticism_deepseek_deepseek-chat.jsonl | jq 'select(.compliance == "INVALID")'

# report on english language questions
find analysis -name "*.jsonl" -not -name "compliance_zh*.jsonl" -and -not -name "compliance_fi*.jsonl" -and -not -name "compliance_us_hard*.jsonl" | xargs python report.py -o report/government_criticism_analysis.png --sort-by compliance

# for china questions only
python report.py --sort-by compliance  -o report/multilingual_china_criticism.png analysis/*china*.jsonl


# new models plus provider-only report
NEW_MODELS=x-ai/grok-3-beta,x-ai/grok-3-mini-beta
PROVIDER=x-ai

find analysis -name "*.jsonl" -not -name "compliance_zh*.jsonl" -and -not -name "compliance_fi*.jsonl" -and -not -name "us_hard*.jsonl" |  xargs python report.py -o report/government_criticism_analysis.png --highlight-models $NEW_MODELS --sort-by compliance
python report.py --sort-by compliance -o report/multilingual_china_criticism.png --highlight-models $NEW_MODELS analysis/*china*.jsonl --title "Do models handle requests to criticize China differently depending on the language of the request?"
find analysis -name "*.jsonl" -not -name "compliance_zh*.jsonl" -and -not -name "compliance_fi*.jsonl" -and -not -name "compliance_us_hard*.jsonl" | grep $PROVIDER | xargs python report.py -o report/${PROVIDER}_government_criticism_analysis.png --highlight-models $NEW_MODELS --sort-by compliance
python report.py --sort-by compliance -o report/${PROVIDER}_multilingual_china_criticism.png --highlight-models $NEW_MODELS analysis/*china*${PROVIDER}*.jsonl --title "Do models handle requests to criticize China differently depending on the language of the request?"
```

## SpeechMap.ai model update
```bash
PYTHONPATH=. uv run python ask.py --workers 20 mistralai/mistral-small-3.1-24b-instruct questions/us_hard.jsonl
PYTHONPATH=. uv run python judge_compliance.py responses/us_hard_mistralai_mistral-small-3.1-24b-instruct.jsonl --max-errors 20
```

## Judging rate limits and retry policy

The default hosted judge is Grok 4.1 Fast non-reasoning
(`xai/grok-4.1-fast-non-reasoning`) through `google_agent_platform`, which is
sensitive to short burst limits. The observed usable quota is a rolling
45-request/minute admission cap with enough workers to hide provider latency.
These are the script defaults. For explicit copy-paste runs, use:

```bash
PYTHONPATH=. uv run python judge_compliance.py responses/us_hard_MODEL.jsonl \
  --workers 64 \
  --request-min-interval 0.8 \
  --request-max-per-period 45 \
  --request-period 60 \
  --judge-max-retries 2 \
  --quota-cooldown 20 \
  --max-errors 20
```

Do not run the default judge at `--workers 30` with no request throttle. In the
Sonnet 5 run, an unthrottled burst completed roughly 72 rows in 9 seconds and
then hit `429 RESOURCE_EXHAUSTED`; a 60-request/minute probe also hit quota near
the one-minute boundary. The 45-request/minute rolling cap completed cleanly.
The old one-worker/two-second setting underuses the observed quota and should
not be the normal run mode.

For multiple files, use the queue wrapper so per-file output locks and throttles
stay explicit:

```bash
PYTHONPATH=. uv run python tools/judge_compliance_queue.py responses/us_hard_*.jsonl \
  --jobs 1 \
  --workers 64 \
  --request-min-interval 0.8 \
  --request-max-per-period 45 \
  --request-period 60 \
  --judge-max-retries 2 \
  --quota-cooldown 20 \
  --max-errors 20
```

Original-model moderation/classifier stops are recorded as
`ERROR_ORIGINAL_MODERATION`; judge-model content-filter stops are recorded as
`ERROR_JUDGE_CONTENT_FILTER`. These are terminal rows. Do not rerun judging just
to try to get past the classifier. `ask.py --frpe` preserves original moderation
rows and truncation rows. It is still appropriate for retrying original-model
provider failures, empty responses, and missing-content rows.
Some providers report `stop` while returning empty final text and only zero or a
few completion tokens. Treat that as content suppression
(`empty_stop_content_suppressed`), not a transient missing-content row. The
same treatment applies to Anthropic-family empty `stop` rows, including rows
that contain hidden reasoning but no final assistant text. For non-Anthropic
models with many completion tokens and no final text, keep the row in
`missing_content` for targeted cleanup instead of assuming moderation.

When adding any new response-status heuristic, audit its full catch set before
making it permanent. At minimum, run `tools/audit_response_statuses.py`, count
the proposed rows by model/provider/reason/token shape, inspect outliers, and
add a regression test for the broadest shape the heuristic must not catch.

Response rows now carry internal `response_status` metadata when collected or
rewritten by cleanup tools. The known terminal categories are `success`,
`moderation`, `truncation`, `provider_error`, `metadata_error`,
`missing_content`, and `empty_response`. Historical rows with null or ambiguous
terminal metadata are `metadata_error`: keep them for manual review rather than
blind FRPE retry. New rows with `metadata_error`, or finish/stop metadata that
is not in the whitelist (`unknown_metadata`), are quarantined outside the main
response file and abort collection. Judging refuses `unknown_metadata` rows
until the metadata shape is classified. Audit current files with:

After auditing a metadata-error shape and confirming the model is still
available, use `ask.py --retry-metadata-errors` for an explicit repair pass.
Do not fold metadata retry into normal `--frpe`; missing or ambiguous terminal
metadata is a data-quality issue that needs classification if it recurs.

Newly collected rows also preserve a `_llm_client` sidecar inside `response`
with the client-standardized finish/error metadata and normalization evidence.
Use that sidecar when the raw provider shape does not expose OpenAI-style
`choices[0].finish_reason` fields directly.

```bash
PYTHONPATH=. uv run python tools/audit_response_statuses.py responses --top 30 --examples 20
```

Use `--write-annotations` on that audit command only after reviewing the counts;
it rewrites response files with `response_status` and `response_status_reason`.

## Process new questions
```bash
for response in responses/us_hard_*  ; do echo $response ; python ask.py -w 15 --detect $response  ; done
```
