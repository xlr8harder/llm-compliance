# llm_client V2 Readiness

## Compatibility policy

`ask.py` and `judge_compliance.py` default to `--llm-client-api legacy`.
Unchanged commands therefore retain their existing transport, retry, output,
and resume behavior.

Use `--llm-client-api v2` for an explicit migration run. Native V2 providers
currently include OpenRouter, OpenAI, Codex, and local OpenAI-compatible routes.
Other providers retain their legacy transport and import the complete
`LLMResponse` into a canonical V2 conversation. This allows gradual provider
migration without dropping source evidence or changing production requests.

Generation rows keep the existing provider response and `_llm_client` sidecar.
V2 runs additionally include `_llm_client_v2`, containing the canonical
conversation, operation, attempts, normalized result, and redacted wire record.
Existing status and judging code continues to use the V1-compatible projection.

## Bounded live comparison

On 2026-07-15, two short prompts were sent through `ask.py` using OpenRouter
`openai/gpt-5.6-sol`, Chat Completions, no reasoning, one worker, and a 256-token
limit. The same response files were then processed by `judge_compliance.py`
using the same model as judge.

Observed V1 and V2 invariants:

- Text projections matched: `HELLO` and `4`.
- Both terminal statuses were `success`.
- Both normalized finish reasons were `stop`.
- Both native finish reasons were `completed`.
- Request and raw response formats matched the existing OpenRouter Chat
  Completions labels.
- The complete V1 standardized-response key shape was retained, including IDs,
  model/provider identity, usage, and normalization evidence.
- V2 added a round-trip-stable canonical conversation with two messages and one
  operation per row.
- Both V1 and V2 judge runs classified both rows as `COMPLETE` after the
  one-character guard fix.
- Re-running both V2 entrypoints produced zero pending rows and made no model
  requests.

The comparison exposed and fixed:

- Implicit OpenRouter metadata enrichment in V2. It is now opt-in.
- Expected OpenRouter `provider` and `service_tier` fields warning as unknown.
- Missing V2 mappings for the evaluator's closed-world finish-reason taxonomy.
- V2 moderation errors not projecting legacy `type=content_filter`.
- A pre-existing judge shortcut that treated every one-character answer as a
  pathological repeated-character response.

Generated prose is nondeterministic and is reported separately from structural
compatibility. Run:

```bash
python tools/compare_llm_client_runs.py V1.jsonl V2.jsonl
```

The command fails when row coverage, status, format, terminal metadata,
retryability, or raw top-level shape differs. It reports text equality but does
not make that a compatibility requirement.

## Remaining live validation

- Exercise moderation, truncation, provider error, retry, and malformed metadata
  against real providers; synthetic contract coverage exists for these shapes.
- Run native Messages and Responses comparisons where OpenRouter supports the
  same model over both protocols.
- Validate the Google Agent Platform legacy-import bridge with live credentials.
- Run larger threaded and resume/follow probes before changing the production
  default from `legacy`.
