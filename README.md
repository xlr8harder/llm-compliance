# SpeechMap data

This repository is the canonical Git history for SpeechMap model responses and
production compliance analyses.

The repository was renamed from `llm-compliance` without rewriting history.
Historical commits therefore retain the original combined code-and-data layout
and object IDs. The current tree is data-focused; active collection, judging,
and judge-development code lives in
[`speechmap-eval`](https://github.com/xlr8harder/speechmap-eval).

## Layout

- `responses/`: original model response rows.
- `analysis/`: current production compliance judgments consumed by the site.
- `analysis.openai_gpt-4o-2024-11-20/`: retained historical judge analyses.
- `questions/`: source question snapshots needed to interpret response rows.
- `model_catalog.jsonl` and `models.txt`: model/provider identity snapshots.
- `backup/`: retained historical response and analysis recovery files.

Judge gold sets, adjudication queues, judge-training datasets, experiment
results, and judge reports belong to `speechmap-eval`.

The static generator and deployment source live in
[`speechmap-site`](https://github.com/xlr8harder/speechmap-site).

## Local layout

The normal checkout layout is:

```text
git/
├── speechmap-data/
├── speechmap-eval/
└── speechmap-site/
```

Both companion repositories default to this sibling layout. Set
`SPEECHMAP_DATA_ROOT` to this checkout when using another location.

## Data integrity

Do not treat missing or malformed rows as ordinary partial output. Response and
analysis completeness, quarantine sidecars, and retained error rows must be
audited with the tools in `speechmap-eval` before committing new model data.

Generated caches, virtual environments, remote execution state, model weights,
and reproducible intermediate datasets do not belong in this repository.
