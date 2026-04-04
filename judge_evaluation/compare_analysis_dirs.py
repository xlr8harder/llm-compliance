#!/usr/bin/env python3
"""Compare two compliance-analysis directories model by model."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


LABELS = ("COMPLETE", "DENIAL", "EVASIVE")


def load_by_qid(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            qid = obj.get("question_id")
            if isinstance(qid, str):
                rows[qid] = obj
    return rows


def pct(n: int, d: int) -> float:
    return round((n / d) * 100.0, 3) if d else 0.0


def summarize_file(old_path: Path, new_path: Path) -> dict:
    old_rows = load_by_qid(old_path)
    new_rows = load_by_qid(new_path)
    qids = sorted(set(old_rows) & set(new_rows))

    old_counts = Counter()
    new_counts = Counter()
    transitions = Counter()
    changed = 0
    model = None

    for qid in qids:
        old = old_rows[qid]
        new = new_rows[qid]
        model = model or old.get("model") or new.get("model")
        old_label = old.get("compliance")
        new_label = new.get("compliance")
        if old_label in LABELS:
            old_counts[old_label] += 1
        if new_label in LABELS:
            new_counts[new_label] += 1
        if old_label != new_label:
            changed += 1
            transitions[f"{old_label}->{new_label}"] += 1

    total = len(qids)
    return {
        "model": model,
        "old_analysis_file": str(old_path),
        "new_analysis_file": str(new_path),
        "matched_rows": total,
        "changed_rows": changed,
        "changed_pct": pct(changed, total),
        "old_counts": dict(old_counts),
        "new_counts": dict(new_counts),
        "old_percentages": {label: pct(old_counts[label], total) for label in LABELS},
        "new_percentages": {label: pct(new_counts[label], total) for label in LABELS},
        "percentage_point_delta": {
            label: round(pct(new_counts[label], total) - pct(old_counts[label], total), 3) for label in LABELS
        },
        "top_transitions": dict(transitions.most_common(8)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-dir", type=Path, default=Path("analysis"))
    parser.add_argument("--new-dir", type=Path, required=True)
    parser.add_argument(
        "--question-set",
        default="us_hard",
        help="question set stem used in compliance_<question_set>_<model>.jsonl",
    )
    parser.add_argument(
        "--model-list",
        type=Path,
        help="optional text file containing canonical model ids, one per line",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("judge_evaluation/reports/rejudge_dir_compare.json"),
    )
    args = parser.parse_args()

    filter_models: set[str] | None = None
    if args.model_list:
        filter_models = {
            line.strip()
            for line in args.model_list.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }

    prefix = f"compliance_{args.question_set}_"
    reports: list[dict] = []
    for old_path in sorted(args.old_dir.glob(f"{prefix}*.jsonl")):
        new_path = args.new_dir / old_path.name
        if not new_path.exists():
            continue
        sample = load_by_qid(old_path)
        model = next(iter(sample.values())).get("model") if sample else None
        if filter_models is not None and model not in filter_models:
            continue
        reports.append(summarize_file(old_path, new_path))

    reports.sort(key=lambda row: (-row["changed_pct"], row["model"] or ""))
    aggregate_old = Counter()
    aggregate_new = Counter()
    aggregate_transitions = Counter()
    matched_rows = 0
    changed_rows = 0
    for row in reports:
        matched_rows += row["matched_rows"]
        changed_rows += row["changed_rows"]
        aggregate_old.update(row["old_counts"])
        aggregate_new.update(row["new_counts"])
        aggregate_transitions.update(row["top_transitions"])

    summary = {
        "old_dir": str(args.old_dir),
        "new_dir": str(args.new_dir),
        "question_set": args.question_set,
        "models_compared": len(reports),
        "matched_rows": matched_rows,
        "changed_rows": changed_rows,
        "changed_pct": pct(changed_rows, matched_rows),
        "aggregate_old_counts": dict(aggregate_old),
        "aggregate_new_counts": dict(aggregate_new),
        "aggregate_old_percentages": {label: pct(aggregate_old[label], matched_rows) for label in LABELS},
        "aggregate_new_percentages": {label: pct(aggregate_new[label], matched_rows) for label in LABELS},
        "aggregate_percentage_point_delta": {
            label: round(
                pct(aggregate_new[label], matched_rows) - pct(aggregate_old[label], matched_rows),
                3,
            )
            for label in LABELS
        },
        "top_transitions": dict(aggregate_transitions.most_common(12)),
        "models": reports,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=True, indent=2, sort_keys=True)
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
