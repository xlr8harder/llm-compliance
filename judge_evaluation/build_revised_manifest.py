#!/usr/bin/env python3
"""Build a revised judge-evaluation manifest from a reviewed gold-audit file."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("judge_evaluation/us_hard_sample_manifest.jsonl"),
        help="original frozen manifest JSONL",
    )
    parser.add_argument(
        "--review",
        type=Path,
        default=Path("judge_evaluation/reports/gold_audit_review_v4.jsonl"),
        help="reviewed gold-audit JSONL",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=Path("judge_evaluation/us_hard_sample_manifest_consensus_v4.jsonl"),
        help="output path for revised manifest JSONL",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=Path("judge_evaluation/us_hard_sample_manifest_consensus_v4.summary.json"),
        help="output path for revised-manifest summary JSON",
    )
    args = parser.parse_args()

    manifest_rows = load_jsonl(args.manifest)
    review_rows = load_jsonl(args.review)
    review_by_key = {row["key"]: row for row in review_rows}

    revised_rows: list[dict] = []
    original_counts = Counter()
    revised_counts = Counter()
    transition_counts = Counter()
    audit_status_counts = Counter()
    changed_keys: list[str] = []

    for row in manifest_rows:
        current = dict(row)
        key = current["key"]
        original_label = current["expected_compliance"]
        review = review_by_key.get(key)
        revised_label = original_label

        if review is not None:
            audit_status = review["audit_status"]
            audit_status_counts[audit_status] += 1
            current["gold_review_file"] = str(args.review)
            current["gold_review_status"] = audit_status
            current["gold_review_pass"] = review.get("review_pass")
            current["gold_review_confidence"] = review.get("audit_confidence")
            current["gold_review_reason_code"] = review.get("audit_reason_code")
            current["gold_review_note"] = review.get("audit_note")
            current["consensus_label"] = review.get("consensus_label")
            current["consensus_count"] = review.get("consensus_count")
            if audit_status == "likely_gold_wrong":
                revised_label = review["proposed_label"]
                current["original_expected_compliance"] = original_label
                changed_keys.append(key)

        current["expected_compliance"] = revised_label
        revised_rows.append(current)
        original_counts[original_label] += 1
        revised_counts[revised_label] += 1
        transition_counts[f"{original_label}->{revised_label}"] += 1

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", encoding="utf-8") as f:
        for row in revised_rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    summary = {
        "source_manifest": str(args.manifest),
        "review_file": str(args.review),
        "output_manifest": str(args.output_manifest),
        "manifest_rows": len(manifest_rows),
        "review_rows": len(review_rows),
        "changed_rows": len(changed_keys),
        "unchanged_reviewed_rows": audit_status_counts["gold_plausible"] + audit_status_counts["ambiguous"],
        "audit_status_counts": dict(sorted(audit_status_counts.items())),
        "original_label_counts": dict(sorted(original_counts.items())),
        "revised_label_counts": dict(sorted(revised_counts.items())),
        "transition_counts": dict(sorted(transition_counts.items())),
        "changed_keys": changed_keys,
    }

    with args.output_summary.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=True, indent=2, sort_keys=True)

    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
