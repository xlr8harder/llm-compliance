#!/usr/bin/env python3
"""Compare alternative judge outputs against the frozen judge-evaluation manifest."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from compliance.data import JSONLHandler, ComplianceAnalysis


LABELS = ("COMPLETE", "DENIAL", "EVASIVE")


def load_manifest(path: Path) -> dict[str, dict]:
    rows = JSONLHandler.load_jsonl(path)
    manifest: dict[str, dict] = {}
    for row in rows:
        key = row.get("key")
        if isinstance(key, str):
            manifest[key] = row
    return manifest


def analysis_key(row: ComplianceAnalysis) -> str:
    return f"{row.model}::{row.question_id}"


def round_pct(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round((numerator / denominator) * 100.0, 3)


def evaluate_file(manifest: dict[str, dict], analysis_path: Path) -> tuple[dict, list[dict]]:
    rows = JSONLHandler.load_jsonl(analysis_path, ComplianceAnalysis)
    by_key = {analysis_key(row): row for row in rows}

    total = len(manifest)
    matched = 0
    correct = 0
    confusion: dict[str, Counter] = {label: Counter() for label in LABELS}
    by_bucket: dict[str, Counter] = defaultdict(Counter)
    by_subtype: dict[str, Counter] = defaultdict(Counter)
    disagreements: list[dict] = []

    for key, expected in manifest.items():
        observed = by_key.get(key)
        bucket = expected["bucket"]
        subtype = expected["bucket_subtype"]
        if observed is None:
            by_bucket[bucket]["missing"] += 1
            by_subtype[subtype]["missing"] += 1
            disagreements.append(
                {
                    "key": key,
                    "bucket": bucket,
                    "bucket_subtype": subtype,
                    "expected_compliance": expected["expected_compliance"],
                    "observed_compliance": "MISSING",
                    "question_id": expected["question_id"],
                    "model": expected["model"],
                }
            )
            continue

        matched += 1
        expected_label = expected["expected_compliance"]
        observed_label = observed.compliance
        confusion.setdefault(expected_label, Counter())[observed_label] += 1
        by_bucket[bucket]["total"] += 1
        by_subtype[subtype]["total"] += 1

        if observed_label == expected_label:
            correct += 1
            by_bucket[bucket]["correct"] += 1
            by_subtype[subtype]["correct"] += 1
            continue

        disagreements.append(
            {
                "key": key,
                "bucket": bucket,
                "bucket_subtype": subtype,
                "expected_compliance": expected_label,
                "observed_compliance": observed_label,
                "question_id": expected["question_id"],
                "model": expected["model"],
                "judge_model": observed.judge_model,
                "judge_api_provider": observed.judge_api_provider,
            }
        )

    missing = total - matched
    extra_keys = sorted(set(by_key) - set(manifest))
    judge_model = rows[0].judge_model if rows else "unknown"
    judge_provider = rows[0].judge_api_provider if rows else "unknown"

    summary = {
        "analysis_file": str(analysis_path),
        "judge_model": judge_model,
        "judge_provider": judge_provider,
        "manifest_rows": total,
        "analysis_rows": len(rows),
        "matched_rows": matched,
        "missing_rows": missing,
        "extra_rows": len(extra_keys),
        "correct_rows": correct,
        "accuracy_pct": round_pct(correct, total),
        "coverage_pct": round_pct(matched, total),
        "confusion": {label: dict(confusion.get(label, Counter())) for label in LABELS},
        "accuracy_by_bucket": {
            bucket: {
                "correct": counts["correct"],
                "total": counts["total"] + counts["missing"],
                "missing": counts["missing"],
                "accuracy_pct": round_pct(counts["correct"], counts["total"] + counts["missing"]),
            }
            for bucket, counts in sorted(by_bucket.items())
        },
        "accuracy_by_subtype": {
            subtype: {
                "correct": counts["correct"],
                "total": counts["total"] + counts["missing"],
                "missing": counts["missing"],
                "accuracy_pct": round_pct(counts["correct"], counts["total"] + counts["missing"]),
            }
            for subtype, counts in sorted(by_subtype.items())
        },
        "disagreement_count": len(disagreements),
        "sample_extra_keys": extra_keys[:20],
    }
    return summary, disagreements


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("judge_evaluation/us_hard_sample_manifest.jsonl"),
        help="manifest JSONL generated by build_sample.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("judge_evaluation/reports"),
        help="directory for per-judge summaries and disagreement files",
    )
    parser.add_argument("analysis_files", nargs="+", type=Path, help="ComplianceAnalysis JSONL files to evaluate")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    if not manifest:
        raise SystemExit(f"manifest is empty or unreadable: {args.manifest}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    overview = []
    for analysis_path in args.analysis_files:
        summary, disagreements = evaluate_file(manifest, analysis_path)
        stem_parts = [analysis_path.parent.name, analysis_path.stem]
        report_stem = "__".join(part for part in stem_parts if part)
        summary_path = args.output_dir / f"{report_stem}.summary.json"
        disagreements_path = args.output_dir / f"{report_stem}.disagreements.jsonl"

        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=True, indent=2, sort_keys=True)

        with disagreements_path.open("w", encoding="utf-8") as f:
            for row in disagreements:
                f.write(json.dumps(row, ensure_ascii=True) + "\n")

        overview.append(
            {
                "analysis_file": str(analysis_path),
                "judge_model": summary["judge_model"],
                "judge_provider": summary["judge_provider"],
                "accuracy_pct": summary["accuracy_pct"],
                "coverage_pct": summary["coverage_pct"],
                "missing_rows": summary["missing_rows"],
                "disagreement_count": summary["disagreement_count"],
                "summary_out": str(summary_path),
                "disagreements_out": str(disagreements_path),
            }
        )

    print(json.dumps(overview, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
