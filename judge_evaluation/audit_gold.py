#!/usr/bin/env python3
"""Audit judge outputs against frozen gold labels and flag likely gold issues."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from compliance.data import JSONLHandler, ComplianceAnalysis, ModelResponse
from judge_evaluation.compare_judges import evaluate_file, load_manifest

LABELS = ("COMPLETE", "DENIAL", "EVASIVE")


def analysis_key(row: ComplianceAnalysis | ModelResponse) -> str:
    return f"{row.model}::{row.question_id}"


def load_analyses(paths: list[Path]) -> dict[str, dict[str, ComplianceAnalysis]]:
    analyses: dict[str, dict[str, ComplianceAnalysis]] = {}
    for path in paths:
        rows = JSONLHandler.load_jsonl(path, ComplianceAnalysis)
        analyses[path.parent.name] = {analysis_key(row): row for row in rows}
    return analyses


def load_responses(path: Path) -> dict[str, ModelResponse]:
    rows = JSONLHandler.load_jsonl(path, ModelResponse)
    return {analysis_key(row): row for row in rows}


def response_text(response: object) -> str:
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        try:
            content = response["choices"][0]["message"]["content"]
        except Exception:  # noqa: BLE001
            return json.dumps(response, ensure_ascii=True)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            return "\n".join(parts)
    return str(response)


def response_excerpt(text: str, limit: int) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def build_scorecard(manifest: dict[str, dict], analysis_paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for path in analysis_paths:
        summary, _ = evaluate_file(manifest, path)
        rows.append(
            {
                "judge_key": path.parent.name,
                "judge_model": summary["judge_model"],
                "judge_provider": summary["judge_provider"],
                "analysis_file": str(path),
                "accuracy_pct": summary["accuracy_pct"],
                "disagreement_count": summary["disagreement_count"],
                "missing_rows": summary["missing_rows"],
                "accuracy_by_bucket": summary["accuracy_by_bucket"],
            }
        )
    rows.sort(key=lambda row: (-row["accuracy_pct"], row["judge_key"]))
    return rows


def build_candidates(
    manifest: dict[str, dict],
    analyses: dict[str, dict[str, ComplianceAnalysis]],
    responses: dict[str, ModelResponse],
    consensus_threshold: int,
    excerpt_chars: int,
) -> tuple[dict, list[dict]]:
    judge_keys = list(analyses)
    by_bucket = Counter()
    by_transition = Counter()
    candidates: list[dict] = []

    for key, expected in manifest.items():
        observed = {judge_key: analyses[judge_key].get(key) for judge_key in judge_keys}
        wrong_labels = [
            row.compliance
            for row in observed.values()
            if row is not None and row.compliance in LABELS and row.compliance != expected["expected_compliance"]
        ]
        if not wrong_labels:
            continue

        label_counts = Counter(wrong_labels)
        consensus_label, consensus_count = label_counts.most_common(1)[0]
        if consensus_count < consensus_threshold:
            continue

        by_bucket[expected["bucket"]] += 1
        by_transition[(expected["expected_compliance"], consensus_label)] += 1
        response = responses.get(key)

        candidates.append(
            {
                "key": key,
                "model": expected["model"],
                "question_id": expected["question_id"],
                "bucket": expected["bucket"],
                "bucket_subtype": expected["bucket_subtype"],
                "gold_label": expected["expected_compliance"],
                "consensus_label": consensus_label,
                "consensus_count": consensus_count,
                "judge_labels": {
                    judge_key: (observed[judge_key].compliance if observed[judge_key] is not None else "MISSING")
                    for judge_key in judge_keys
                },
                "response_excerpt": (
                    response_excerpt(response_text(response.response), excerpt_chars)
                    if response is not None and response.response
                    else ""
                ),
            }
        )

    candidates.sort(
        key=lambda row: (
            -row["consensus_count"],
            row["bucket"],
            row["gold_label"],
            row["question_id"],
            row["model"],
        )
    )
    summary = {
        "judge_keys": judge_keys,
        "candidate_count": len(candidates),
        "consensus_threshold": consensus_threshold,
        "by_bucket": dict(sorted(by_bucket.items())),
        "by_transition": {
            f"{gold}->{observed}": count
            for (gold, observed), count in sorted(by_transition.items())
        },
    }
    return summary, candidates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("judge_evaluation/us_hard_sample_manifest.jsonl"),
        help="frozen manifest JSONL",
    )
    parser.add_argument(
        "--responses",
        type=Path,
        default=Path("judge_evaluation/us_hard_sample_responses.jsonl"),
        help="judge-evaluation sample responses JSONL",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("judge_evaluation/reports/gold_audit"),
        help="prefix for summary and candidate outputs",
    )
    parser.add_argument(
        "--consensus-threshold",
        type=int,
        default=3,
        help="minimum number of judges that must agree against gold on the same label",
    )
    parser.add_argument(
        "--excerpt-chars",
        type=int,
        default=240,
        help="max characters of response excerpt to include per candidate",
    )
    parser.add_argument("analysis_files", nargs="+", type=Path, help="ComplianceAnalysis JSONL files to audit")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    if not manifest:
        raise SystemExit(f"manifest is empty or unreadable: {args.manifest}")

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    analyses = load_analyses(args.analysis_files)
    responses = load_responses(args.responses)
    scorecard = build_scorecard(manifest, args.analysis_files)
    consensus_summary, candidates = build_candidates(
        manifest=manifest,
        analyses=analyses,
        responses=responses,
        consensus_threshold=args.consensus_threshold,
        excerpt_chars=args.excerpt_chars,
    )

    summary = {
        "scorecard": scorecard,
        "consensus_summary": consensus_summary,
    }
    summary_path = args.output_prefix.with_suffix(".summary.json")
    candidates_path = args.output_prefix.with_suffix(".candidates.jsonl")

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=True, indent=2, sort_keys=True)

    with candidates_path.open("w", encoding="utf-8") as f:
        for row in candidates:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    print(
        json.dumps(
            {
                "summary_out": str(summary_path),
                "candidates_out": str(candidates_path),
                "scorecard_rows": len(scorecard),
                "candidate_count": len(candidates),
            },
            ensure_ascii=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
