#!/usr/bin/env python3
"""Build an annotated judge-comparison sample from judged us_hard outputs.

The dataset is intended for evaluating alternative judge models on a frozen
subset of question/response pairs. It produces:

* a judge-ready ModelResponse JSONL sample
* a manifest JSONL with expected labels and bucket annotations
* a JSON summary with corpus and selection statistics

Default bucket targets:
* 100 COMPLETE
* 100 DENIAL
* 100 EVASIVE
* 100 HARD
  - hardness-ranked compliant responses
  - up to a small cap per prompt for diversity
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


ALLOWED_LABELS = {"COMPLETE", "DENIAL", "EVASIVE"}
MIN_HARD_TOTAL = 10
TOKEN_RE = re.compile(r"[A-Za-z0-9_']+")


@dataclass(frozen=True)
class Record:
    key: str
    question_id: str
    question: str
    model: str
    domain: str
    category: str
    compliance: str
    source_file: str
    response_chars: int
    response_lines: int
    response_tokens_est: int
    unique_ratio: float
    markdown_density: float
    question_total: int = 0
    question_denial_rate: float = 0.0
    question_complete_rate: float = 0.0
    question_evasive_rate: float = 0.0
    question_majority_rate: float = 0.0
    question_contention: float = 0.0
    question_entropy: float = 0.0
    question_hard_rank: int = -1
    weird_score: float = 0.0
    weird_reasons: tuple[str, ...] = ()


def stable_hash(text: str) -> int:
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest(), 16)


def extract_response_text(row: dict) -> str | None:
    try:
        content = row["response"]["choices"][0]["message"]["content"]
    except Exception:
        return None
    if not isinstance(content, str):
        return None
    content = content.strip()
    return content or None


def load_good_analysis_rows(analysis_dir: Path, analysis_files: Sequence[Path] | None = None) -> list[dict]:
    rows: list[dict] = []
    paths = list(analysis_files) if analysis_files else sorted(analysis_dir.glob("compliance_us_hard_*.jsonl"))
    for path in paths:
        file_rows: list[dict] = []
        bad_file = False
        for line in path.open():
            if not line.strip():
                continue
            try:
                file_rows.append(json.loads(line))
            except json.JSONDecodeError:
                bad_file = True
                break
        if bad_file:
            continue
        if len(file_rows) != 2120:
            continue
        if any(str(r.get("compliance", "")).startswith("ERROR") for r in file_rows):
            continue
        for row in file_rows:
            if row.get("compliance") not in ALLOWED_LABELS:
                continue
            text = extract_response_text(row)
            if not text:
                continue
            token_list = TOKEN_RE.findall(text.lower())
            token_count = max(len(token_list), 1)
            unique_ratio = len(set(token_list)) / token_count if token_list else 1.0
            md_count = text.count("\n#") + text.count("\n- ") + text.count("\n* ") + text.count("```")
            rows.append(
                {
                    "question_id": row["question_id"],
                    "question": row["question"],
                    "model": row["model"],
                    "domain": row.get("domain") or "unknown",
                    "category": row.get("category") or "us_hard",
                    "compliance": row["compliance"],
                    "source_file": path.name,
                    "response_chars": len(text),
                    "response_lines": text.count("\n") + 1,
                    "response_tokens_est": token_count,
                    "unique_ratio": unique_ratio,
                    "markdown_density": md_count / max(text.count("\n") + 1, 1),
                }
            )
    return rows


def build_records(rows: Sequence[dict]) -> list[Record]:
    question_stats: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        question_stats[row["question_id"]][row["compliance"]] += 1
        question_stats[row["question_id"]]["TOTAL"] += 1

    eligible_hard = []
    for qid, stats in question_stats.items():
        total = stats["TOTAL"]
        complete = stats["COMPLETE"]
        if total >= MIN_HARD_TOTAL and complete > 0:
            eligible_hard.append(
                (
                    complete / total,
                    stats["DENIAL"] / total,
                    stats["EVASIVE"] / total,
                    total,
                    qid,
                )
            )
    # Rank prompts by how rarely they succeed, then prefer denial-heavy prompts
    # and larger sample sizes as tie-breakers.
    eligible_hard.sort(key=lambda x: (x[0], -x[1], -x[2], -x[3], x[4]))
    hard_rank = {qid: idx + 1 for idx, (*_rest, qid) in enumerate(eligible_hard)}

    interim: list[Record] = []
    lengths: list[int] = []
    lines: list[int] = []
    markdownish: list[float] = []
    repetition: list[float] = []
    for row in rows:
        rep_score = (1.0 - row["unique_ratio"]) if row["response_tokens_est"] >= 80 else 0.0
        rec = Record(
            key=f"{row['model']}::{row['question_id']}",
            question_id=row["question_id"],
            question=row["question"],
            model=row["model"],
            domain=row["domain"],
            category=row["category"],
            compliance=row["compliance"],
            source_file=row["source_file"],
            response_chars=row["response_chars"],
            response_lines=row["response_lines"],
            response_tokens_est=row["response_tokens_est"],
            unique_ratio=row["unique_ratio"],
            markdown_density=row["markdown_density"],
            question_total=question_stats[row["question_id"]]["TOTAL"],
            question_denial_rate=question_stats[row["question_id"]]["DENIAL"] / question_stats[row["question_id"]]["TOTAL"],
            question_complete_rate=question_stats[row["question_id"]]["COMPLETE"] / question_stats[row["question_id"]]["TOTAL"],
            question_evasive_rate=question_stats[row["question_id"]]["EVASIVE"] / question_stats[row["question_id"]]["TOTAL"],
            question_majority_rate=0.0,
            question_contention=0.0,
            question_entropy=0.0,
            question_hard_rank=hard_rank.get(row["question_id"], -1),
            weird_score=0.0,
            weird_reasons=(),
        )
        interim.append(rec)
        lengths.append(rec.response_chars)
        lines.append(rec.response_lines)
        markdownish.append(rec.markdown_density)
        repetition.append(rep_score)

    sorted_lengths = sorted(lengths)
    sorted_lines = sorted(lines)
    sorted_markdownish = sorted(markdownish)
    sorted_repetition = sorted(repetition)

    def percentile(sorted_values: Sequence[float], value: float) -> float:
        if not sorted_values:
            return 0.0
        left = bisect_left(sorted_values, value)
        right = bisect_right(sorted_values, value)
        return (left + 0.5 * (right - left)) / len(sorted_values)

    final: list[Record] = []
    for rec in interim:
        rep_score = (1.0 - rec.unique_ratio) if rec.response_tokens_est >= 80 else 0.0
        rates = [
            rec.question_complete_rate,
            rec.question_denial_rate,
            rec.question_evasive_rate,
        ]
        majority_rate = max(rates)
        question_contention = 1.0 - majority_rate
        question_entropy = 0.0
        for rate in rates:
            if rate > 0:
                question_entropy -= rate * math.log(rate, 2)
        question_entropy /= math.log(3, 2)

        len_pct = percentile(sorted_lengths, rec.response_chars)
        line_pct = percentile(sorted_lines, rec.response_lines)
        md_pct = percentile(sorted_markdownish, rec.markdown_density)
        rep_pct = percentile(sorted_repetition, rep_score)
        weird_score = len_pct * 0.45 + line_pct * 0.2 + md_pct * 0.15 + rep_pct * 0.2
        weird_reasons: list[str] = []
        if len_pct >= 0.98:
            weird_reasons.append("very_long")
        if line_pct >= 0.98:
            weird_reasons.append("many_lines")
        if md_pct >= 0.98 and rec.markdown_density > 0:
            weird_reasons.append("high_markdown_density")
        if rep_pct >= 0.98 and rep_score > 0:
            weird_reasons.append("low_token_diversity")
        final.append(
            Record(
                **{
                    **rec.__dict__,
                    "question_majority_rate": majority_rate,
                    "question_contention": question_contention,
                    "question_entropy": question_entropy,
                    "weird_score": weird_score,
                    "weird_reasons": tuple(weird_reasons),
                }
            )
        )
    return final


def grouped_by_model(records: Iterable[Record], *, sort_key=None) -> dict[str, list[Record]]:
    groups: dict[str, list[Record]] = defaultdict(list)
    for rec in records:
        groups[rec.model].append(rec)
    for model, items in groups.items():
        if sort_key is None:
            items.sort(key=lambda r: stable_hash(f"{r.question_id}|{r.model}|{r.source_file}"))
        else:
            items.sort(key=sort_key)
    return dict(sorted(groups.items()))


def label_rate(rec: Record) -> float:
    if rec.compliance == "COMPLETE":
        return rec.question_complete_rate
    if rec.compliance == "DENIAL":
        return rec.question_denial_rate
    if rec.compliance == "EVASIVE":
        return rec.question_evasive_rate
    return 0.0


def standard_bucket_sort_key(rec: Record):
    # Prefer disagreement-heavy prompts first. The dedicated HARD bucket runs
    # before the standard buckets and claims the lowest-success prompts.
    return (
        -rec.question_contention,
        -rec.question_entropy,
        rec.question_hard_rank if rec.question_hard_rank > 0 else math.inf,
        abs(label_rate(rec) - (1.0 / 3.0)),
        stable_hash(f"standard|{rec.question_id}|{rec.model}|{rec.source_file}"),
    )


def select_round_robin(
    groups: dict[str, list[Record]],
    target: int,
    used_keys: set[str],
    used_qids: set[str],
    max_per_qid: int = 1,
) -> list[Record]:
    selected: list[Record] = []
    qid_counts: Counter = Counter()
    group_lists = {k: list(v) for k, v in groups.items()}
    model_order = list(group_lists.keys())
    while len(selected) < target:
        made_progress = False
        for model in model_order:
            items = group_lists[model]
            while items:
                rec = items.pop(0)
                if rec.key in used_keys:
                    continue
                if rec.question_id in used_qids:
                    continue
                if qid_counts[rec.question_id] >= max_per_qid:
                    continue
                selected.append(rec)
                used_keys.add(rec.key)
                qid_counts[rec.question_id] += 1
                used_qids.add(rec.question_id)
                made_progress = True
                break
            if len(selected) >= target:
                break
        if not made_progress:
            break
    return selected


def select_hard_ranked(
    records: Sequence[Record],
    used_keys: set[str],
    used_qids: set[str],
    target: int = 100,
    max_per_qid: int = 2,
) -> list[Record]:
    by_qid: dict[str, list[Record]] = defaultdict(list)
    for rec in records:
        if rec.compliance == "COMPLETE" and rec.question_hard_rank > 0:
            by_qid[rec.question_id].append(rec)
    ranked_qids = sorted(by_qid, key=lambda qid: (by_qid[qid][0].question_hard_rank, qid))
    for qid in ranked_qids:
        by_qid[qid].sort(key=lambda r: stable_hash(f"hard-ranked|{r.key}"))

    selected: list[Record] = []
    per_qid: Counter = Counter()
    while len(selected) < target:
        progress = False
        for qid in ranked_qids:
            if per_qid[qid] >= max_per_qid:
                continue
            items = by_qid[qid]
            while items:
                rec = items.pop(0)
                if rec.key in used_keys:
                    continue
                selected.append(rec)
                used_keys.add(rec.key)
                used_qids.add(qid)
                per_qid[qid] += 1
                progress = True
                break
            if len(selected) >= target:
                break
        if not progress:
            break
    return selected


def build_sample(records: Sequence[Record]) -> tuple[list[tuple[Record, str, str]], dict]:
    used_keys: set[str] = set()
    used_qids: set[str] = set()
    sample: list[tuple[Record, str, str]] = []

    hard_rows = [
        (rec, "HARD", "hard_ranked")
        for rec in select_hard_ranked(records, used_keys, used_qids, 100, max_per_qid=2)
    ]

    success_pool = [r for r in records if r.compliance == "COMPLETE" and r.key not in used_keys]
    denial_pool = [r for r in records if r.compliance == "DENIAL" and r.key not in used_keys]
    evasive_pool = [r for r in records if r.compliance == "EVASIVE" and r.key not in used_keys]

    success = select_round_robin(
        grouped_by_model(success_pool, sort_key=standard_bucket_sort_key),
        100,
        used_keys,
        used_qids,
        max_per_qid=1,
    )
    denial = select_round_robin(
        grouped_by_model(denial_pool, sort_key=standard_bucket_sort_key),
        100,
        used_keys,
        used_qids,
        max_per_qid=1,
    )
    evasive = select_round_robin(
        grouped_by_model(evasive_pool, sort_key=standard_bucket_sort_key),
        100,
        used_keys,
        used_qids,
        max_per_qid=1,
    )

    for rec in success:
        sample.append((rec, "COMPLETE", "standard_success"))
    for rec in denial:
        sample.append((rec, "DENIAL", "standard_denial"))
    for rec in evasive:
        sample.append((rec, "EVASIVE", "standard_evasive"))
    for rec, bucket, subtype in hard_rows:
        sample.append((rec, bucket, subtype))

    bucket_counts = Counter(bucket for _rec, bucket, _subtype in sample)
    subtype_counts = Counter(subtype for _rec, _bucket, subtype in sample)
    label_counts = Counter(rec.compliance for rec, _bucket, _subtype in sample)
    prompt_coverage = Counter(rec.question_id for rec, _bucket, _subtype in sample)

    summary = {
        "total_rows": len(sample),
        "bucket_counts": dict(bucket_counts),
        "bucket_subtype_counts": dict(subtype_counts),
        "expected_label_counts": dict(label_counts),
        "unique_question_ids": len(prompt_coverage),
        "repeated_question_ids": {qid: n for qid, n in prompt_coverage.items() if n > 1},
        "hard_prompt_coverage": dict(Counter(rec.question_id for rec, bucket, _subtype in sample if bucket == "HARD")),
        "avg_question_contention_by_bucket": {
            bucket: round(
                sum(rec.question_contention for rec, row_bucket, _subtype in sample if row_bucket == bucket)
                / max(1, bucket_counts[bucket]),
                6,
            )
            for bucket in bucket_counts
        },
    }
    return sample, summary


def _emit_manifest_row(rec: Record, bucket: str, subtype: str) -> dict:
    return {
        "key": rec.key,
        "bucket": bucket,
        "bucket_subtype": subtype,
        "expected_compliance": rec.compliance,
        "question_id": rec.question_id,
        "question": rec.question,
        "domain": rec.domain,
        "category": rec.category,
        "model": rec.model,
        "source_analysis_file": rec.source_file,
        "question_total_judgments": rec.question_total,
        "question_denial_rate": round(rec.question_denial_rate, 6),
        "question_complete_rate": round(rec.question_complete_rate, 6),
        "question_evasive_rate": round(rec.question_evasive_rate, 6),
        "question_majority_rate": round(rec.question_majority_rate, 6),
        "question_contention": round(rec.question_contention, 6),
        "question_entropy": round(rec.question_entropy, 6),
        "question_hard_rank": rec.question_hard_rank,
        "response_chars": rec.response_chars,
        "response_lines": rec.response_lines,
        "response_tokens_est": rec.response_tokens_est,
        "weird_score": round(rec.weird_score, 6),
        "weird_reasons": list(rec.weird_reasons),
    }


def _emit_response_row(materialized: dict) -> dict:
    return {
        "question_id": materialized["question_id"],
        "question": materialized["question"],
        "model": materialized["model"],
        "timestamp": materialized["timestamp"],
        "response": materialized["response"],
        "api_provider": materialized.get("api_provider"),
        "api_model": materialized.get("api_model"),
        "category": materialized.get("category"),
        "domain": materialized.get("domain"),
    }


def materialize_selected_rows(
    analysis_dir: Path,
    wanted_keys: set[str],
    analysis_files: Sequence[Path] | None = None,
) -> dict[str, dict]:
    found: dict[str, dict] = {}
    paths = list(analysis_files) if analysis_files else sorted(analysis_dir.glob("compliance_us_hard_*.jsonl"))
    for path in paths:
        if len(found) == len(wanted_keys):
            break
        bad_file = False
        lines = []
        for line in path.open():
            if not line.strip():
                continue
            try:
                lines.append(json.loads(line))
            except json.JSONDecodeError:
                bad_file = True
                break
        if bad_file or len(lines) != 2120:
            continue
        if any(str(r.get("compliance", "")).startswith("ERROR") for r in lines):
            continue
        for row in lines:
            key = f"{row.get('model')}::{row.get('question_id')}"
            if key not in wanted_keys or key in found:
                continue
            response = row.get("response")
            if not isinstance(response, dict):
                continue
            if not extract_response_text(row):
                continue
            timestamp = row.get("timestamp")
            if not isinstance(timestamp, str) or not timestamp:
                continue
            found[key] = {
                "question_id": row.get("question_id"),
                "question": row.get("question"),
                "model": row.get("model"),
                "timestamp": timestamp,
                "response": response,
                "api_provider": row.get("original_api_provider"),
                "api_model": row.get("api_model"),
                "category": row.get("category") or "us_hard",
                "domain": row.get("domain") or "unknown",
            }
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=Path("analysis"))
    parser.add_argument("--analysis-files", nargs="*", type=Path, help="explicit analysis files to sample from")
    parser.add_argument(
        "--responses-out",
        type=Path,
        default=Path("judge_evaluation/us_hard_sample_responses.jsonl"),
    )
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=Path("judge_evaluation/us_hard_sample_manifest.jsonl"),
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=Path("judge_evaluation/us_hard_sample_summary.json"),
    )
    args = parser.parse_args()

    analysis_files = None
    if args.analysis_files:
        analysis_files = [p if p.is_absolute() else args.analysis_dir / p.name for p in args.analysis_files]

    raw_rows = load_good_analysis_rows(args.analysis_dir, analysis_files=analysis_files)
    records = build_records(raw_rows)
    selected, summary = build_sample(records)
    text_analysis_files = analysis_files if analysis_files else None
    materialized = materialize_selected_rows(
        args.analysis_dir,
        {rec.key for rec, _bucket, _subtype in selected},
        text_analysis_files,
    )
    manifest_rows = [
        _emit_manifest_row(rec, bucket, subtype)
        for rec, bucket, subtype in selected
        if rec.key in materialized
    ]
    response_rows = [
        _emit_response_row(materialized[rec.key])
        for rec, _bucket, _subtype in selected
        if rec.key in materialized
    ]
    summary["materialized_rows"] = len(response_rows)

    args.responses_out.parent.mkdir(parents=True, exist_ok=True)
    with args.responses_out.open("w", encoding="utf-8") as f:
        for row in response_rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest_out.open("w", encoding="utf-8") as f:
        for row in manifest_rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    with args.summary_out.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "eligible_analysis_rows": len(raw_rows),
                "eligible_records": len(records),
                **summary,
            },
            f,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )

    print(
        json.dumps(
            {
                "responses_out": str(args.responses_out),
                "manifest_out": str(args.manifest_out),
                "summary_out": str(args.summary_out),
                **summary,
            },
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
