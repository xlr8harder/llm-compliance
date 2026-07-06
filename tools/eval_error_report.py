#!/usr/bin/env python3
"""Summarize retained response and judge errors before committing an eval."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from compliance.data import ComplianceAnalysis, JSONLHandler, ModelResponse, Question


SUCCESS_COMPLIANCE_LABELS = {"COMPLETE", "EVASIVE", "DENIAL"}
SUCCESS_RESPONSE_STATUS = "success"
BLOCKING_QUARANTINE_STATUSES = ("metadata_error", "unknown_metadata")
MESSAGE_RE = re.compile(r"\s+")


def is_quarantine_sidecar(path: Path) -> bool:
    return any(path.name.endswith(f".jsonl.{status}.jsonl") for status in BLOCKING_QUARANTINE_STATUSES)


def response_paths(inputs: Iterable[Path]) -> list[Path]:
    paths: list[Path] = []
    for path in inputs:
        if path.is_dir():
            paths.extend(candidate for candidate in sorted(path.glob("*.jsonl")) if not is_quarantine_sidecar(candidate))
        elif not is_quarantine_sidecar(path):
            paths.append(path)
    return sorted(dict.fromkeys(paths))


def analysis_path_for(response_path: Path, analysis_dir: Path, suffix: str) -> Path:
    return analysis_dir / f"compliance_{response_path.stem}{suffix}.jsonl"


def infer_question_path(response_path: Path, question_dir: Path) -> Path | None:
    response_stem = response_path.stem
    question_paths = sorted(question_dir.glob("*.jsonl"), key=lambda path: len(path.stem), reverse=True)
    for question_path in question_paths:
        question_stem = question_path.stem
        if response_stem == question_stem or response_stem.startswith(f"{question_stem}_"):
            return question_path
    return None


def load_question_ids(path: Path | None) -> tuple[str | None, set[str]]:
    if path is None or not path.exists():
        return str(path) if path is not None else None, set()
    questions = JSONLHandler.load_jsonl(path, Question)
    return str(path), {question.id for question in questions}


def count_jsonl_rows(path: Path) -> int:
    rows = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows += 1
    return rows


def blocking_quarantine_files(response_path: Path) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    for status in BLOCKING_QUARANTINE_STATUSES:
        path = response_path.with_suffix(f"{response_path.suffix}.{status}.jsonl")
        if path.exists():
            files.append({"path": str(path), "rows": count_jsonl_rows(path), "status": status})
    return files


def normalize_message(value: Any, *, limit: int = 120) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = MESSAGE_RE.sub(" ", value.strip())
    if len(text) > limit:
        return f"{text[: limit - 3]}..."
    return text


def error_family_from_error(error: Any, source: str) -> str | None:
    if not error:
        return None
    if not isinstance(error, dict):
        return source

    parts = [source]
    for key in ("status_code", "code", "type"):
        value = error.get(key)
        if value is not None:
            parts.append(f"{key}={value}")

    message = normalize_message(error.get("message") or error.get("detail") or error.get("error"))
    if message is not None:
        parts.append(f"message={message}")
    return ";".join(parts)


def provider_error_family(resp: ModelResponse) -> str:
    response = resp.response if isinstance(resp.response, dict) else {}

    family = error_family_from_error(response.get("error"), "top_error")
    if family is not None:
        return family

    try:
        choice_error = response["choices"][0].get("error")
    except Exception:  # noqa: BLE001
        choice_error = None
    family = error_family_from_error(choice_error, "choice_error")
    if family is not None:
        return family

    llm_client = response.get("_llm_client")
    if isinstance(llm_client, dict):
        family = error_family_from_error(llm_client.get("error_info"), "llm_client_error")
        if family is not None:
            return family

    return "<none>"


def finish_pair(resp: ModelResponse) -> str:
    response = resp.response if isinstance(resp.response, dict) else {}
    finish_reason: Any = None
    native_finish_reason: Any = None

    try:
        choice = response["choices"][0]
    except Exception:  # noqa: BLE001
        choice = None
    if isinstance(choice, dict):
        finish_reason = choice.get("finish_reason")
        native_finish_reason = choice.get("native_finish_reason")
    else:
        finish_reason = response.get("stop_reason")

    llm_client = response.get("_llm_client")
    if isinstance(llm_client, dict):
        standardized = llm_client.get("standardized_response")
        if isinstance(standardized, dict):
            finish_reason = finish_reason if finish_reason is not None else standardized.get("finish_reason")
            native_finish_reason = (
                native_finish_reason
                if native_finish_reason is not None
                else standardized.get("native_finish_reason")
            )
        error_info = llm_client.get("error_info")
        if isinstance(error_info, dict):
            native_finish_reason = (
                native_finish_reason
                if native_finish_reason is not None
                else error_info.get("native_finish_reason")
            )

    return f"finish={finish_reason!r};native={native_finish_reason!r}"


def print_counter(title: str, counter: Counter[str], limit: int) -> None:
    print(title)
    if not counter:
        print("  <none>")
        return
    for key, count in counter.most_common(limit):
        print(f"  {key}: {count}")


def load_analysis_by_question_id(path: Path) -> dict[str, ComplianceAnalysis]:
    analyses = JSONLHandler.load_jsonl(path, ComplianceAnalysis)
    return {row.question_id: row for row in analyses}


def audit_eval(
    response_path: Path,
    analysis_path: Path,
    *,
    question_path: Path | None,
    examples: int,
) -> dict[str, Any]:
    responses = JSONLHandler.load_jsonl(response_path, ModelResponse)
    analyses_by_id = load_analysis_by_question_id(analysis_path) if analysis_path.exists() else {}
    resolved_question_path, question_ids = load_question_ids(question_path)

    response_status = Counter()
    response_reason = Counter()
    compliance = Counter()
    retained_response_status = Counter()
    retained_response_reason = Counter()
    retained_compliance = Counter()
    retained_finish = Counter()
    retained_provider_error = Counter()
    retained_model_provider = Counter()
    retained_examples: dict[str, list[str]] = defaultdict(list)

    retained_count = 0
    missing_analysis = 0
    seen_question_ids: set[str] = set()

    for resp in responses:
        seen_question_ids.add(resp.question_id)
        status, reason = resp.classify_response_status()
        analysis = analyses_by_id.get(resp.question_id)
        label = analysis.compliance if analysis is not None else "<missing_analysis>"

        response_status[status] += 1
        response_reason[reason] += 1
        compliance[label] += 1
        if analysis is None:
            missing_analysis += 1

        is_retained_error = status != SUCCESS_RESPONSE_STATUS or label not in SUCCESS_COMPLIANCE_LABELS
        if not is_retained_error:
            continue

        retained_count += 1
        retained_response_status[status] += 1
        retained_response_reason[reason] += 1
        retained_compliance[label] += 1
        retained_finish[finish_pair(resp)] += 1
        retained_provider_error[provider_error_family(resp)] += 1
        retained_model_provider[f"model={resp.model};provider={resp.api_provider};api_model={resp.api_model}"] += 1

        bucket = f"{status} | {label}"
        if len(retained_examples[bucket]) < examples:
            retained_examples[bucket].append(resp.question_id)

    extra_analysis = len(set(analyses_by_id) - seen_question_ids)
    missing_response_question_ids = sorted(question_ids - seen_question_ids)
    extra_response_question_ids = sorted(seen_question_ids - question_ids) if question_ids else []
    quarantine_files = blocking_quarantine_files(response_path)

    blocking_issues: list[str] = []
    if quarantine_files:
        count = sum(file["rows"] for file in quarantine_files)
        blocking_issues.append(f"unresolved_quarantine_files={len(quarantine_files)} rows={count}")
    if missing_response_question_ids:
        blocking_issues.append(f"missing_prompt_responses={len(missing_response_question_ids)}")
    if extra_response_question_ids:
        blocking_issues.append(f"responses_not_in_question_set={len(extra_response_question_ids)}")
    if missing_analysis:
        blocking_issues.append(f"missing_analysis_rows={missing_analysis}")
    if extra_analysis:
        blocking_issues.append(f"analysis_rows_without_response={extra_analysis}")

    return {
        "response_path": str(response_path),
        "analysis_path": str(analysis_path),
        "question_path": resolved_question_path,
        "rows": len(responses),
        "analysis_rows": len(analyses_by_id),
        "question_rows": len(question_ids) if question_ids else None,
        "missing_analysis": missing_analysis,
        "extra_analysis": extra_analysis,
        "missing_response_question_ids": missing_response_question_ids,
        "extra_response_question_ids": extra_response_question_ids,
        "blocking_quarantine_files": quarantine_files,
        "blocking_issues": blocking_issues,
        "retained_error_rows": retained_count,
        "response_status": response_status,
        "response_status_reason": response_reason,
        "compliance": compliance,
        "retained_response_status": retained_response_status,
        "retained_response_status_reason": retained_response_reason,
        "retained_compliance": retained_compliance,
        "retained_finish": retained_finish,
        "retained_provider_error": retained_provider_error,
        "retained_model_provider": retained_model_provider,
        "retained_examples": retained_examples,
    }


def print_eval_report(report: dict[str, Any], *, top: int) -> None:
    print(f"Response file: {report['response_path']}")
    print(f"Analysis file: {report['analysis_path']}")
    print(f"Question file: {report['question_path'] or '<not checked>'}")
    print(
        "Rows: "
        f"responses={report['rows']} analyses={report['analysis_rows']} "
        f"questions={report['question_rows'] if report['question_rows'] is not None else '<not checked>'} "
        f"missing_analysis={report['missing_analysis']} extra_analysis={report['extra_analysis']}"
    )
    missing_response_ids = report["missing_response_question_ids"]
    extra_response_ids = report["extra_response_question_ids"]
    print(
        "Prompt coverage: "
        f"missing_responses={len(missing_response_ids)} "
        f"extra_responses={len(extra_response_ids)}"
    )
    if missing_response_ids:
        print(f"  missing examples: {', '.join(missing_response_ids[:10])}")
    if extra_response_ids:
        print(f"  extra examples: {', '.join(extra_response_ids[:10])}")

    print("Blocking quarantine files:")
    quarantine_files = report["blocking_quarantine_files"]
    if quarantine_files:
        for file in quarantine_files:
            print(f"  {file['path']}: rows={file['rows']} status={file['status']}")
    else:
        print("  <none>")

    print("Blocking issues:")
    blocking_issues = report["blocking_issues"]
    if blocking_issues:
        for issue in blocking_issues:
            print(f"  {issue}")
    else:
        print("  <none>")

    print(f"Retained non-success/error rows: {report['retained_error_rows']}")
    print_counter("All response_status:", report["response_status"], top)
    print_counter("All compliance labels:", report["compliance"], top)
    print_counter("Retained response_status:", report["retained_response_status"], top)
    print_counter("Retained response_status_reason:", report["retained_response_status_reason"], top)
    print_counter("Retained compliance labels:", report["retained_compliance"], top)
    print_counter("Retained finish/native reasons:", report["retained_finish"], top)
    print_counter("Retained provider error families:", report["retained_provider_error"], top)
    print_counter("Retained model/provider:", report["retained_model_provider"], top)

    examples: dict[str, list[str]] = report["retained_examples"]
    print("Retained examples:")
    if not examples:
        print("  <none>")
        return
    for bucket, question_ids in sorted(examples.items()):
        print(f"  {bucket}: {', '.join(question_ids)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("responses", nargs="+", type=Path, help="response JSONL files or directories")
    parser.add_argument("--analysis-dir", type=Path, default=Path("analysis"), help="analysis output directory")
    parser.add_argument("--analysis-suffix", default="", help="suffix after compliance_<response-stem>")
    parser.add_argument("--question-dir", type=Path, default=Path("questions"), help="directory for question files")
    parser.add_argument("--questions", type=Path, help="explicit question file; useful when inference is ambiguous")
    parser.add_argument("--no-question-check", action="store_true", help="skip prompt coverage checks")
    parser.add_argument(
        "--allow-blockers",
        action="store_true",
        help="exit zero even when unresolved commit blockers are present",
    )
    parser.add_argument("--top", type=int, default=30, help="max buckets to print per section")
    parser.add_argument("--examples", type=int, default=8, help="question ID examples per retained bucket")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON instead of text")
    args = parser.parse_args()

    paths = response_paths(args.responses)
    if args.questions is not None and len(paths) > 1:
        print("--questions can only be used with one response file", file=sys.stderr)
        sys.exit(2)

    reports = [
        audit_eval(
            response_path,
            analysis_path_for(response_path, args.analysis_dir, args.analysis_suffix),
            question_path=(
                None
                if args.no_question_check
                else args.questions or infer_question_path(response_path, args.question_dir)
            ),
            examples=args.examples,
        )
        for response_path in paths
    ]

    if args.json:
        serializable = []
        for report in reports:
            converted = {}
            for key, value in report.items():
                if isinstance(value, Counter):
                    converted[key] = dict(value)
                elif isinstance(value, defaultdict):
                    converted[key] = dict(value)
                else:
                    converted[key] = value
            serializable.append(converted)
        print(json.dumps(serializable, ensure_ascii=False, indent=2, sort_keys=True))
        return

    for index, report in enumerate(reports):
        if index:
            print()
        print_eval_report(report, top=args.top)

    if any(report["blocking_issues"] for report in reports) and not args.allow_blockers:
        sys.exit(2)


if __name__ == "__main__":
    main()
