#!/usr/bin/env python3
"""Audit ModelResponse terminal metadata against the internal whitelist."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter, defaultdict
from dataclasses import fields
from pathlib import Path
from typing import Any, Iterable

from compliance.data import ModelResponse


MODEL_RESPONSE_FIELDS = {field.name for field in fields(ModelResponse)}


def response_paths(inputs: Iterable[Path]) -> list[Path]:
    paths: list[Path] = []
    for path in inputs:
        if path.is_dir():
            paths.extend(sorted(path.glob("*.jsonl")))
        else:
            paths.append(path)
    return sorted(dict.fromkeys(paths))


def model_response_from_row(row: dict[str, Any]) -> ModelResponse:
    data = {key: value for key, value in row.items() if key in MODEL_RESPONSE_FIELDS}
    return ModelResponse(**data)


def finish_metadata(response: Any) -> dict[str, Any]:
    if not isinstance(response, dict):
        return {"shape": type(response).__name__}
    metadata: dict[str, Any] = {"shape": "unknown"}
    choices = response.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        choice = choices[0]
        metadata["shape"] = "chat_choices"
        metadata["finish_reason"] = choice.get("finish_reason")
        metadata["native_finish_reason"] = choice.get("native_finish_reason")
        if choice.get("error") is not None:
            metadata["choice_error_type"] = (
                choice["error"].get("type") if isinstance(choice.get("error"), dict) else type(choice["error"]).__name__
            )
    elif "choices" in response:
        metadata["shape"] = "malformed_choices"
    elif "stop_reason" in response or "content" in response:
        metadata["shape"] = "anthropic_messages"
        metadata["stop_reason"] = response.get("stop_reason")
    if response.get("error") is not None:
        metadata["top_error_type"] = (
            response["error"].get("type") if isinstance(response.get("error"), dict) else type(response["error"]).__name__
        )
    return metadata


def rewrite_with_annotations(path: Path, rows: list[dict[str, Any]]) -> None:
    temp_path: Path | None = None
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        delete=False,
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    ) as tmp:
        temp_path = Path(tmp.name)
        for row in rows:
            tmp.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temp_path, path)
    temp_path = None


def audit_file(
    path: Path,
    *,
    write_annotations: bool,
    include_unknown_annotations: bool,
    example_limit: int,
) -> tuple[Counter[str], Counter[str], Counter[tuple[str, str]], dict[str, Counter[str]], list[dict[str, Any]], int]:
    by_status: Counter[str] = Counter()
    by_reason: Counter[str] = Counter()
    by_status_reason: Counter[tuple[str, str]] = Counter()
    by_model_status: dict[str, Counter[str]] = defaultdict(Counter)
    examples: list[dict[str, Any]] = []
    annotated_rows: list[dict[str, Any]] = []
    rows_seen = 0

    with path.open("r", encoding="utf-8") as src:
        for line_number, line in enumerate(src, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            rows_seen += 1
            row = json.loads(stripped)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")

            model_response = model_response_from_row(row)
            status, reason = model_response.classify_response_status()
            by_status[status] += 1
            by_reason[reason] += 1
            by_status_reason[(status, reason)] += 1
            by_model_status[model_response.model][status] += 1

            if status == "unknown_metadata" and len(examples) < example_limit:
                examples.append(
                    {
                        "path": str(path),
                        "line": line_number,
                        "question_id": model_response.question_id,
                        "model": model_response.model,
                        "status": status,
                        "reason": reason,
                        "metadata": finish_metadata(model_response.response),
                    }
                )

            if write_annotations:
                if status != "unknown_metadata" or include_unknown_annotations:
                    row["response_status"] = status
                    row["response_status_reason"] = reason
                annotated_rows.append(row)

    if write_annotations:
        rewrite_with_annotations(path, annotated_rows)

    return by_status, by_reason, by_status_reason, by_model_status, examples, rows_seen


def print_counter(title: str, counter: Counter[Any], limit: int) -> None:
    print(title)
    for key, count in counter.most_common(limit):
        print(f"  {key}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit response finish/stop metadata against ModelResponse classification.",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[Path("responses")],
        help="Response JSONL files or directories to scan.",
    )
    parser.add_argument("--top", type=int, default=25, help="Rows to print per summary section.")
    parser.add_argument("--examples", type=int, default=20, help="Unknown-metadata examples to print.")
    parser.add_argument(
        "--write-annotations",
        action="store_true",
        help="Rewrite files with response_status and response_status_reason fields.",
    )
    parser.add_argument(
        "--include-unknown-annotations",
        action="store_true",
        help="When rewriting, also write unknown_metadata annotations.",
    )
    args = parser.parse_args()

    total_status: Counter[str] = Counter()
    total_reason: Counter[str] = Counter()
    total_status_reason: Counter[tuple[str, str]] = Counter()
    total_model_status: dict[str, Counter[str]] = defaultdict(Counter)
    examples: list[dict[str, Any]] = []
    rows_seen = 0
    paths = response_paths(args.paths)

    for path in paths:
        by_status, by_reason, by_status_reason, by_model_status, file_examples, file_rows = audit_file(
            path,
            write_annotations=args.write_annotations,
            include_unknown_annotations=args.include_unknown_annotations,
            example_limit=max(0, args.examples - len(examples)),
        )
        rows_seen += file_rows
        total_status.update(by_status)
        total_reason.update(by_reason)
        total_status_reason.update(by_status_reason)
        for model, counts in by_model_status.items():
            total_model_status[model].update(counts)
        examples.extend(file_examples)

    print(f"Scanned {rows_seen} rows across {len(paths)} files.")
    print_counter("By response_status:", total_status, args.top)
    print_counter("By response_status_reason:", total_reason, args.top)
    print_counter("By status/reason:", total_status_reason, args.top)

    print("Models with unknown_metadata:")
    unknown_models = Counter(
        {model: counts["unknown_metadata"] for model, counts in total_model_status.items() if counts["unknown_metadata"]}
    )
    for model, count in unknown_models.most_common(args.top):
        print(f"  {model}: {count}")

    if examples:
        print("Unknown metadata examples:")
        for example in examples[: args.examples]:
            print(json.dumps(example, ensure_ascii=False, sort_keys=True))

    if args.write_annotations:
        print("Wrote response_status annotations.")


if __name__ == "__main__":
    main()
