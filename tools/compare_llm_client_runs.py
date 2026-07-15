#!/usr/bin/env python3
"""Compare V1 and V2 ask.py JSONL outputs without comparing generated prose."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            question_id = row.get("question_id")
            if not isinstance(question_id, str) or not question_id:
                raise ValueError(f"{path}:{line_number} has no question_id")
            if question_id in rows:
                raise ValueError(f"{path}:{line_number} duplicates {question_id!r}")
            rows[question_id] = row
    return rows


def compare_runs(
    legacy_rows: dict[str, dict[str, Any]],
    v2_rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    legacy_ids = set(legacy_rows)
    v2_ids = set(v2_rows)
    comparisons = []
    for question_id in sorted(legacy_ids & v2_ids):
        legacy = legacy_rows[question_id]
        v2 = v2_rows[question_id]
        legacy_response = legacy.get("response", {})
        v2_response = v2.get("response", {})
        legacy_client = legacy_response.get("_llm_client", {})
        v2_client = v2_response.get("_llm_client", {})
        legacy_standard = legacy_client.get("standardized_response", {})
        v2_standard = v2_client.get("standardized_response", {})
        mismatches = []
        for field in (
            "response_status",
            "response_status_reason",
            "request_format",
            "raw_response_format",
        ):
            if legacy.get(field) != v2.get(field):
                mismatches.append(field)
        for field in ("finish_reason", "native_finish_reason"):
            if legacy_standard.get(field) != v2_standard.get(field):
                mismatches.append(f"standardized_response.{field}")
        if set(legacy_standard) != set(v2_standard):
            mismatches.append("standardized_response.keys")
        if set(legacy_standard.get("usage", {})) != set(v2_standard.get("usage", {})):
            mismatches.append("standardized_response.usage_keys")
        for field in ("success", "is_retryable"):
            if legacy_client.get(field) != v2_client.get(field):
                mismatches.append(f"_llm_client.{field}")

        legacy_raw_keys = set(legacy_response) - {"_llm_client", "_llm_client_v2"}
        v2_raw_keys = set(v2_response) - {"_llm_client", "_llm_client_v2"}
        if legacy_raw_keys != v2_raw_keys:
            mismatches.append("raw_response_keys")
        comparisons.append(
            {
                "question_id": question_id,
                "compatible": not mismatches,
                "mismatches": mismatches,
                "content_equal": legacy_standard.get("content")
                == v2_standard.get("content"),
                "v2_canonical_present": isinstance(
                    v2_response.get("_llm_client_v2"), dict
                ),
            }
        )
    return {
        "legacy_only": sorted(legacy_ids - v2_ids),
        "v2_only": sorted(v2_ids - legacy_ids),
        "rows": comparisons,
        "compatible": not (legacy_ids ^ v2_ids)
        and all(row["compatible"] for row in comparisons),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("legacy", type=Path)
    parser.add_argument("v2", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = compare_runs(load_rows(args.legacy), load_rows(args.v2))
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0 if report["compatible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
