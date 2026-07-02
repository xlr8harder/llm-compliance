#!/usr/bin/env python3
"""Recode analysis rows stopped by original-provider moderation.

This migrates existing ComplianceAnalysis JSONL artifacts to match the runtime
semantics in ModelResponse.is_original_moderation_error(): if the original
response was stopped by a provider moderation/classifier signal, the row should
be ERROR_ORIGINAL_MODERATION and should not depend on a judge result.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from compliance.data import ModelResponse


ERROR_ORIGINAL_MODERATION = "ERROR_ORIGINAL_MODERATION"


@dataclass
class MigrationStats:
    files_seen: int = 0
    files_changed: int = 0
    rows_seen: int = 0
    rows_changed: int = 0
    changed_by_old_label: Counter[str] = field(default_factory=Counter)
    changed_by_reason: Counter[str] = field(default_factory=Counter)
    changed_by_model: Counter[str] = field(default_factory=Counter)

    def update(self, other: "MigrationStats") -> None:
        self.files_seen += other.files_seen
        self.files_changed += other.files_changed
        self.rows_seen += other.rows_seen
        self.rows_changed += other.rows_changed
        self.changed_by_old_label.update(other.changed_by_old_label)
        self.changed_by_reason.update(other.changed_by_reason)
        self.changed_by_model.update(other.changed_by_model)


def _analysis_paths(inputs: Iterable[Path]) -> list[Path]:
    paths: list[Path] = []
    for path in inputs:
        if path.is_dir():
            paths.extend(sorted(path.glob("compliance_*.jsonl")))
        else:
            paths.append(path)
    return sorted(dict.fromkeys(paths))


def _model_response_from_analysis(row: dict[str, Any]) -> ModelResponse:
    return ModelResponse(
        question_id=str(row.get("question_id") or ""),
        question=str(row.get("question") or ""),
        model=str(row.get("model") or ""),
        timestamp=str(row.get("timestamp") or ""),
        response=row.get("response") if isinstance(row.get("response"), dict) else {},
        api_provider=row.get("original_api_provider"),
        api_model=row.get("api_model"),
        category=row.get("category"),
        domain=row.get("domain"),
    )


def moderation_reason(row: dict[str, Any]) -> str | None:
    """Return original-provider moderation reason for a ComplianceAnalysis row."""
    return _model_response_from_analysis(row).original_moderation_reason()


def recode_row(
    row: dict[str, Any],
    *,
    migrated_at: str,
) -> tuple[dict[str, Any], str | None, str | None]:
    """Return (possibly updated row, reason, old_label)."""
    reason = moderation_reason(row)
    if reason is None:
        return row, None, None
    old_label = row.get("compliance")
    if old_label == ERROR_ORIGINAL_MODERATION:
        return row, reason, None

    updated = dict(row)
    old_analysis = str(updated.get("judge_analysis") or "")
    updated["compliance"] = ERROR_ORIGINAL_MODERATION
    updated["judge_analysis"] = (
        "Post-processed: original response stopped by provider "
        f"moderation/classifier ({reason}); changed compliance from "
        f"{old_label!r} to {ERROR_ORIGINAL_MODERATION}."
    )
    if old_analysis:
        updated["judge_analysis"] += f" Previous judge analysis: {old_analysis}"
    updated["timestamp"] = migrated_at
    updated.pop("raw_judge_response", None)
    return updated, reason, str(old_label)


def migrate_file(path: Path, *, dry_run: bool = False, migrated_at: str | None = None) -> MigrationStats:
    stats = MigrationStats(files_seen=1)
    migrated_at = migrated_at or datetime.now(timezone.utc).isoformat()
    changed_rows: list[dict[str, Any]] = []
    temp_path: Path | None = None

    with path.open("r", encoding="utf-8") as src:
        for line_number, line in enumerate(src, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            stats.rows_seen += 1
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            updated, reason, old_label = recode_row(row, migrated_at=migrated_at)
            if old_label is not None and reason is not None:
                stats.rows_changed += 1
                stats.changed_by_old_label[old_label] += 1
                stats.changed_by_reason[reason] += 1
                stats.changed_by_model[str(row.get("model") or "<missing>")] += 1
            changed_rows.append(updated)

    if stats.rows_changed:
        stats.files_changed = 1
        if not dry_run:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                delete=False,
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
            ) as tmp:
                temp_path = Path(tmp.name)
                for row in changed_rows:
                    tmp.write(json.dumps(row, ensure_ascii=False) + "\n")
            os.replace(temp_path, path)
            temp_path = None

    if temp_path is not None and temp_path.exists():
        temp_path.unlink()
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recode ComplianceAnalysis rows with original-provider moderation stops.",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[Path("analysis")],
        help="Analysis JSONL files or directories containing compliance_*.jsonl files.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report changes without rewriting files.")
    parser.add_argument("--top", type=int, default=20, help="Number of top changed models to print.")
    args = parser.parse_args()

    paths = _analysis_paths(args.paths)
    total = MigrationStats()
    for path in paths:
        file_stats = migrate_file(path, dry_run=args.dry_run)
        total.update(file_stats)

    action = "Would recode" if args.dry_run else "Recoded"
    print(
        f"{action} {total.rows_changed} / {total.rows_seen} rows "
        f"across {total.files_changed} / {total.files_seen} files."
    )
    if total.rows_changed:
        print("By old label:", dict(total.changed_by_old_label))
        print("By moderation reason:", dict(total.changed_by_reason))
        print("Top models:")
        for model, count in total.changed_by_model.most_common(args.top):
            print(f"  {model}: {count}")


if __name__ == "__main__":
    main()
