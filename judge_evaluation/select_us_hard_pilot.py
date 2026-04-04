#!/usr/bin/env python3
"""Build a stratified pilot set for rejudging us_hard models."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


LABELS = ("COMPLETE", "DENIAL", "EVASIVE")


@dataclass(frozen=True)
class ModelStats:
    model: str
    creator: str
    response_path: Path
    analysis_path: Path
    total: int
    complete_pct: float
    denial_pct: float
    evasive_pct: float

    @property
    def dominant_label(self) -> str:
        return max(
            LABELS,
            key=lambda label: {
                "COMPLETE": self.complete_pct,
                "DENIAL": self.denial_pct,
                "EVASIVE": self.evasive_pct,
            }[label],
        )

    @property
    def balance_score(self) -> float:
        vals = [self.complete_pct, self.denial_pct, self.evasive_pct]
        return max(vals) - min(vals)

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "creator": self.creator,
            "response_file": str(self.response_path),
            "analysis_file": str(self.analysis_path),
            "total": self.total,
            "complete_pct": round(self.complete_pct * 100.0, 3),
            "denial_pct": round(self.denial_pct * 100.0, 3),
            "evasive_pct": round(self.evasive_pct * 100.0, 3),
            "dominant_label": self.dominant_label,
            "balance_score": round(self.balance_score * 100.0, 3),
        }


def load_metadata(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            row = json.loads(line)
            model = row.get("model_identifier")
            if isinstance(model, str):
                rows[model] = row
    return rows


def load_model_stats(
    analysis_dir: Path,
    responses_dir: Path,
    metadata: dict[str, dict],
    exclude_skip: bool,
) -> list[ModelStats]:
    rows: list[ModelStats] = []
    for analysis_path in sorted(analysis_dir.glob("compliance_us_hard_*.jsonl")):
        response_stem = analysis_path.stem.removeprefix("compliance_")
        response_path = responses_dir / f"{response_stem}.jsonl"
        if not response_path.exists():
            continue

        counts = Counter()
        model_name: str | None = None
        total = 0
        with analysis_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if model_name is None:
                    model_name = obj.get("model")
                compliance = obj.get("compliance")
                if compliance in LABELS:
                    counts[compliance] += 1
                    total += 1
        if not model_name or total == 0:
            continue
        if exclude_skip and metadata.get(model_name, {}).get("skip"):
            continue

        rows.append(
            ModelStats(
                model=model_name,
                creator=model_name.split("/", 1)[0],
                response_path=response_path,
                analysis_path=analysis_path,
                total=total,
                complete_pct=counts["COMPLETE"] / total,
                denial_pct=counts["DENIAL"] / total,
                evasive_pct=counts["EVASIVE"] / total,
            )
        )
    return rows


def pick_diverse(
    candidates: list[ModelStats],
    count: int,
    selected: dict[str, str],
    key_fn: Callable[[ModelStats], tuple],
) -> list[ModelStats]:
    ordered = sorted(candidates, key=key_fn)
    picks: list[ModelStats] = []

    for creator_pass in (True, False):
        if len(picks) >= count:
            break
        for row in ordered:
            if row.model in selected:
                continue
            if row.model in {pick.model for pick in picks}:
                continue
            if creator_pass and any(p.creator == row.creator for p in picks):
                continue
            picks.append(row)
            if len(picks) >= count:
                break
    return picks


def build_plan(rows: list[ModelStats], sample_size: int) -> tuple[list[dict], list[ModelStats]]:
    selected: dict[str, str] = {}
    plan: list[dict] = []

    dominant_complete = [row for row in rows if row.dominant_label == "COMPLETE"]
    dominant_denial = [row for row in rows if row.dominant_label == "DENIAL"]
    dominant_evasive = [row for row in rows if row.dominant_label == "EVASIVE"]
    balanced = sorted(rows, key=lambda row: (row.balance_score, -row.evasive_pct, row.model))

    buckets = [
        (
            "complete_anchors",
            4,
            dominant_complete,
            lambda row: (-row.complete_pct, row.balance_score, row.model),
        ),
        (
            "denial_anchors",
            4,
            dominant_denial,
            lambda row: (-row.denial_pct, row.balance_score, row.model),
        ),
        (
            "evasive_anchors",
            4,
            dominant_evasive,
            lambda row: (-row.evasive_pct, row.balance_score, row.model),
        ),
        (
            "balanced_boundary",
            3,
            balanced,
            lambda row: (row.balance_score, -row.evasive_pct, row.model),
        ),
        (
            "evasive_stress",
            max(sample_size - 15, 0),
            rows,
            lambda row: (-row.evasive_pct, row.balance_score, row.model),
        ),
    ]

    chosen: list[ModelStats] = []
    for bucket_name, target, candidates, key_fn in buckets:
        if target <= 0:
            continue
        picks = pick_diverse(candidates, target, selected, key_fn)
        for pick in picks:
            selected[pick.model] = bucket_name
            chosen.append(pick)
        plan.append(
            {
                "bucket": bucket_name,
                "target": target,
                "selected": [pick.to_dict() for pick in picks],
            }
        )

    if len(chosen) < sample_size:
        remaining = pick_diverse(
            rows,
            sample_size - len(chosen),
            selected,
            lambda row: (-row.evasive_pct, row.balance_score, row.model),
        )
        for pick in remaining:
            selected[pick.model] = "fill"
            chosen.append(pick)
        plan.append(
            {
                "bucket": "fill",
                "target": sample_size - len(chosen) + len(remaining),
                "selected": [pick.to_dict() for pick in remaining],
            }
        )

    chosen = sorted(chosen, key=lambda row: row.model)
    return plan, chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=Path("analysis"))
    parser.add_argument("--responses-dir", type=Path, default=Path("responses"))
    parser.add_argument("--metadata", type=Path, default=Path("../speechmap/model_metadata.json"))
    parser.add_argument("--sample-size", type=int, default=18)
    parser.add_argument("--include-skip", action="store_true", help="include models marked skip in Speechmap metadata")
    parser.add_argument(
        "--output-models",
        type=Path,
        default=Path("judge_evaluation/us_hard_grok_pilot_models.txt"),
        help="output file containing one canonical model id per line",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=Path("judge_evaluation/us_hard_grok_pilot_summary.json"),
        help="JSON summary of the selected pilot set",
    )
    args = parser.parse_args()

    metadata = load_metadata(args.metadata)
    rows = load_model_stats(
        args.analysis_dir,
        args.responses_dir,
        metadata,
        exclude_skip=not args.include_skip,
    )
    if not rows:
        raise SystemExit("no us_hard analysis files found")

    plan, chosen = build_plan(rows, args.sample_size)
    args.output_models.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)

    with args.output_models.open("w", encoding="utf-8") as f:
        for row in chosen:
            f.write(f"{row.model}\n")

    summary = {
        "analysis_dir": str(args.analysis_dir),
        "responses_dir": str(args.responses_dir),
        "sample_size": len(chosen),
        "universe_size": len(rows),
        "plan": plan,
        "models": [row.to_dict() for row in chosen],
    }
    with args.output_summary.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=True, indent=2, sort_keys=True)

    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
