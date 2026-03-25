#!/usr/bin/env python3
"""Safely queue `judge_compliance.py` across multiple response files.

This runs at most N response files concurrently, with one `judge_compliance.py`
child process per file. That preserves the existing per-file judging logic while
preventing accidental overlap on the same analysis output.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from judge_compliance import (
    DEFAULT_JUDGE_MODEL,
    DEFAULT_JUDGE_PROVIDER,
    DEFAULT_JUDGE_REASONING_ENABLED,
)
JUDGE_SCRIPT = REPO_ROOT / "judge_compliance.py"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("response_files", nargs="*", type=Path, help="response files to judge")
    parser.add_argument("--response-list", type=Path, help="optional file with one response path per line")
    parser.add_argument("--jobs", type=int, default=4, help="max number of response files to judge concurrently")
    parser.add_argument("--workers", type=int, default=30, help="per-file worker count passed to judge_compliance.py")
    parser.add_argument("--force-restart", action="store_true", help="discard existing analysis file for each file")
    parser.add_argument("--max-errors", type=int, default=5, help="abort a child run after N permanent judge errors")
    parser.add_argument("--no-summary", action="store_true", help="skip child summaries")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL, help="judge model ID")
    parser.add_argument("--judge-provider", default=DEFAULT_JUDGE_PROVIDER, help="judge provider ID")
    parser.add_argument("--prompt-template-file", type=Path, help="optional custom prompt template file")
    parser.add_argument("--reasoning", action="store_true", help="enable reasoning for the judge model")
    parser.add_argument("--no-reasoning", dest="no_reasoning", action="store_true", help="disable reasoning for the judge model")
    parser.add_argument("--reasoning-effort", choices=["low", "medium", "high"], help="reasoning effort level")
    parser.add_argument("--output-dir", type=Path, default=Path("analysis"), help="directory for ComplianceAnalysis output")
    parser.add_argument("--output-stem-suffix", default="", help="suffix appended to compliance_<stem>.jsonl")
    parser.add_argument(
        "--log-dir",
        type=Path,
        help="directory for per-file child logs (default: <output-dir>/judge_queue_logs)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="optional JSON report path for batch results",
    )
    return parser


def load_response_files(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    paths.extend(args.response_files)
    if args.response_list is not None:
        for raw in args.response_list.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            paths.append(Path(line))

    if not paths:
        raise SystemExit("provide response files directly or via --response-list")

    deduped: list[Path] = []
    seen_paths: set[Path] = set()
    seen_outputs: dict[Path, Path] = {}
    for path in paths:
        resolved = path.resolve()
        if resolved in seen_paths:
            continue
        seen_paths.add(resolved)
        analysis_path = (args.output_dir / f"compliance_{path.stem}{args.output_stem_suffix}.jsonl").resolve()
        existing = seen_outputs.get(analysis_path)
        if existing is not None:
            raise SystemExit(
                f"two inputs would target the same output file: {existing} and {path} -> {analysis_path}"
            )
        seen_outputs[analysis_path] = path
        deduped.append(path)
    return deduped


def build_child_command(args: argparse.Namespace, response_file: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(JUDGE_SCRIPT),
        str(response_file),
        "--workers",
        str(args.workers),
        "--max-errors",
        str(args.max_errors),
        "--judge-model",
        args.judge_model,
        "--judge-provider",
        args.judge_provider,
        "--output-dir",
        str(args.output_dir),
    ]
    if args.force_restart:
        cmd.append("--force-restart")
    if args.no_summary:
        cmd.append("--no-summary")
    if args.prompt_template_file is not None:
        cmd.extend(["--prompt-template-file", str(args.prompt_template_file)])
    if args.reasoning:
        cmd.append("--reasoning")
    if args.no_reasoning:
        cmd.append("--no-reasoning")
    if args.reasoning_effort is not None:
        cmd.extend(["--reasoning-effort", args.reasoning_effort])
    if args.output_stem_suffix:
        cmd.extend(["--output-stem-suffix", args.output_stem_suffix])
    return cmd


def run_one(args: argparse.Namespace, response_file: Path, log_dir: Path) -> dict[str, Any]:
    started = time.time()
    log_path = log_dir / f"{response_file.stem}.log"
    cmd = build_child_command(args, response_file)

    print(f"[start] {response_file} -> {log_path}", flush=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"$ {' '.join(cmd)}\n\n")
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

    elapsed = time.time() - started
    result = {
        "response_file": str(response_file),
        "log_path": str(log_path),
        "returncode": proc.returncode,
        "elapsed_seconds": round(elapsed, 3),
    }
    status = "ok" if proc.returncode == 0 else "fail"
    print(f"[{status}] {response_file} rc={proc.returncode} elapsed={elapsed:.1f}s", flush=True)
    return result


def main() -> None:
    args = build_parser().parse_args()
    if args.reasoning and args.no_reasoning:
        raise SystemExit("specify only one of --reasoning or --no-reasoning")
    if (
        args.judge_model == DEFAULT_JUDGE_MODEL
        and args.judge_provider == DEFAULT_JUDGE_PROVIDER
        and not args.reasoning
        and not args.no_reasoning
        and not DEFAULT_JUDGE_REASONING_ENABLED
    ):
        args.no_reasoning = True
    if args.reasoning_effort is not None and not args.reasoning:
        raise SystemExit("--reasoning-effort requires --reasoning")
    if args.jobs < 1:
        raise SystemExit("--jobs must be >= 1")

    response_files = load_response_files(args)
    log_dir = args.log_dir or (args.output_dir / "judge_queue_logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"queueing {len(response_files)} files with jobs={args.jobs}, workers-per-file={args.workers}, "
        f"output-dir={args.output_dir}",
        flush=True,
    )

    results: list[dict[str, Any]] = []
    started = time.time()
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(run_one, args, path, log_dir): path for path in response_files}
        for future in as_completed(futures):
            results.append(future.result())

    results.sort(key=lambda item: item["response_file"])
    total_elapsed = time.time() - started
    failed = [item for item in results if item["returncode"] != 0]

    summary = {
        "jobs": args.jobs,
        "workers_per_file": args.workers,
        "judge_model": args.judge_model,
        "judge_provider": args.judge_provider,
        "output_dir": str(args.output_dir),
        "log_dir": str(log_dir),
        "total_files": len(results),
        "failed_files": len(failed),
        "elapsed_seconds": round(total_elapsed, 3),
        "results": results,
    }

    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(
        f"done: files={len(results)} failed={len(failed)} elapsed={total_elapsed:.1f}s logs={log_dir}",
        flush=True,
    )
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
