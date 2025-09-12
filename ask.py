#!/usr/bin/env python3
"""ask.py – collect or resume LLM answers in JSONL format.

Two modes
---------
* normal  – supply a questions file, provider and model.
* detect  – supply an existing responses file; the script infers
            provider/model/category, cleans permanent-error rows (optional)
            and resumes.

Features
--------
* `--frpe`  – "Force-Retry Permanent Errors" cleanup.
* Optional coherency gate (OpenRouter sub-provider blacklist).
* Provider calls go through llm_client.retry_request.
* ModelCatalog is lazily extended when a full mapping is supplied on the CLI.
* Quiet but informative output: INFO log lines and a tqdm bar.
"""
from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Dict, Any
import threading
import time
from collections import deque

from tqdm import tqdm  # type: ignore

# ---------------------------------------------------------------------------
# compliance layer
# ---------------------------------------------------------------------------
from compliance.data import JSONLHandler, ModelResponse, Question
from compliance.models import ModelCatalog

# ---------------------------------------------------------------------------
# llm_client layer
# ---------------------------------------------------------------------------
import llm_client
from llm_client.retry import retry_request
from llm_client.testing import run_coherency_tests

LOGGER = logging.getLogger("ask")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

###############################################################################
# RateLimiter helper
###############################################################################

class RateLimiter:
    """Allow at most *max_calls* every *period* seconds (rolling window)."""

    def __init__(self, max_calls: int, period: float):
        self.max_calls = max_calls
        self.period = period
        self._lock = threading.Lock()
        self._timestamps: deque[float] = deque()    # monotonic times

    def acquire(self) -> None:
        """Block until the caller may proceed."""
        while True:
            with self._lock:
                now = time.monotonic()
                while self._timestamps and now - self._timestamps[0] >= self.period:
                    self._timestamps.popleft()

                if len(self._timestamps) < self.max_calls:
                    self._timestamps.append(now)
                    return

                sleep_for = self.period - (now - self._timestamps[0])

            time.sleep(sleep_for)

###############################################################################
# Helper functions
###############################################################################

def load_questions(file_path: Path) -> List[Question]:
    questions = JSONLHandler.load_jsonl(file_path, Question)
    if not questions:
        raise ValueError(f"no valid questions in {file_path}")
    return questions


def load_model_responses(file_path: Path) -> List[ModelResponse]:
    if not file_path.exists():
        return []
    return JSONLHandler.load_jsonl(file_path, ModelResponse)


def clean_frpe(responses_path: Path) -> List[ModelResponse]:
    existing_rows = load_model_responses(responses_path)
    kept_rows = [row for row in existing_rows if row.is_success()]
    if len(kept_rows) != len(existing_rows):
        LOGGER.info("FRPE: removed %d permanent-error rows", len(existing_rows) - len(kept_rows))
    JSONLHandler.save_jsonl(kept_rows, responses_path, append=False)
    return kept_rows


def _print_final_state(responses_path: Path) -> None:
    """Log and print a concise summary of the responses file.

    Outputs total rows, apparent permanent errors, and prints the absolute
    path as the final line for convenient copy/paste into judge_compliance.py.
    """
    rows = load_model_responses(responses_path)
    total = len(rows)
    errors = sum(1 for r in rows if not r.is_success())
    # Print a concise summary line followed by the absolute path
    print(f"SUMMARY total={total} apparent_errors={errors}")
    print(str(responses_path.resolve()))


def detect_metadata(responses_path: Path):
    for row in load_model_responses(responses_path):
        return {
            "provider": row.api_provider,
            "api_model": row.api_model,
            "canonical": row.model,
            "category": row.category,
        }
    raise RuntimeError(f"{responses_path} has no readable ModelResponse entries")


def resolve_catalog_entry(
    catalog: ModelCatalog,
    canonical_name: Optional[str],
    provider: str,
    api_model: str,
):
    canonical, provider_resolved, api_model_resolved = catalog.resolve_model(
        canonical_name=canonical_name,
        provider=provider,
        provider_model_id=api_model,
    )
    if not provider_resolved or not api_model_resolved:
        raise RuntimeError("Cannot resolve provider / model – please provide explicit flags")
    return canonical or canonical_name, provider_resolved, api_model_resolved

###############################################################################
# Worker function
###############################################################################

def ask_worker(
    question: Question,
    provider_name: str,
    api_model: str,
    ignore_list: Optional[List[str]],
    limiter: Optional[RateLimiter],
    overrides: Optional[Dict[str, Any]] = None,
    system_prompt: Optional[str] = None,
    force_subprovider: Optional[str] = None,
):
    if limiter:
        limiter.acquire()

    provider = llm_client.get_provider(provider_name)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question.question})

    kwargs: Dict[str, Any] = {}
    if overrides:
        kwargs.update(overrides)
    kwargs.setdefault("timeout", 180)

    if provider_name == "openrouter":
        if force_subprovider:
            kwargs["only"] = [force_subprovider]
        elif ignore_list:
            kwargs["ignore_list"] = list(ignore_list)

    response = retry_request(
        provider=provider,
        messages=messages,
        model_id=api_model,
        max_retries=4,
        context={"qid": question.id},
        **kwargs,
    )
    return response

###############################################################################
# CLI parsing
###############################################################################

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Query an LLM or resume a previous run.")

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--questions", type=Path, help="questions JSONL file (normal mode)")
    mode_group.add_argument("--detect", type=Path, help="existing responses file to resume")

    parser.add_argument("--provider", choices=llm_client.PROVIDER_MAP.keys(), help="API provider")
    parser.add_argument("--model", help="provider-specific model id")
    parser.add_argument("--canonical-name", dest="canonical_name", help="canonical model name for logging")

    parser.add_argument("--out", type=Path, help="output JSONL path (normal mode)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--frpe", action="store_true", help="clean permanent errors before retrying")

    parser.add_argument("--rate-limit", type=int, default=0, help="Max requests allowed per period (0 = unlimited)")
    parser.add_argument("--rate-period", type=float, default=60.0, help="Window size in seconds for --rate-limit")

    parser.add_argument(
        "--no-coherency",
        dest="coherency",
        action="store_false",
        default=True,
        help="skip coherency tests",
    )
    parser.add_argument("--catalog", type=Path, default=Path("model_catalog.jsonl"))

    # Reasoning controls (mutually exclusive enable/disable)
    rgroup = parser.add_mutually_exclusive_group()
    rgroup.add_argument(
        "--reasoning",
        action="store_true",
        help="Enable model reasoning (sets reasoning.enabled=true where supported)",
    )
    rgroup.add_argument(
        "--no-reasoning",
        dest="no_reasoning",
        action="store_true",
        help="Disable model reasoning (sets reasoning.enabled=false where supported)",
    )
    parser.add_argument("--reasoning-tokens", type=int, help="Reasoning max_tokens budget (requires --reasoning)")
    parser.add_argument("--reasoning-effort", choices=["low", "medium", "high"], help="Reasoning effort level (requires --reasoning)")

    parser.add_argument("--system-prompt", help="System prompt to include in requests")

    parser.add_argument(
        "--force-subprovider",
        help="OpenRouter only: restrict to a single subprovider (uses OpenRouter 'only').",
    )

    return parser

###############################################################################
# Main entry point
###############################################################################

def main(argv: Optional[List[str]] = None) -> None:  # noqa: D401
    args = build_arg_parser().parse_args(argv)

    if args.detect:
        meta = detect_metadata(args.detect)
        provider_name = args.provider or meta["provider"]
        api_model = args.model or meta["api_model"]
        canonical_cli = args.canonical_name or meta["canonical"]
        category = meta["category"]
        if not category:
            LOGGER.error("Category missing in responses file – cannot locate questions file")
            sys.exit(1)
        questions_file = Path("questions") / f"{category}.jsonl"
        responses_path = args.detect
    else:
        if not (args.questions and args.provider and args.model):
            LOGGER.error("--questions, --provider and --model are required in normal mode")
            sys.exit(1)
        provider_name = args.provider
        api_model = args.model
        canonical_cli = args.canonical_name
        questions_file = args.questions
        stem = questions_file.stem
        safe_model = (canonical_cli or api_model).replace("/", "_")
        responses_path = args.out or Path("responses") / f"{stem}_{safe_model}.jsonl"

    responses_path.parent.mkdir(parents=True, exist_ok=True)

    catalog = ModelCatalog(args.catalog)
    canonical_name, provider_name, api_model = resolve_catalog_entry(
        catalog, canonical_cli, provider_name, api_model
    )

    if args.force_subprovider and provider_name != "openrouter":
        LOGGER.error("--force-subprovider is only supported with provider=openrouter")
        sys.exit(1)

    catalog_entry = catalog.get_model(canonical_name) if canonical_name else None
    overrides: Dict[str, Any] = dict(catalog_entry.get_request_overrides()) if catalog_entry else {}

    # Build unified reasoning payload
    reasoning_cfg: Dict[str, Any] = dict(overrides.get("reasoning", {}))

    # CLI toggles override catalog defaults
    if args.reasoning:
        reasoning_cfg["enabled"] = True
    elif getattr(args, "no_reasoning", False):
        reasoning_cfg["enabled"] = False

    # Validate option usage and apply values only when enabled
    if (args.reasoning_tokens is not None or args.reasoning_effort is not None) and not reasoning_cfg.get("enabled"):
        LOGGER.error("Reasoning options require --reasoning. Specify --reasoning to enable thinking mode.")
        sys.exit(2)
    if reasoning_cfg.get("enabled"):
        if args.reasoning_tokens is not None:
            reasoning_cfg["max_tokens"] = args.reasoning_tokens
        if args.reasoning_effort is not None:
            reasoning_cfg["effort"] = args.reasoning_effort

    # Enforce one-of rule for effort vs max_tokens
    if reasoning_cfg.get("enabled") and ("max_tokens" in reasoning_cfg and "effort" in reasoning_cfg):
        LOGGER.error("Reasoning conflict: cannot set both --reasoning-tokens and --reasoning-effort.")
        sys.exit(2)

    # Only include a reasoning block in overrides when explicitly enabled/disabled
    if "enabled" in reasoning_cfg:
        # If disabled, drop any stray budget keys
        if not reasoning_cfg["enabled"]:
            reasoning_cfg.pop("max_tokens", None)
            reasoning_cfg.pop("effort", None)
        overrides["reasoning"] = reasoning_cfg

    # Persist catalog mapping if fully specified on CLI
    if args.canonical_name and args.provider and args.model:
        catalog.add_or_update_model(
            canonical_name=args.canonical_name,
            provider=args.provider,
            provider_model_id=args.model,
            request_overrides=overrides,
        )
        catalog.save_catalog()

    system_prompt_info = f"system_prompt={len(args.system_prompt)} chars" if args.system_prompt else "system_prompt=none"
    
    # Prepare request_overrides for this run. If neither --reasoning nor --no-reasoning
    # was specified, omit the reasoning block entirely from the API request to avoid
    # confusing models that don't support it.
    request_overrides: Dict[str, Any] = dict(overrides) if overrides else {}
    if not args.reasoning and not getattr(args, "no_reasoning", False):
        request_overrides.pop("reasoning", None)
    subprov_info = f"subprov={args.force_subprovider or '<auto>'}"
    LOGGER.info(
        "Configuration → model=%s (canon=%s) provider=%s %s questions=%s overrides=%s %s",
        api_model,
        canonical_name or "<none>",
        provider_name,
        subprov_info,
        questions_file.name,
        request_overrides if request_overrides else "<none>",
        system_prompt_info,
    )

    if args.frpe:
        kept_rows = clean_frpe(responses_path)
        done_ids = {row.question_id for row in kept_rows}
    else:
        done_ids = {row.question_id for row in load_model_responses(responses_path)}

    questions = load_questions(questions_file)
    pending_questions = [q for q in questions if q.id not in done_ids]

    LOGGER.info("Questions: total=%d pending=%d", len(questions), len(pending_questions))
    if not pending_questions:
        LOGGER.info("Nothing to do – all questions already answered")
        _print_final_state(responses_path)
        return

    ignore_list: Optional[List[str]] = None
    if args.coherency:
        LOGGER.info("Running coherency tests …")
        tests_passed, failed_subproviders = run_coherency_tests(
            api_model,
            provider_name,
            openrouter_only=[args.force_subprovider] if args.force_subprovider else None,
            num_workers=args.workers,
            request_overrides=request_overrides if request_overrides else None,
        )
        if args.force_subprovider:
            if failed_subproviders and args.force_subprovider in failed_subproviders:
                LOGGER.error("Coherency failed for forced subprovider '%s' – aborting", args.force_subprovider)
                sys.exit(1)
        else:
            if not tests_passed:
                LOGGER.error("Coherency tests FAILED – aborting")
                sys.exit(1)
            if failed_subproviders:
                ignore_list = failed_subproviders
                LOGGER.info("OpenRouter: ignoring %s", ", ".join(ignore_list))

    LOGGER.info("Processing → output=%s", responses_path)

    limiter: Optional[RateLimiter]
    if args.rate_limit > 0:
        limiter = RateLimiter(args.rate_limit, args.rate_period)
        LOGGER.info("Rate limiter active → ≤ %d req / %.0fs", args.rate_limit, args.rate_period)
    else:
        limiter = None

    with ThreadPoolExecutor(max_workers=args.workers) as pool, tqdm(total=len(pending_questions)) as tqdm_bar:
        future_map = {
            pool.submit(
                ask_worker,
                question,
                provider_name,
                api_model,
                ignore_list,
                limiter,
                request_overrides,
                args.system_prompt,
                args.force_subprovider,
            ): question
            for question in pending_questions
        }
        for future in as_completed(future_map):
            question = future_map[future]
            tqdm_bar.update(1)
            try:
                api_response = future.result()
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("Worker raised for QID %s: %s", question.id, exc)
                continue

            model_response = ModelResponse(
                question_id=question.id,
                question=question.question,
                model=canonical_name or api_model,
                timestamp=datetime.now(timezone.utc).isoformat(),
                response=api_response.raw_provider_response,
                api_provider=provider_name,
                api_model=api_model,
                category=question.category,
                domain=question.domain,
            )

            JSONLHandler.save_jsonl([model_response], responses_path, append=True)

        LOGGER.info(
            "Completed run → wrote %d new responses (file now has %d total)",
            len(pending_questions),
            len(load_model_responses(responses_path)),
        )

    # Print final state summary and absolute path for convenience
    _print_final_state(responses_path)


if __name__ == "__main__":
    main()
