#!/usr/bin/env python3
"""ask.py – collect or resume LLM answers in JSONL format.

Two modes
---------
* **normal**  – supply a *questions* file, provider and model.
* **detect**  – supply an *existing responses* file; the script infers
               provider/model/category, cleans permanent‑error rows (optional)
               and resumes.

Features preserved from the legacy implementation
-------------------------------------------------
* `--frpe`  – “Force‑Retry Permanent Errors” cleanup.
* Optional coherency gate (OpenRouter sub‑provider blacklist).
* Provider calls go through **llm_client.retry_request**.
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
                # drop expired timestamps
                while self._timestamps and now - self._timestamps[0] >= self.period:
                    self._timestamps.popleft()

                if len(self._timestamps) < self.max_calls:
                    self._timestamps.append(now)
                    return  # permission granted – exit

                # how long until the earliest timestamp leaves the window?
                sleep_for = self.period - (now - self._timestamps[0])

            # *outside* the lock while sleeping
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
        LOGGER.info("FRPE: removed %d permanent‑error rows", len(existing_rows) - len(kept_rows))
    JSONLHandler.save_jsonl(kept_rows, responses_path, append=False)
    return kept_rows


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
# Worker function (added *overrides* param)
###############################################################################

def ask_worker(
    question: Question,
    provider_name: str,
    api_model: str,
    ignore_list: Optional[List[str]],
    limiter: Optional[RateLimiter],
    overrides: Optional[Dict[str, Any]] = None,
):
    if limiter:
        limiter.acquire()

    provider = llm_client.get_provider(provider_name)
    options: dict[str, object] = {"timeout": 180}
    if ignore_list and provider_name == "openrouter":
        options["ignore_list"] = ignore_list

    response = retry_request(
        provider=provider,
        messages=[{"role": "user", "content": question.question}],
        model_id=api_model,
        max_retries=4,
        context={"qid": question.id},
        **options,
        **(overrides or {}),  # merge generic request overrides
    )
    return response

###############################################################################
# CLI parsing (only two new flags added)
###############################################################################

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Query an LLM or resume a previous run.")

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--questions", type=Path, help="questions JSONL file (normal mode)")
    mode_group.add_argument("--detect", type=Path, help="existing responses file to resume")

    parser.add_argument("--provider", choices=llm_client.PROVIDER_MAP.keys(), help="API provider")
    parser.add_argument("--model", help="provider‑specific model id")
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

    # NEW flags for override control
    parser.add_argument("--reasoning-tokens", type=int, help="OpenRouter reasoning.max_tokens value")
    parser.add_argument("--reasoning-effort", choices=["low", "medium", "high"], help="OpenRouter effort level")

    return parser

###############################################################################
# Main entry point (override logic only addition)
###############################################################################

def main(argv: Optional[List[str]] = None) -> None:  # noqa: D401
    args = build_arg_parser().parse_args(argv)

    # ------------------------------------------------------------------
    # Determine mode‑specific paths and metadata
    # ------------------------------------------------------------------
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
        responses_path = args.detect  # overwrite same file
    else:  # normal mode
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

    # ------------------------------------------------------------------
    # ModelCatalog resolution / update plus overrides
    # ------------------------------------------------------------------
    catalog = ModelCatalog(args.catalog)
    canonical_name, provider_name, api_model = resolve_catalog_entry(
        catalog, canonical_cli, provider_name, api_model
    )

    catalog_entry = catalog.get_model(canonical_name) if canonical_name else None
    overrides: Dict[str, Any] = dict(catalog_entry.get_request_overrides()) if catalog_entry else {}

    # Merge CLI flags
    if args.reasoning_tokens is not None:
        overrides.setdefault("reasoning", {})["max_tokens"] = args.reasoning_tokens
    if args.reasoning_effort is not None:
        overrides["effort"] = args.reasoning_effort

    # Persist mapping when fully specified on CLI
    if args.canonical_name and args.provider and args.model:
        catalog.add_or_update_model(
            canonical_name=args.canonical_name,
            provider=args.provider,
            provider_model_id=args.model,
            request_overrides=overrides,
        )
        catalog.save_catalog()

    # ------------------------------------------------------------------
    # Log final configuration summary (useful in loops)
    # ------------------------------------------------------------------
    LOGGER.info(
        "Configuration → model=%s (canon=%s) provider=%s questions=%s overrides=%s",
        api_model,
        canonical_name or "<none>",
        provider_name,
        questions_file.name,
        overrides if overrides else "<none>",
    )

    # ------------------------------------------------------------------
    # FRPE cleanup & establish done‑set
    # ------------------------------------------------------------------
    if args.frpe:
        kept_rows = clean_frpe(responses_path)
        done_ids = {row.question_id for row in kept_rows}
    else:
        done_ids = {row.question_id for row in load_model_responses(responses_path)}

    # ------------------------------------------------------------------
    # Load questions and filter pending ones
    # ------------------------------------------------------------------
    questions = load_questions(questions_file)
    pending_questions = [question for question in questions if question.id not in done_ids]

    LOGGER.info("Questions: total=%d pending=%d", len(questions), len(pending_questions))

    if not pending_questions:
        LOGGER.info("Nothing to do – all questions already answered")
        return

    # ------------------------------------------------------------------
    # Optional coherency tests (only when there is work to do)
    # ------------------------------------------------------------------
    ignore_list: Optional[List[str]] = None
    if args.coherency:
        LOGGER.info("Running coherency tests …")
        tests_passed, failed_subproviders = run_coherency_tests(api_model, provider_name)
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

    # ------------------------------------------------------------------
    # ThreadPool execution
    # ------------------------------------------------------------------
    with ThreadPoolExecutor(max_workers=args.workers) as pool, tqdm(total=len(pending_questions)) as tqdm_bar:
        future_map = {
            pool.submit(ask_worker, question, provider_name, api_model, ignore_list, limiter, overrides): question
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


if __name__ == "__main__":
    main()
