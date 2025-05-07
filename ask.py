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
from typing import List, Optional

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
# Helper functions
###############################################################################

def load_questions(file_path: Path) -> List[Question]:
    """Read validated ``Question`` rows from *file_path*."""
    questions = JSONLHandler.load_jsonl(file_path, Question)
    if not questions:
        raise ValueError(f"no valid questions in {file_path}")
    return questions


def load_model_responses(file_path: Path) -> List[ModelResponse]:
    """Safe wrapper – returns an empty list when the file is absent."""
    if not file_path.exists():
        return []
    return JSONLHandler.load_jsonl(file_path, ModelResponse)


def clean_frpe(responses_path: Path) -> List[ModelResponse]:
    """Remove permanent‑error rows, rewrite the file, and return the kept rows."""
    existing_rows = load_model_responses(responses_path)
    kept_rows = [row for row in existing_rows if row.is_success()]
    if len(kept_rows) != len(existing_rows):
        LOGGER.info("FRPE: removed %d permanent‑error rows", len(existing_rows) - len(kept_rows))
        JSONLHandler.save_jsonl(kept_rows, responses_path, append=False)
    return kept_rows


def detect_metadata(responses_path: Path):
    """Extract provider / api_model / canonical / category from the first row."""
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
    """Resolve and, if complete information provided, store the mapping."""
    canonical, provider_resolved, api_model_resolved = catalog.resolve_model(
        canonical_name=canonical_name,
        provider=provider,
        provider_model_id=api_model,
    )
    if not provider_resolved or not api_model_resolved:
        raise RuntimeError("Cannot resolve provider / model – please provide explicit flags")
    return canonical or canonical_name, provider_resolved, api_model_resolved


###############################################################################
# Worker function (ThreadPool)
###############################################################################

def ask_worker(
    question: Question,
    provider_name: str,
    api_model: str,
    ignore_list: Optional[List[str]],
):
    provider = llm_client.get_provider(provider_name)
    options: dict[str, object] = {"timeout": 90}
    if ignore_list and provider_name == "openrouter":
        options["ignore_list"] = ignore_list

    response = retry_request(
        provider=provider,
        messages=[{"role": "user", "content": question.question}],
        model_id=api_model,
        max_retries=4,
        context={"qid": question.id},
        **options,
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
    parser.add_argument("--model", help="provider‑specific model id")
    parser.add_argument("--canonical-name", dest="canonical_name", help="canonical model name for logging")

    parser.add_argument("--out", type=Path, help="output JSONL path (normal mode)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--frpe", action="store_true", help="clean permanent errors before retrying")

    parser.add_argument(
        "--no-coherency",
        dest="coherency",
        action="store_false",
        default=True,
        help="skip coherency tests",
    )
    parser.add_argument("--catalog", type=Path, default=Path("model_catalog.jsonl"))

    return parser


###############################################################################
# Main entry point
###############################################################################

def main(argv: Optional[List[str]] = None) -> None:  # noqa: D401
    """Entry‑point wrapper so unit tests can call ``main([])``."""
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
    # ModelCatalog resolution / update
    # ------------------------------------------------------------------
    catalog = ModelCatalog(args.catalog)
    canonical_name, provider_name, api_model = resolve_catalog_entry(
        catalog, canonical_cli, provider_name, api_model
    )
    if args.canonical_name and args.provider and args.model:
        catalog.add_or_update_model(
            canonical_name=args.canonical_name,
            provider=args.provider,
            provider_model_id=args.model,
        )
        catalog.save_catalog()

    # ------------------------------------------------------------------
    # Log final configuration summary (useful in loops)
    # ------------------------------------------------------------------
    LOGGER.info(
        "Configuration → model=%s (canon=%s) provider=%s questions=%s",
        api_model,
        canonical_name or "<none>",
        provider_name,
        questions_file.name,
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

    # ------------------------------------------------------------------
    # ThreadPool execution
    # ------------------------------------------------------------------
    with ThreadPoolExecutor(max_workers=args.workers) as pool, tqdm(total=len(pending_questions)) as tqdm_bar:
        future_map = {
            pool.submit(ask_worker, question, provider_name, api_model, ignore_list): question
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
