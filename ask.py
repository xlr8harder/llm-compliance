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
* `--frpe`  – retry non-moderation permanent errors; keep moderation rows.
* `--retry-metadata-errors` – explicitly retry legacy metadata-error rows.
* Optional coherency gate (OpenRouter sub-provider blacklist).
* Provider calls go through llm_client.retry_request.
* ModelCatalog is lazily extended when a full mapping is supplied on the CLI.
* Quiet but informative output: INFO log lines and a tqdm bar.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Dict, Any
import threading
import time
from collections import deque
import hmac
import hashlib
import base64

from tqdm import tqdm  # type: ignore

try:
    import fcntl  # Unix-only file locking
except ImportError:  # pragma: no cover - non-Unix platforms
    fcntl = None

# ---------------------------------------------------------------------------
# compliance layer
# ---------------------------------------------------------------------------
from compliance.data import (
    JSONLHandler,
    ModelResponse,
    Question,
    RESPONSE_STATUS_METADATA_ERROR,
    RESPONSE_STATUS_TRUNCATION,
    RESPONSE_STATUS_UNKNOWN_METADATA,
    UnknownResponseMetadataError,
)
from compliance.models import ModelCatalog
from compliance.utils.llm_requests import request_model_response, resolve_catalog_entry

# ---------------------------------------------------------------------------
# llm_client layer
# ---------------------------------------------------------------------------
import llm_client
from llm_client.testing import run_coherency_tests

LOGGER = logging.getLogger("ask")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

NO_EXTERNAL_API_SENTINEL = Path(".no_external_model_apis")
NEW_RESPONSE_BLOCKED_STATUSES = {
    RESPONSE_STATUS_METADATA_ERROR,
    RESPONSE_STATUS_UNKNOWN_METADATA,
}
DEFAULT_MAX_TRUNCATIONS = 0
TRUNCATION_ABORT_EXIT_CODE = 4
ROUTE_ABORT_EXIT_CODE = 5
DEFAULT_OPENROUTER_EXCLUDED_SUBPROVIDERS = (
    "Azure",
    "Amazon Bedrock",
    "Google",
    "Google AI Studio",
    "Novita",
)
OPENROUTER_ALL_PROVIDERS_IGNORED_MESSAGE = "All providers have been ignored"
OPENROUTER_NO_ENDPOINTS_FOUND_PREFIX = "No endpoints found for "

# Human-readable anonymized model names
ADJECTIVES: List[str] = [
    "brisk", "calm", "clever", "bold", "bright", "brave", "swift", "silent",
    "gentle", "keen", "lively", "mellow", "noble", "quiet", "rapid", "sharp",
    "shy", "smart", "solid", "steady", "still", "strong", "sure", "tender",
    "true", "vivid", "witty", "young", "zesty", "agile", "apt", "blithe",
    "candid", "crisp", "eager", "fierce", "glad", "grand", "humble", "just",
    "light", "mild", "neat", "quick", "spry", "stark", "urban", "vast",
    "warm", "wise", "spruce", "sturdy", "plucky", "sunny", "cosy", "tidy",
    "dapper", "elegant", "fluent", "graceful", "jolly", "nimble", "polished",
    "sincere",
]

NOUNS: List[str] = [
    "sparrow", "falcon", "eagle", "owl", "robin", "raven", "heron", "swan",
    "crane", "stork", "goose", "duck", "wren", "finch", "ibis", "hawk",
    "tiger", "lion", "leopard", "panther", "cougar", "jaguar", "lynx", "puma",
    "wolf", "fox", "bear", "otter", "beaver", "badger", "weasel", "raccoon",
    "moose", "elk", "deer", "antelope", "bison", "buffalo", "camel", "llama",
    "alpaca", "goat", "sheep", "cow", "bull", "horse", "zebra", "donkey",
    "yak", "boar", "pig", "hare", "rabbit", "mouse", "rat", "squirrel",
    "hamster", "mole", "hedgehog", "bat", "dolphin", "whale", "shark", "seal",
    "walrus", "eel", "trout", "salmon", "tuna", "cod", "perch", "carp",
    "pike", "mackerel", "anchovy", "sardine", "octopus", "squid", "crab",
    "lobster", "shrimp", "ant", "bee", "wasp", "beetle", "moth", "butterfly",
    "dragonfly", "spider", "firefly", "termite", "fly", "gnat", "mosquito",
    "locust", "mantis", "cicada", "snake", "python", "cobra", "viper", "gecko",
    "lizard", "iguana", "turtle", "tortoise", "alligator", "crocodile", "newt",
    "salamander", "frog", "toad", "skink", "river", "forest", "mountain",
    "valley", "ocean", "desert", "island", "harbor", "meadow", "garden",
    "canyon", "prairie", "tundra", "glacier", "lagoon", "reef",
]

def _derive_pseudonym(secret: str, model_id: str) -> str:
    """Derive a short, human‑readable, stable pseudonym from secret+model.

    Uses HMAC-SHA256(secret, model_id) as the entropy source, maps to
    adjective-noun plus a 2-char base32 suffix to reduce collisions while
    keeping names short.
    """
    digest = hmac.new(secret.encode("utf-8"), model_id.encode("utf-8"), hashlib.sha256).digest()
    # Three 32-bit chunks for word indices
    a_idx = int.from_bytes(digest[0:4], "big") % len(ADJECTIVES)
    n1_idx = int.from_bytes(digest[4:8], "big") % len(NOUNS)
    n2_idx = int.from_bytes(digest[8:12], "big") % len(NOUNS)
    # Base32 suffix from remaining bytes (letters and digits, lowercased)
    # Using 7 chars → 35 bits; with 3 words (~6 + 7 + 7 bits) total ≈ 55 bits.
    b32 = base64.b32encode(digest[12:]).decode("ascii").lower().rstrip("=")
    suffix = b32[:7]
    return f"{ADJECTIVES[a_idx]}-{NOUNS[n1_idx]}-{NOUNS[n2_idx]}-{suffix}"

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


def ensure_known_response_rows(rows: List[ModelResponse], responses_path: Path) -> None:
    """Fail fast if an existing response row has unclassified metadata."""
    for row in rows:
        try:
            row.ensure_known_response_status()
        except UnknownResponseMetadataError as exc:
            LOGGER.error(
                "%s contains unknown response metadata; classify it before continuing: %s",
                responses_path,
                exc,
            )
            sys.exit(3)


def validate_new_response_metadata(row: ModelResponse) -> None:
    """
    Enforce stricter metadata policy for newly collected rows.

    Existing metadata_error rows are legacy-readable, but new rows with missing
    or ambiguous terminal metadata should be investigated before entering the
    main response corpus.
    """
    row.ensure_known_response_status()
    if row.response_status in NEW_RESPONSE_BLOCKED_STATUSES:
        raise UnknownResponseMetadataError(
            "new response has disallowed terminal metadata "
            f"for question_id={row.question_id!r} model={row.model!r}: "
            f"{row.response_status}: {row.response_status_reason}"
        )


def format_truncation_guard_message(
    *,
    truncation_count: int,
    max_truncations: int,
    max_tokens: Any,
    responses_path: Path,
    in_flight: int,
) -> str:
    return (
        "Truncation guard reached: collected "
        f"{truncation_count} newly truncated response(s) "
        f"(threshold --max-truncations={max_truncations}, current max_tokens={max_tokens}). "
        "This is a cost and data-quality decision point: either rerun with a larger "
        "--max-tokens after choosing the output budget, or pass --allow-truncations "
        "to intentionally keep collecting capped partial answers. "
        f"Stopped submitting new requests; waiting for {in_flight} in-flight request(s) "
        f"to finish. Responses collected so far are in {responses_path}."
    )


def _is_empty_response(row: ModelResponse) -> bool:
    """Return True when the ModelResponse.response payload is empty.

    This specifically targets the observed "response": {} rows which should be
    treated as permanent errors for retry/cleanup purposes.
    """
    try:
        # Empty dict {}, None, or other falsy non-dict should be considered empty
        return not bool(row.response)
    except Exception:  # noqa: BLE001
        return True


def parse_subprovider_args(values: Optional[List[str]]) -> List[str]:
    """Parse repeatable and comma-separated subprovider CLI values."""
    parsed: List[str] = []
    seen: set[str] = set()
    for value in values or []:
        for item in value.split(","):
            name = item.strip()
            if not name:
                continue
            key = name.casefold()
            if key in seen:
                continue
            seen.add(key)
            parsed.append(name)
    return parsed


def merge_subprovider_excludes(
    *,
    default_excludes: tuple[str, ...] = DEFAULT_OPENROUTER_EXCLUDED_SUBPROVIDERS,
    extra_excludes: Optional[List[str]] = None,
    allowed_subproviders: Optional[List[str]] = None,
    failed_subproviders: Optional[List[str]] = None,
) -> List[str]:
    """Build the OpenRouter ignore_list while preserving explicit names."""
    excluded: Dict[str, str] = {}
    for name in list(default_excludes) + parse_subprovider_args(extra_excludes):
        excluded[name.casefold()] = name
    for name in parse_subprovider_args(allowed_subproviders):
        excluded.pop(name.casefold(), None)
    for name in parse_subprovider_args(failed_subproviders):
        excluded.setdefault(name.casefold(), name)
    return list(excluded.values())


def response_subprovider(row: ModelResponse) -> Optional[str]:
    if isinstance(row.response, dict):
        provider = row.response.get("provider")
        if isinstance(provider, str) and provider:
            return provider
    return None


def subprovider_in_list(subprovider: Optional[str], names: List[str]) -> bool:
    if not subprovider:
        return False
    return subprovider.casefold() in {name.casefold() for name in names}


def api_response_error_message(api_response) -> Optional[str]:
    raw_response = getattr(api_response, "raw_provider_response", None)
    if isinstance(raw_response, dict):
        error = raw_response.get("error")
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            return error["message"]
    error_info = getattr(api_response, "error_info", None)
    if isinstance(error_info, dict) and isinstance(error_info.get("message"), str):
        return error_info["message"]
    return None


def openrouter_route_exhausted(api_response) -> bool:
    message = api_response_error_message(api_response)
    return bool(
        message
        and (
            OPENROUTER_ALL_PROVIDERS_IGNORED_MESSAGE in message
            or message.startswith(OPENROUTER_NO_ENDPOINTS_FOUND_PREFIX)
        )
    )


def response_payload_with_client_metadata(api_response) -> Dict[str, Any]:
    """Return provider raw response with a non-destructive llm_client sidecar."""
    raw_response = api_response.raw_provider_response
    if isinstance(raw_response, dict):
        response_payload: Dict[str, Any] = dict(raw_response)
    else:
        response_payload = {"_raw_provider_response": raw_response}

    client_metadata: Dict[str, Any] = {
        "success": bool(getattr(api_response, "success", False)),
        "is_retryable": bool(getattr(api_response, "is_retryable", False)),
    }
    standardized_response = getattr(api_response, "standardized_response", None)
    if isinstance(standardized_response, dict):
        client_metadata["standardized_response"] = standardized_response
    error_info = getattr(api_response, "error_info", None)
    if isinstance(error_info, dict):
        client_metadata["error_info"] = error_info
    if getattr(api_response, "request_format", None) is not None:
        client_metadata["request_format"] = api_response.request_format
    if getattr(api_response, "raw_response_format", None) is not None:
        client_metadata["raw_response_format"] = api_response.raw_response_format

    response_payload["_llm_client"] = client_metadata
    return response_payload


def save_model_responses_in_place(rows: List[ModelResponse], responses_path: Path) -> None:
    """Rewrite a responses file without replacing its inode.

    The producer lock is held on the response file itself. Atomic replacement
    would leave the lock on the old inode, so cleanup rewrites must truncate the
    locked file in place.
    """
    responses_path.parent.mkdir(parents=True, exist_ok=True)
    with responses_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")


def order_model_responses_like_existing(
    rows: List[ModelResponse],
    original_rows: List[ModelResponse],
) -> List[ModelResponse]:
    """Return rows in the pre-cleanup order, preserving new-row relative order."""
    if not original_rows:
        return rows
    current_by_qid = {row.question_id: row for row in rows}
    original_ids = {row.question_id for row in original_rows}
    ordered_rows = [
        current_by_qid.get(original.question_id, original)
        for original in original_rows
        if original.question_id in current_by_qid
    ]
    ordered_rows.extend(row for row in rows if row.question_id not in original_ids)
    return ordered_rows


def save_responses_like_existing_order(
    responses_path: Path,
    original_rows: List[ModelResponse],
) -> None:
    if not original_rows:
        return
    rows = load_model_responses(responses_path)
    ordered_rows = order_model_responses_like_existing(rows, original_rows)
    if [row.question_id for row in ordered_rows] == [row.question_id for row in rows]:
        return
    save_model_responses_in_place(ordered_rows, responses_path)


def restore_removed_rows(
    *,
    responses_path: Path,
    original_rows: List[ModelResponse],
    kept_question_ids: set[str],
) -> int:
    """Restore rows removed for retry when a route-level precondition fails."""
    current_rows = load_model_responses(responses_path)
    current_by_qid = {row.question_id: row for row in current_rows}
    restore_rows = [
        row
        for row in original_rows
        if row.question_id not in kept_question_ids and row.question_id not in current_by_qid
    ]
    if not restore_rows:
        return 0
    original_ids = {row.question_id for row in original_rows}
    ordered_rows: List[ModelResponse] = []
    for original in original_rows:
        ordered_rows.append(current_by_qid.get(original.question_id, original))
    ordered_rows.extend(row for row in current_rows if row.question_id not in original_ids)
    save_model_responses_in_place(ordered_rows, responses_path)
    return len(restore_rows)


def clean_frpe(
    responses_path: Path,
    *,
    retry_frpe: bool = True,
    retry_metadata_errors: bool = False,
    retry_truncations: bool = False,
    retry_moderation_subproviders: Optional[List[str]] = None,
) -> List[ModelResponse]:
    existing_rows = load_model_responses(responses_path)
    ensure_known_response_rows(existing_rows, responses_path)
    kept_rows: List[ModelResponse] = []
    removed_empty = 0
    removed_error = 0
    removed_metadata_error = 0
    removed_truncation = 0
    removed_subprovider_moderation = 0
    removed_duplicate = 0
    kept_moderation = 0
    seen_question_ids: set[str] = set()
    retry_moderation_subproviders = parse_subprovider_args(retry_moderation_subproviders)

    for row in existing_rows:
        if retry_frpe and _is_empty_response(row):
            removed_empty += 1
            continue
        if row.question_id in seen_question_ids:
            removed_duplicate += 1
            continue
        if row.is_original_moderation_error():
            if subprovider_in_list(response_subprovider(row), retry_moderation_subproviders):
                removed_subprovider_moderation += 1
                continue
            seen_question_ids.add(row.question_id)
            kept_rows.append(row)
            kept_moderation += 1
            continue
        if retry_frpe and row.is_frpe_retry_candidate():
            removed_error += 1
            continue
        status, _reason = row.classify_response_status()
        if retry_metadata_errors and status == RESPONSE_STATUS_METADATA_ERROR:
            removed_metadata_error += 1
            continue
        if retry_truncations and status == RESPONSE_STATUS_TRUNCATION:
            removed_truncation += 1
            continue
        seen_question_ids.add(row.question_id)
        kept_rows.append(row)

    removed_total = (
        removed_empty
        + removed_error
        + removed_metadata_error
        + removed_truncation
        + removed_subprovider_moderation
        + removed_duplicate
    )
    if removed_total:
        LOGGER.info(
            (
                "Retry cleanup: removed %d rows (retryable_errors=%d, metadata_errors=%d, "
                "truncations=%d, subprovider_moderation=%d, "
                "empty_response=%d, duplicate_question_id=%d, kept_original_moderation=%d)"
            ),
            removed_total,
            removed_error,
            removed_metadata_error,
            removed_truncation,
            removed_subprovider_moderation,
            removed_empty,
            removed_duplicate,
            kept_moderation,
        )
    save_model_responses_in_place(kept_rows, responses_path)
    return kept_rows


def _print_final_state(responses_path: Path) -> None:
    """Log and print a concise summary of the responses file.

    Outputs total rows, apparent permanent errors, and prints the absolute
    path as the final line for convenient copy/paste into judge_compliance.py.
    """
    rows = load_model_responses(responses_path)
    total = len(rows)
    # Count as apparent errors both schema-detected permanent errors and empty payloads
    errors = sum(1 for r in rows if _is_empty_response(r) or not r.is_success())
    # Print a concise summary line followed by the absolute path
    print(f"SUMMARY total={total} apparent_errors={errors}")
    print(str(responses_path.resolve()))


def acquire_responses_lock(responses_path: Path, *, skip_lock: bool = False):
    """Prevent concurrent writers for the same responses file."""
    if skip_lock:
        LOGGER.warning("Skipping responses file lock for %s", responses_path)
        return None
    if fcntl is None:
        LOGGER.warning("File locking not available; proceeding without a lock")
        return None
    # Lock the responses file itself to avoid creating extra lock files.
    lock_file = responses_path.open("a", encoding="utf-8")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        LOGGER.error("Responses file is locked: %s", responses_path)
        lock_file.close()
        sys.exit(1)
    return lock_file


def detect_metadata(responses_path: Path):
    for row in load_model_responses(responses_path):
        return {
            "provider": row.api_provider,
            "api_model": row.api_model,
            "canonical": row.model,
            "category": row.category,
            "request_format": row.request_format,
        }
    raise RuntimeError(f"{responses_path} has no readable ModelResponse entries")


def apply_reasoning_request_overrides(
    overrides: Dict[str, Any],
    *,
    request_format: Optional[str],
    reasoning: bool,
    no_reasoning: bool,
    reasoning_tokens: Optional[int],
    reasoning_effort: Optional[str],
) -> tuple[Dict[str, Any], Optional[str]]:
    """Apply CLI reasoning switches to catalog request overrides."""
    request_overrides = dict(overrides)
    if not request_format and isinstance(request_overrides.get("request_format"), str):
        request_format = str(request_overrides["request_format"])

    if request_format == "anthropic_messages":
        if reasoning_tokens is not None:
            raise ValueError("--reasoning-tokens is not supported with --request-format anthropic_messages; use --reasoning-effort.")
        if reasoning_effort in {"none", "minimal"}:
            raise ValueError(
                "--reasoning-effort none/minimal is not supported with --request-format anthropic_messages."
            )
        if reasoning_effort is not None and not reasoning:
            raise ValueError("Reasoning options require --reasoning. Specify --reasoning to enable thinking mode.")
        if reasoning:
            thinking_cfg = dict(request_overrides.get("thinking", {}))
            thinking_cfg["type"] = "adaptive"
            request_overrides["thinking"] = thinking_cfg
            output_config = dict(request_overrides.get("output_config", {}))
            output_config["effort"] = reasoning_effort or output_config.get("effort") or "high"
            request_overrides["output_config"] = output_config
        elif no_reasoning:
            request_overrides.pop("thinking", None)
            request_overrides.pop("output_config", None)
        request_overrides.pop("reasoning", None)
    else:
        reasoning_cfg: Dict[str, Any] = dict(request_overrides.get("reasoning", {}))

        if reasoning:
            reasoning_cfg["enabled"] = True
        elif no_reasoning:
            reasoning_cfg["enabled"] = False

        if (reasoning_tokens is not None or reasoning_effort is not None) and not reasoning_cfg.get("enabled"):
            raise ValueError("Reasoning options require --reasoning. Specify --reasoning to enable thinking mode.")
        if reasoning_cfg.get("enabled"):
            if reasoning_tokens is not None:
                reasoning_cfg["max_tokens"] = reasoning_tokens
            if reasoning_effort is not None:
                reasoning_cfg["effort"] = reasoning_effort

        if reasoning_cfg.get("enabled") and ("max_tokens" in reasoning_cfg and "effort" in reasoning_cfg):
            raise ValueError("Reasoning conflict: cannot set both --reasoning-tokens and --reasoning-effort.")

        if "enabled" in reasoning_cfg:
            if not reasoning_cfg["enabled"]:
                reasoning_cfg.pop("max_tokens", None)
                reasoning_cfg.pop("effort", None)
            request_overrides["reasoning"] = reasoning_cfg

    if request_format:
        request_overrides["request_format"] = request_format
    return request_overrides, request_format


def ask_worker(
    question: Question,
    provider_name: str,
    api_model: str,
    ignore_list: Optional[List[str]],
    limiter: Optional[RateLimiter],
    overrides: Optional[Dict[str, Any]] = None,
    system_prompt: Optional[str] = None,
    force_subprovider: Optional[str] = None,
    timeout_seconds: int = 180,
):
    if limiter:
        limiter.acquire()

    return request_model_response(
        provider_name=provider_name,
        api_model=api_model,
        prompt=question.question,
        ignore_list=ignore_list,
        overrides=overrides,
        system_prompt=system_prompt,
        force_subprovider=force_subprovider,
        timeout=timeout_seconds,
        context={"qid": question.id},
    )

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
    parser.add_argument("--max-tokens", type=int, help="Max completion tokens for each answer (unset by default)")
    parser.add_argument(
        "--max-truncations",
        type=int,
        default=DEFAULT_MAX_TRUNCATIONS,
        help=(
            "Abort after this many newly collected truncation responses unless "
            "--allow-truncations is set. 0 aborts on the first new truncation."
        ),
    )
    parser.add_argument(
        "--allow-truncations",
        action="store_true",
        help=(
            "continue collecting responses even when the provider reports a length/max_tokens "
            "truncation. Use only after deciding capped partial answers are acceptable."
        ),
    )
    parser.add_argument("--timeout-seconds", type=int, default=180, help="Per-request timeout in seconds")
    parser.add_argument(
        "--request-format",
        help="llm_client request format, e.g. chat_completions or anthropic_messages",
    )
    parser.add_argument(
        "--frpe",
        action="store_true",
        help="clean non-moderation permanent errors before retrying",
    )
    parser.add_argument(
        "--retry-metadata-errors",
        action="store_true",
        help=(
            "explicitly remove legacy metadata_error rows for retry. "
            "Use only after auditing the metadata shape and current model availability."
        ),
    )
    parser.add_argument(
        "--retry-truncations",
        action="store_true",
        help=(
            "explicitly remove existing length/max_tokens truncation rows before retrying, "
            "typically with a larger --max-tokens value"
        ),
    )
    parser.add_argument(
        "--retry-moderation-subprovider",
        action="append",
        default=[],
        help=(
            "remove terminal moderation rows from a named upstream subprovider before retrying. "
            "May be repeated or comma-separated. Use only for audited provider-route artifacts, "
            "for example Azure or Novita moderation on OpenRouter."
        ),
    )
    parser.add_argument(
        "--anonymize",
        action="store_true",
        help=(
            "Record only a stable, secret-hashed model identifier in outputs. "
            "If the questions filename starts with 'anon', anonymization is enabled by default."
        ),
    )

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
    parser.add_argument(
        "--reasoning-effort",
        choices=["none", "minimal", "low", "medium", "high", "max", "xhigh"],
        help="Reasoning effort level (requires --reasoning)",
    )

    parser.add_argument("--system-prompt", help="System prompt to include in requests")

    parser.add_argument(
        "--force-subprovider",
        help="OpenRouter only: restrict to a single subprovider (uses OpenRouter 'only').",
    )
    parser.add_argument(
        "--exclude-subprovider",
        action="append",
        default=[],
        help=(
            "OpenRouter only: add subprovider(s) to the ignore list. May be repeated or "
            "comma-separated. Known moderated backends are excluded by default."
        ),
    )
    parser.add_argument(
        "--allow-subprovider",
        action="append",
        default=[],
        help=(
            "OpenRouter only: remove subprovider(s) from the default ignore list. May be "
            "repeated or comma-separated. Use --allow-subprovider Azure to permit Azure."
        ),
    )
    parser.add_argument(
        "--skip-lock",
        action="store_true",
        help="do not acquire the responses file writer lock; use only for manual recovery",
    )

    return parser

###############################################################################
# Main entry point
###############################################################################

def main(argv: Optional[List[str]] = None) -> None:  # noqa: D401
    args = build_arg_parser().parse_args(argv)

    if args.max_truncations < 0:
        LOGGER.error("--max-truncations must be >= 0")
        sys.exit(2)

    if NO_EXTERNAL_API_SENTINEL.exists() and os.environ.get("ALLOW_EXTERNAL_MODEL_APIS") != "1":
        LOGGER.error(
            "External model API calls are disabled by %s. "
            "Set ALLOW_EXTERNAL_MODEL_APIS=1 only when API model usage has been explicitly approved.",
            NO_EXTERNAL_API_SENTINEL,
        )
        sys.exit(2)

    # Enable anonymization by default when the questions file starts with 'anon'
    # (normal mode only — detect mode relies on an existing responses file name)
    anonymize = bool(args.anonymize)

    if args.detect:
        meta = detect_metadata(args.detect)
        provider_name = args.provider or meta["provider"]
        api_model = args.model or meta["api_model"]
        canonical_cli = args.canonical_name or meta["canonical"]
        request_format = args.request_format or meta.get("request_format")
        category = meta["category"]
        if not category:
            LOGGER.error("Category missing in responses file – cannot locate questions file")
            sys.exit(1)
        if not (provider_name and api_model):
            LOGGER.error(
                "Cannot infer provider/model from detected file. "
                "If the file was anonymized, supply --provider and --model explicitly."
            )
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
        request_format = args.request_format
        questions_file = args.questions
        # Default-on anonymization if filename starts with 'anon'
        if questions_file.name.startswith("anon"):
            anonymize = True

    catalog = ModelCatalog(args.catalog)
    canonical_name, provider_name, api_model = resolve_catalog_entry(
        catalog, canonical_cli, provider_name, api_model
    )

    # Prepare anonymized naming if requested
    secret_value: Optional[str] = None
    anon_model_name: Optional[str] = None
    if anonymize:
        secret_path = Path(".secret")
        if not secret_path.exists():
            LOGGER.error("--anonymize requires a secret file at .secret")
            sys.exit(1)
        secret_value = secret_path.read_text(encoding="utf-8").strip()
        if not secret_value:
            LOGGER.error(".secret exists but is empty – cannot anonymize")
            sys.exit(1)
        # Derive a readable, stable pseudonym
        anon_model_name = _derive_pseudonym(secret_value, api_model)

    # Compute responses_path now that we know if anonymization applies (normal mode only)
    if not args.detect:
        stem = questions_file.stem
        if args.out:
            responses_path = args.out
        else:
            if anonymize and anon_model_name:
                safe_model = anon_model_name
            else:
                safe_model = (canonical_cli or api_model).replace("/", "_")
            responses_path = Path("responses") / f"{stem}_{safe_model}.jsonl"

    # Ensure output directory exists now that responses_path is finalized
    responses_path.parent.mkdir(parents=True, exist_ok=True)
    _responses_lock = acquire_responses_lock(responses_path, skip_lock=args.skip_lock)

    if args.force_subprovider and provider_name != "openrouter":
        LOGGER.error("--force-subprovider is only supported with provider=openrouter")
        sys.exit(1)
    if (args.exclude_subprovider or args.allow_subprovider) and provider_name != "openrouter":
        LOGGER.error("--exclude-subprovider/--allow-subprovider are only supported with provider=openrouter")
        sys.exit(1)

    catalog_entry = catalog.get_model(canonical_name) if canonical_name else None
    overrides: Dict[str, Any] = dict(catalog_entry.get_request_overrides()) if catalog_entry else {}

    try:
        overrides, request_format = apply_reasoning_request_overrides(
            overrides,
            request_format=request_format,
            reasoning=args.reasoning,
            no_reasoning=getattr(args, "no_reasoning", False),
            reasoning_tokens=args.reasoning_tokens,
            reasoning_effort=args.reasoning_effort,
        )
    except ValueError as exc:
        LOGGER.error(str(exc))
        sys.exit(2)

    if args.max_tokens is not None:
        overrides["max_tokens"] = args.max_tokens

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
    
    # Prepare request_overrides for this run.
    #
    # In normal mode, if neither --reasoning nor --no-reasoning was specified,
    # omit the reasoning block entirely from the API request to avoid confusing
    # models that don't support it.
    #
    # In detect mode, preserve the catalog/default reasoning override so retries
    # resume the same operational mode as the original run.
    request_overrides: Dict[str, Any] = dict(overrides) if overrides else {}
    if not args.detect and not args.reasoning and not getattr(args, "no_reasoning", False):
        request_overrides.pop("reasoning", None)
    if args.max_tokens is not None:
        request_overrides["max_tokens"] = args.max_tokens
    if request_format:
        request_overrides["request_format"] = request_format
    default_ignore_list: Optional[List[str]] = None
    if provider_name == "openrouter" and not args.force_subprovider:
        default_ignore_list = merge_subprovider_excludes(
            extra_excludes=args.exclude_subprovider,
            allowed_subproviders=args.allow_subprovider,
        )
    subprov_info = (
        f"subprov={args.force_subprovider or '<auto>'} "
        f"ignore={default_ignore_list or '<none>'}"
    )
    LOGGER.info(
        "Configuration → model=%s (canon=%s) provider=%s %s questions=%s overrides=%s %s anonymize=%s",
        api_model if not anonymize else "<anon>",
        (anon_model_name or canonical_name or "<none>") if anonymize else (canonical_name or "<none>"),
        provider_name,
        subprov_info,
        questions_file.name,
        request_overrides if request_overrides else "<none>",
        system_prompt_info,
        "on" if anonymize else "off",
    )

    original_rows_before_cleanup: List[ModelResponse] = []
    if (
        args.frpe
        or args.retry_metadata_errors
        or args.retry_truncations
        or args.retry_moderation_subprovider
    ):
        original_rows_before_cleanup = load_model_responses(responses_path)
        kept_rows = clean_frpe(
            responses_path,
            retry_frpe=args.frpe or args.retry_metadata_errors,
            retry_metadata_errors=args.retry_metadata_errors,
            retry_truncations=args.retry_truncations,
            retry_moderation_subproviders=args.retry_moderation_subprovider,
        )
        done_ids = {row.question_id for row in kept_rows}
    else:
        existing_rows = load_model_responses(responses_path)
        ensure_known_response_rows(existing_rows, responses_path)
        done_ids = {row.question_id for row in existing_rows}

    questions = load_questions(questions_file)
    pending_questions = [q for q in questions if q.id not in done_ids]

    LOGGER.info("Questions: total=%d pending=%d", len(questions), len(pending_questions))
    if not pending_questions:
        LOGGER.info("Nothing to do – all questions already answered")
        _print_final_state(responses_path)
        return

    ignore_list: Optional[List[str]] = list(default_ignore_list) if default_ignore_list else None
    if args.coherency:
        LOGGER.info("Running coherency tests …")
        tests_passed, failed_subproviders = run_coherency_tests(
            api_model,
            provider_name,
            openrouter_only=[args.force_subprovider] if args.force_subprovider else None,
            num_workers=args.workers,
            request_overrides=request_overrides if request_overrides else None,
            verbose=True,
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
                ignore_list = merge_subprovider_excludes(
                    extra_excludes=args.exclude_subprovider,
                    allowed_subproviders=args.allow_subprovider,
                    failed_subproviders=failed_subproviders,
                )
                LOGGER.info("OpenRouter: ignoring %s", ", ".join(ignore_list))
    elif ignore_list:
        LOGGER.info("OpenRouter: ignoring %s", ", ".join(ignore_list))

    LOGGER.info("Processing → output=%s", responses_path)

    limiter: Optional[RateLimiter]
    if args.rate_limit > 0:
        limiter = RateLimiter(args.rate_limit, args.rate_period)
        LOGGER.info("Rate limiter active → ≤ %d req / %.0fs", args.rate_limit, args.rate_period)
    else:
        limiter = None

    truncation_count = 0
    truncation_guard_fired = False
    responses_written = 0
    max_tokens_display = request_overrides.get("max_tokens", "<provider default>")

    with ThreadPoolExecutor(max_workers=args.workers) as pool, tqdm(total=len(pending_questions)) as tqdm_bar:
        pending_iter = iter(pending_questions)
        future_map = {}

        def submit_next() -> bool:
            if truncation_guard_fired:
                return False
            try:
                question = next(pending_iter)
            except StopIteration:
                return False
            future = pool.submit(
                ask_worker,
                question,
                provider_name,
                api_model,
                ignore_list,
                limiter,
                request_overrides,
                args.system_prompt,
                args.force_subprovider,
                args.timeout_seconds,
            )
            future_map[future] = question
            return True

        while len(future_map) < args.workers and submit_next():
            pass

        while future_map:
            done_futures, _pending_futures = wait(future_map, return_when=FIRST_COMPLETED)
            # Drain any additional completions that landed before we started processing.
            done_futures = {future for future in future_map if future.done()} | done_futures

            for future in done_futures:
                question = future_map.pop(future)
                tqdm_bar.update(1)
                try:
                    api_response = future.result()
                except Exception as exc:  # noqa: BLE001
                    LOGGER.exception("Worker raised for QID %s: %s", question.id, exc)
                    continue

                if provider_name == "openrouter" and openrouter_route_exhausted(api_response):
                    restored_count = restore_removed_rows(
                        responses_path=responses_path,
                        original_rows=original_rows_before_cleanup,
                        kept_question_ids=done_ids,
                    )
                    LOGGER.error(
                        (
                            "OpenRouter has no acceptable upstream provider for model=%s after "
                            "applying subprovider constraints (force=%s, ignore=%s). "
                            "Aborting without writing a provider-error row for QID %s. "
                            "Choose an explicit policy: relax the route with --allow-subprovider, "
                            "force a known acceptable subprovider, or skip this model. "
                            "Restored %d row(s) removed for this retry attempt."
                        ),
                        api_model,
                        args.force_subprovider or "<none>",
                        ignore_list or "<none>",
                        question.id,
                        restored_count,
                    )
                    sys.exit(ROUTE_ABORT_EXIT_CODE)

                # Defensive logging: sometimes providers return an empty payload ({}).
                # Keep a record but surface a warning so FRPE can cleanly retry later.
                try:
                    if not api_response.raw_provider_response:
                        LOGGER.warning(
                            "Empty provider response for QID %s (model=%s provider=%s) — recorded for FRPE",
                            question.id,
                            api_model,
                            provider_name,
                        )
                except Exception:  # noqa: BLE001
                    pass

                model_response = ModelResponse(
                    question_id=question.id,
                    question=question.question,
                    model=(anon_model_name or (canonical_name or api_model)) if anonymize else (canonical_name or api_model),
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    response=response_payload_with_client_metadata(api_response),
                    api_provider=None if anonymize else provider_name,
                    api_model=None if anonymize else api_model,
                    category=question.category,
                    request_format=getattr(api_response, "request_format", None),
                    raw_response_format=getattr(api_response, "raw_response_format", None),
                    domain=question.domain,
                )
                try:
                    validate_new_response_metadata(model_response)
                except UnknownResponseMetadataError as exc:
                    status = model_response.response_status or RESPONSE_STATUS_UNKNOWN_METADATA
                    quarantine_path = responses_path.with_suffix(
                        f"{responses_path.suffix}.{status}.jsonl"
                    )
                    JSONLHandler.save_jsonl([model_response], quarantine_path, append=True)
                    LOGGER.error(
                        (
                            "Disallowed response metadata for QID %s; quarantined row to %s "
                            "and aborting so the metadata can be classified: %s"
                        ),
                        question.id,
                        quarantine_path,
                        exc,
                    )
                    sys.exit(3)

                JSONLHandler.save_jsonl([model_response], responses_path, append=True)
                responses_written += 1

                if model_response.response_status == RESPONSE_STATUS_TRUNCATION:
                    truncation_count += 1
                    if args.allow_truncations:
                        LOGGER.warning(
                            "Collected truncation response for QID %s; continuing because --allow-truncations is set",
                            question.id,
                        )
                    elif not truncation_guard_fired and truncation_count >= args.max_truncations:
                        truncation_guard_fired = True
                        LOGGER.error(
                            format_truncation_guard_message(
                                truncation_count=truncation_count,
                                max_truncations=args.max_truncations,
                                max_tokens=max_tokens_display,
                                responses_path=responses_path,
                                in_flight=len(future_map),
                            )
                        )

            while len(future_map) < args.workers and submit_next():
                pass

        if truncation_guard_fired:
            LOGGER.error(
                "Run stopped by truncation guard after writing %d new response(s); file now has %d total rows",
                responses_written,
                len(load_model_responses(responses_path)),
            )
            _print_final_state(responses_path)
            sys.exit(TRUNCATION_ABORT_EXIT_CODE)

        LOGGER.info(
            "Completed run → wrote %d new responses (file now has %d total)",
            responses_written,
            len(load_model_responses(responses_path)),
        )
        if responses_written:
            save_responses_like_existing_order(responses_path, original_rows_before_cleanup)

    # Print final state summary and absolute path for convenience
    _print_final_state(responses_path)


if __name__ == "__main__":
    main()
