from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from ask import clean_frpe, response_payload_with_client_metadata, validate_new_response_metadata
from compliance.data import JSONLHandler, ModelResponse, UnknownResponseMetadataError


def _response(question_id: str, response: dict) -> ModelResponse:
    return ModelResponse(
        question_id=question_id,
        question=f"{question_id}?",
        model="test/model",
        timestamp=datetime.now(timezone.utc).isoformat(),
        response=response,
        api_provider="openrouter",
        api_model="test/model",
        category="us_hard",
    )


def test_frpe_keeps_terminal_moderation_and_truncation_but_retries_other_failures(tmp_path):
    responses_path = tmp_path / "responses.jsonl"
    content_filter = _response(
        "blocked",
        {
            "choices": [
                {
                    "finish_reason": "content_filter",
                    "native_finish_reason": "refusal",
                    "message": {"role": "assistant", "content": None},
                }
            ]
        },
    )
    max_tokens = _response(
        "truncated",
        {
            "choices": [
                {
                    "finish_reason": "max_tokens",
                    "message": {"role": "assistant", "content": "partial"},
                }
            ]
        },
    )
    ok = _response(
        "ok",
        {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "answer"},
                }
            ]
        },
    )
    empty_payload = _response("empty", {})
    JSONLHandler.save_jsonl([content_filter, max_tokens, ok, empty_payload], responses_path)

    kept = clean_frpe(responses_path)

    assert [row.question_id for row in kept] == ["blocked", "truncated", "ok"]
    reloaded = JSONLHandler.load_jsonl(responses_path, ModelResponse)
    assert [row.question_id for row in reloaded] == ["blocked", "truncated", "ok"]
    assert [row.response_status for row in reloaded] == [
        "moderation",
        "truncation",
        "success",
    ]


def test_frpe_keeps_legacy_metadata_error_rows_for_manual_review(tmp_path):
    responses_path = tmp_path / "responses.jsonl"
    metadata_error = _response(
        "legacy-null",
        {
            "choices": [
                {
                    "finish_reason": None,
                    "message": {"role": "assistant", "content": "legacy answer"},
                }
            ]
        },
    )
    ok = _response(
        "ok",
        {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "answer"},
                }
            ]
        },
    )
    JSONLHandler.save_jsonl([metadata_error, ok], responses_path)

    kept = clean_frpe(responses_path)

    assert [row.question_id for row in kept] == ["legacy-null", "ok"]
    assert kept[0].response_status == "metadata_error"


def test_new_metadata_error_response_is_blocked_before_main_write():
    row = _response(
        "new-null",
        {
            "choices": [
                {
                    "finish_reason": None,
                    "message": {"role": "assistant", "content": "new answer"},
                }
            ]
        },
    )

    with pytest.raises(UnknownResponseMetadataError):
        validate_new_response_metadata(row)
    assert row.response_status == "metadata_error"


def test_response_payload_preserves_llm_client_metadata_sidecar():
    payload = response_payload_with_client_metadata(
        SimpleNamespace(
            success=False,
            is_retryable=False,
            raw_provider_response={"candidates": [{"finishReason": "RECITATION"}]},
            standardized_response=None,
            error_info={
                "type": "content_filter",
                "finish_reason": "content_filter",
                "native_finish_reason": "RECITATION",
            },
            request_format="chat_completions",
            raw_response_format="google.generate_content",
        )
    )

    assert payload["candidates"][0]["finishReason"] == "RECITATION"
    assert payload["_llm_client"]["error_info"]["native_finish_reason"] == "RECITATION"
    assert payload["_llm_client"]["request_format"] == "chat_completions"
