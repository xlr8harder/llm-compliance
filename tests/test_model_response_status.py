from datetime import datetime, timezone

import pytest

from compliance.data import ModelResponse, UnknownResponseMetadataError


def _response(response: dict, *, model: str = "test/model") -> ModelResponse:
    return ModelResponse(
        question_id="q1",
        question="Question?",
        model=model,
        timestamp=datetime.now(timezone.utc).isoformat(),
        response=response,
        api_provider="openrouter",
        api_model=model,
        category="us_hard",
    )


def test_chat_stop_is_whitelisted_success_and_annotated():
    row = _response(
        {
            "choices": [
                {
                    "finish_reason": "stop",
                    "native_finish_reason": "completed",
                    "message": {"role": "assistant", "content": "answer"},
                }
            ]
        }
    )

    row.ensure_known_response_status()

    assert row.is_success() is True
    assert row.response_status == "success"
    assert row.response_status_reason == "finish_reason:stop;native_finish_reason:completed"
    assert row.to_dict()["response_status"] == "success"


def test_native_sensitive_is_terminal_moderation():
    row = _response(
        {
            "choices": [
                {
                    "finish_reason": "stop",
                    "native_finish_reason": "sensitive",
                    "message": {"role": "assistant", "content": "partial"},
                }
            ]
        }
    )

    assert row.classify_response_status() == ("moderation", "sensitive")
    assert row.is_success() is False
    assert row.is_original_moderation_error() is True
    assert row.is_frpe_retry_candidate() is False


def test_length_finish_reason_is_terminal_truncation_not_frpe_retry():
    row = _response(
        {
            "choices": [
                {
                    "finish_reason": "length",
                    "native_finish_reason": "max_output_tokens",
                    "message": {"role": "assistant", "content": "partial"},
                }
            ]
        }
    )

    assert row.classify_response_status() == ("truncation", "finish_reason:length")
    assert row.is_success() is False
    assert row.is_truncation_error() is True
    assert row.is_frpe_retry_candidate() is False


def test_provider_error_remains_frpe_retry_candidate():
    row = _response(
        {
            "error": {
                "type": "server_error",
                "message": "upstream provider failed",
            }
        }
    )

    assert row.classify_response_status() == ("provider_error", "top_error:server_error")
    assert row.is_success() is False
    assert row.is_frpe_retry_candidate() is True


def test_provider_error_code_is_preserved_in_status_reason():
    row = _response(
        {
            "error": {
                "code": 429,
                "message": "rate limit exceeded",
            }
        }
    )

    assert row.classify_response_status() == ("provider_error", "top_error_code:429")
    assert row.is_frpe_retry_candidate() is True


def test_provider_safety_error_text_is_terminal_moderation():
    row = _response(
        {
            "error": {
                "message": (
                    "Provider returned error: Content violates safety guidelines. "
                    "Failed check: SAFETY_CHECK_TYPE_CSAM"
                ),
                "code": 403,
            }
        }
    )

    assert row.classify_response_status() == ("moderation", "moderation_error_text")
    assert row.is_original_moderation_error() is True
    assert row.is_frpe_retry_candidate() is False


def test_llm_client_standardized_success_metadata_is_used_for_direct_provider_shape():
    row = _response(
        {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "direct google answer"}],
                        "role": "model",
                    },
                    "finishReason": "STOP",
                }
            ],
            "_llm_client": {
                "success": True,
                "is_retryable": False,
                "standardized_response": {
                    "content": "direct google answer",
                    "finish_reason": "stop",
                    "native_finish_reason": "STOP",
                    "normalization_evidence": {
                        "finish_reason": {
                            "source": "candidates[0].finishReason",
                            "value": "STOP",
                            "normalized": "stop",
                        }
                    },
                },
            },
        }
    )

    assert row.final_content_text() == "direct google answer"
    assert row.classify_response_status() == (
        "success",
        "_llm_client.standardized_response.finish_reason:stop;native_finish_reason:STOP",
    )


def test_llm_client_recitation_error_info_is_terminal_moderation():
    row = _response(
        {
            "candidates": [
                {
                    "content": {},
                    "finishReason": "RECITATION",
                }
            ],
            "_llm_client": {
                "success": False,
                "is_retryable": False,
                "error_info": {
                    "type": "content_filter",
                    "finish_reason": "content_filter",
                    "native_finish_reason": "RECITATION",
                    "normalization_evidence": {
                        "finish_reason": {
                            "source": "candidates[0].finishReason",
                            "value": "RECITATION",
                            "normalized": "content_filter",
                        }
                    },
                },
            },
        }
    )

    assert row.classify_response_status() == ("moderation", "RECITATION")
    assert row.is_original_moderation_error() is True
    assert row.is_frpe_retry_candidate() is False


def test_success_finish_with_empty_text_is_terminal_suppression():
    row = _response(
        {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": ""},
                }
            ],
            "usage": {"completion_tokens": 1},
        }
    )

    assert row.classify_response_status() == (
        "moderation",
        "empty_stop_content_suppressed",
    )
    assert row.is_original_moderation_error() is True
    assert row.is_success() is False
    assert row.is_frpe_retry_candidate() is False


def test_anthropic_empty_stop_with_hidden_reasoning_is_terminal_suppression():
    row = _response(
        {
            "choices": [
                {
                    "finish_reason": "stop",
                    "native_finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "reasoning": "hidden reasoning without a final answer",
                    },
                }
            ],
            "usage": {"completion_tokens": 81},
        },
        model="anthropic/claude-opus-4.1-thinking",
    )

    assert row.classify_response_status() == (
        "moderation",
        "empty_stop_content_suppressed",
    )
    assert row.is_frpe_retry_candidate() is False


def test_non_anthropic_long_empty_stop_remains_missing_content():
    row = _response(
        {
            "choices": [
                {
                    "finish_reason": "stop",
                    "native_finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "reasoning": "long hidden reasoning without a final answer",
                    },
                }
            ],
            "usage": {"completion_tokens": 1025},
        },
        model="qwen/qwq-32b",
    )

    assert row.classify_response_status() == ("missing_content", "missing_final_content")
    assert row.is_original_moderation_error() is False
    assert row.is_frpe_retry_candidate() is True


def test_other_finish_reason_is_known_metadata_error():
    row = _response(
        {
            "choices": [
                {
                    "finish_reason": "OTHER",
                    "message": {"role": "assistant", "content": "text"},
                }
            ]
        }
    )

    assert row.classify_response_status() == (
        "metadata_error",
        "finish_reason:OTHER",
    )
    row.ensure_known_response_status()
    assert row.is_success() is False
    assert row.is_frpe_retry_candidate() is False


def test_missing_chat_finish_reason_is_known_metadata_error():
    row = _response(
        {
            "choices": [
                {
                    "finish_reason": None,
                    "message": {"role": "assistant", "content": "text"},
                }
            ]
        }
    )

    assert row.classify_response_status() == ("metadata_error", "missing_finish_reason")
    row.ensure_known_response_status()
    assert row.is_success() is False
    assert row.is_frpe_retry_candidate() is False


def test_unrecognized_finish_reason_raises_under_strict_validation():
    row = _response(
        {
            "choices": [
                {
                    "finish_reason": "new_provider_reason",
                    "message": {"role": "assistant", "content": "text"},
                }
            ]
        }
    )

    assert row.classify_response_status() == (
        "unknown_metadata",
        "unknown_finish_reason:new_provider_reason",
    )
    with pytest.raises(UnknownResponseMetadataError):
        row.ensure_known_response_status()
