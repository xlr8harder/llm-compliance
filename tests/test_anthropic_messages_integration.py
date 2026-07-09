import pytest

from ask import apply_reasoning_request_overrides
from compliance.data import ModelResponse
from compliance.utils.reasoning import summarize_reasoning_payload
from tools.probe_reasoning import (
    PROBE_OUTCOME_REASONING_MANDATORY,
    build_probe_overrides,
    classify_probe_result,
    reasoning_present,
    recommend_configuration,
)


def test_summarize_reasoning_payload_detects_anthropic_messages_blocks():
    raw = {
        "id": "msg_123",
        "model": "anthropic/claude-opus-4.7",
        "content": [
            {"type": "thinking", "thinking": "work through it"},
            {"type": "text", "text": "final answer"},
            {"type": "redacted_thinking", "data": "encrypted_blob"},
        ],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 12, "output_tokens": 34},
    }

    summary = summarize_reasoning_payload(raw)

    assert summary.response_model == "anthropic/claude-opus-4.7"
    assert summary.finish_reason == "end_turn"
    assert summary.has_content is True
    assert summary.content_chars == len("final answer")
    assert summary.has_reasoning_text is True
    assert summary.has_reasoning_details is True
    assert summary.reasoning_detail_count == 2
    assert summary.reasoning_present is True


def test_model_response_accepts_anthropic_messages_text_blocks():
    response = ModelResponse(
        question_id="q1",
        question="Question?",
        model="anthropic/claude-opus-4.7-reasoning",
        timestamp="2026-04-17T00:00:00+00:00",
        response={
            "id": "msg_123",
            "model": "anthropic/claude-opus-4.7",
            "content": [
                {"type": "thinking", "thinking": "work through it"},
                {"type": "text", "text": "final answer"},
                {"type": "redacted_thinking", "data": "encrypted_blob"},
            ],
            "stop_reason": "end_turn",
        },
        request_format="anthropic_messages",
        raw_response_format="openrouter.anthropic_messages",
    )

    assert response.final_content_text() == "final answer"
    assert response.is_success() is True
    assert response.is_permanent_error() is False
    assert response.classify_response_status() == (
        "success",
        "stop_reason:end_turn",
    )


def test_model_response_marks_anthropic_messages_empty_refusal_as_error():
    response = ModelResponse(
        question_id="q1",
        question="Question?",
        model="anthropic/claude-opus-4.7-reasoning",
        timestamp="2026-04-17T00:00:00+00:00",
        response={
            "id": "msg_123",
            "model": "anthropic/claude-opus-4.7",
            "content": [],
            "stop_reason": "refusal",
        },
        request_format="anthropic_messages",
        raw_response_format="openrouter.anthropic_messages",
    )

    assert response.final_content_text() == ""
    assert response.is_success() is False
    assert response.is_permanent_error() is True
    assert response.is_original_moderation_error() is True
    assert response.is_frpe_retry_candidate() is False
    assert response.classify_response_status() == ("moderation", "refusal")


def test_model_response_marks_anthropic_messages_max_tokens_as_error():
    response = ModelResponse(
        question_id="q1",
        question="Question?",
        model="anthropic/claude-opus-4.7-reasoning",
        timestamp="2026-04-17T00:00:00+00:00",
        response={
            "id": "msg_123",
            "model": "anthropic/claude-opus-4.7",
            "content": [
                {"type": "thinking", "thinking": "work through it"},
                {"type": "text", "text": "partial final answer"},
            ],
            "stop_reason": "max_tokens",
        },
        request_format="anthropic_messages",
        raw_response_format="openrouter.anthropic_messages",
    )

    assert response.final_content_text() == "partial final answer"
    assert response.is_success() is False
    assert response.is_permanent_error() is True
    assert response.is_original_moderation_error() is False
    assert response.is_truncation_error() is True
    assert response.is_frpe_retry_candidate() is False
    assert response.classify_response_status() == (
        "truncation",
        "stop_reason:max_tokens",
    )


def test_model_response_marks_chat_content_filter_refusal_as_nonretryable_moderation():
    response = ModelResponse(
        question_id="q1",
        question="Question?",
        model="anthropic/claude-sonnet-5",
        timestamp="2026-04-17T00:00:00+00:00",
        response={
            "choices": [
                {
                    "finish_reason": "content_filter",
                    "native_finish_reason": "refusal",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "refusal": None,
                    },
                }
            ]
        },
        request_format="chat_completions",
        raw_response_format="openrouter.chat_completions",
    )

    assert response.final_content_text() == ""
    assert response.original_moderation_reason() == "content_filter"
    assert response.is_success() is False
    assert response.is_permanent_error() is True
    assert response.is_original_moderation_error() is True
    assert response.is_frpe_retry_candidate() is False
    assert response.classify_response_status() == ("moderation", "content_filter")


def test_model_response_marks_native_refusal_with_partial_text_as_moderation():
    response = ModelResponse(
        question_id="q1",
        question="Question?",
        model="anthropic/claude-opus-4.7",
        timestamp="2026-04-17T00:00:00+00:00",
        response={
            "choices": [
                {
                    "finish_reason": "stop",
                    "native_finish_reason": "refusal",
                    "message": {
                        "role": "assistant",
                        "content": "Partial text returned before the provider classifier stopped it.",
                    },
                }
            ]
        },
        request_format="chat_completions",
        raw_response_format="openrouter.chat_completions",
    )

    assert response.final_content_text().startswith("Partial text")
    assert response.original_moderation_reason() == "refusal"
    assert response.is_success() is False
    assert response.is_original_moderation_error() is True
    assert response.is_frpe_retry_candidate() is False


def test_anthropic_adaptive_probe_uses_messages_request_format():
    overrides = build_probe_overrides({}, "anthropic-adaptive-high")

    assert overrides == {
        "request_format": "anthropic_messages",
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": "high"},
    }


def test_recommendation_includes_messages_request_format_for_adaptive_probe():
    recommendation = recommend_configuration(
        "anthropic/claude-opus-4.7",
        [
            {"probe": "default", "summary": {"reasoning_present": False}},
            {"probe": "reasoning", "summary": {"reasoning_present": False}},
            {"probe": "anthropic-adaptive-high", "summary": {"reasoning_present": True}},
        ],
    )

    assert recommendation["mode"] == "paired_modes"
    assert "--request-format" in recommendation["reasoning_run_flags"]
    assert recommendation["reasoning_request_overrides"]["request_format"] == "anthropic_messages"
    assert recommendation["reasoning_request_overrides"]["thinking"] == {"type": "adaptive"}
    assert recommendation["reasoning_request_overrides"]["output_config"] == {"effort": "high"}

def test_ask_overrides_enable_anthropic_messages_adaptive_high_by_default():
    overrides, request_format = apply_reasoning_request_overrides(
        {"reasoning": {"enabled": True}},
        request_format="anthropic_messages",
        reasoning=True,
        no_reasoning=False,
        reasoning_tokens=None,
        reasoning_effort=None,
    )

    assert request_format == "anthropic_messages"
    assert overrides == {
        "request_format": "anthropic_messages",
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": "high"},
    }


def test_ask_overrides_accept_anthropic_messages_effort_override():
    overrides, _ = apply_reasoning_request_overrides(
        {},
        request_format="anthropic_messages",
        reasoning=True,
        no_reasoning=False,
        reasoning_tokens=None,
        reasoning_effort="medium",
    )

    assert overrides["output_config"] == {"effort": "medium"}


def test_ask_overrides_strip_anthropic_messages_thinking_when_disabled():
    overrides, _ = apply_reasoning_request_overrides(
        {
            "request_format": "anthropic_messages",
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "high"},
        },
        request_format=None,
        reasoning=False,
        no_reasoning=True,
        reasoning_tokens=None,
        reasoning_effort=None,
    )

    assert overrides == {"request_format": "anthropic_messages"}


def test_ask_overrides_reject_anthropic_messages_reasoning_tokens():
    with pytest.raises(ValueError, match="reasoning-tokens"):
        apply_reasoning_request_overrides(
            {},
            request_format="anthropic_messages",
            reasoning=True,
            no_reasoning=False,
            reasoning_tokens=1024,
            reasoning_effort=None,
        )


def test_build_probe_overrides_supports_reasoning_effort_none():
    overrides = build_probe_overrides({}, "reasoning-effort-none")

    assert overrides == {"reasoning": {"effort": "none"}}


def test_recommendation_uses_effort_none_when_enabled_false_does_not_disable_reasoning():
    recommendation = recommend_configuration(
        "x-ai/grok-4.3",
        [
            {"probe": "default", "summary": {"reasoning_present": True}},
            {"probe": "reasoning", "summary": {"reasoning_present": True}},
            {"probe": "no-reasoning", "summary": {"reasoning_present": True}},
            {"probe": "reasoning-effort-none", "summary": {"reasoning_present": False}},
        ],
    )

    assert recommendation["mode"] == "paired_modes"
    assert recommendation["base_run_flags"] == ["--reasoning", "--reasoning-effort", "none"]
    assert recommendation["reasoning_canonical_name"] == "x-ai/grok-4.3-reasoning"


def _openrouter_probe_result(probe, raw_response, request_overrides=None):
    return {
        "probe": probe,
        "request_overrides": request_overrides or {},
        "request_format": "chat_completions",
        "raw_response_format": (
            "openrouter.error" if raw_response.get("error") else "openrouter.chat_completions"
        ),
        "summary": summarize_reasoning_payload(raw_response).to_dict(),
    }


def _grok_45_success_response(*, reasoning_tokens, reasoning_text):
    return {
        "id": "gen-1783538845-BocK8mIzbL4Eaasx3nyY",
        "provider": "xAI",
        "model": "x-ai/grok-4.5-20260708",
        "object": "chat.completion",
        "created": 1783538845,
        "choices": [
            {
                "logprobs": None,
                "finish_reason": "stop",
                "native_finish_reason": "completed",
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "ok",
                    "refusal": None,
                    "reasoning": reasoning_text,
                },
            }
        ],
        "usage": {
            "prompt_tokens": 211,
            "completion_tokens": reasoning_tokens + 1,
            "total_tokens": 212 + reasoning_tokens,
            "cost": 0.000434,
            "is_byok": False,
            "prompt_tokens_details": {
                "cached_tokens": 128,
                "cache_write_tokens": 0,
                "audio_tokens": 0,
                "video_tokens": 0,
            },
            "cost_details": {
                "upstream_inference_cost": 0.000434,
                "upstream_inference_prompt_cost": 0.00023,
                "upstream_inference_completions_cost": 0.000204,
            },
            "completion_tokens_details": {
                "reasoning_tokens": reasoning_tokens,
                "image_tokens": 0,
                "audio_tokens": 0,
            },
        },
    }


def _openrouter_reasoning_mandatory_error():
    return {
        "error": {
            "message": "Reasoning is mandatory for this endpoint and cannot be disabled.",
            "code": 400,
            "metadata": {
                "provider_name": None,
                "previous_errors": [
                    {
                        "code": 400,
                        "message": "Reasoning is mandatory for this endpoint and cannot be disabled.",
                    }
                ],
            },
        },
        "user_id": "user_2kHrsSPEE5bxQH9UuXEB7ICTBno",
    }


def _openrouter_grok_region_error():
    return {
        "error": {
            "message": "Provider returned error",
            "code": 403,
            "metadata": {
                "raw": (
                    '{"code":"permission-denied","error":"The model grok-4.5 '
                    'is not available in your region."}'
                ),
                "provider_name": "xAI",
                "is_byok": False,
                "previous_errors": [
                    {
                        "code": 403,
                        "message": "Provider returned error",
                        "provider_name": "xAI",
                        "raw": (
                            '{"code":"permission-denied","error":"The model grok-4.5 '
                            'is not available in your region."}'
                        ),
                    }
                ],
            },
        },
        "user_id": "user_2kHrsSPEE5bxQH9UuXEB7ICTBno",
    }


def test_recommendation_treats_openrouter_mandatory_reasoning_as_single_mode():
    no_reasoning = _openrouter_probe_result(
        "no-reasoning",
        _openrouter_reasoning_mandatory_error(),
        {"reasoning": {"enabled": False}},
    )

    assert classify_probe_result(no_reasoning) == PROBE_OUTCOME_REASONING_MANDATORY
    assert reasoning_present(no_reasoning) is None

    recommendation = recommend_configuration(
        "x-ai/grok-4.5",
        [
            _openrouter_probe_result(
                "default",
                _grok_45_success_response(
                    reasoning_tokens=4239,
                    reasoning_text="The user asks for exactly ok.",
                ),
            ),
            _openrouter_probe_result(
                "reasoning",
                _grok_45_success_response(
                    reasoning_tokens=4933,
                    reasoning_text="The user asks for exactly ok.",
                ),
                {"reasoning": {"enabled": True}},
            ),
            no_reasoning,
        ],
    )

    assert recommendation["mode"] == "single_mode_reasoning_only"
    assert recommendation["base_run_flags"] == ["--reasoning"]
    assert recommendation["reasoning_canonical_name"] is None
    assert "mandatory-reasoning" in recommendation["notes"]


def test_recommendation_still_uses_effort_none_if_it_successfully_suppresses_reasoning():
    recommendation = recommend_configuration(
        "x-ai/grok-4.5",
        [
            _openrouter_probe_result(
                "default",
                _grok_45_success_response(
                    reasoning_tokens=4239,
                    reasoning_text="The user asks for exactly ok.",
                ),
            ),
            _openrouter_probe_result(
                "reasoning",
                _grok_45_success_response(
                    reasoning_tokens=4933,
                    reasoning_text="The user asks for exactly ok.",
                ),
                {"reasoning": {"enabled": True}},
            ),
            _openrouter_probe_result(
                "no-reasoning",
                _openrouter_reasoning_mandatory_error(),
                {"reasoning": {"enabled": False}},
            ),
            _openrouter_probe_result(
                "reasoning-effort-none",
                _grok_45_success_response(reasoning_tokens=0, reasoning_text=""),
                {"reasoning": {"effort": "none"}},
            ),
        ],
    )

    assert recommendation["mode"] == "paired_modes"
    assert recommendation["base_run_flags"] == ["--reasoning", "--reasoning-effort", "none"]
    assert recommendation["reasoning_canonical_name"] == "x-ai/grok-4.5-reasoning"


def test_recommendation_keeps_provider_errors_indeterminate_not_base_only():
    provider_error_results = [
        _openrouter_probe_result("default", _openrouter_grok_region_error()),
        _openrouter_probe_result(
            "reasoning",
            _openrouter_grok_region_error(),
            {"reasoning": {"enabled": True}},
        ),
    ]

    assert reasoning_present(provider_error_results[0]) is None
    assert classify_probe_result(provider_error_results[0]) == "provider_error"

    recommendation = recommend_configuration("x-ai/grok-4.5", provider_error_results)

    assert recommendation["mode"] == "indeterminate"
    assert recommendation["base_run_flags"] is None


def test_ask_overrides_accept_chat_reasoning_effort_none():
    overrides, request_format = apply_reasoning_request_overrides(
        {},
        request_format=None,
        reasoning=True,
        no_reasoning=False,
        reasoning_tokens=None,
        reasoning_effort="none",
    )

    assert request_format is None
    assert overrides == {"reasoning": {"enabled": True, "effort": "none"}}


def test_ask_overrides_reject_anthropic_messages_reasoning_effort_none():
    with pytest.raises(ValueError, match="none/minimal"):
        apply_reasoning_request_overrides(
            {},
            request_format="anthropic_messages",
            reasoning=True,
            no_reasoning=False,
            reasoning_tokens=None,
            reasoning_effort="none",
        )
