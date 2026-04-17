import pytest

from ask import apply_reasoning_request_overrides
from compliance.data import ModelResponse
from compliance.utils.reasoning import summarize_reasoning_payload
from tools.probe_reasoning import build_probe_overrides, recommend_configuration


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
