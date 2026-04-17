"""
Helpers for inspecting whether a model response included reasoning traces.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass
class ReasoningSummary:
    provider: Optional[str] = None
    response_model: Optional[str] = None
    finish_reason: Optional[str] = None
    has_content: bool = False
    content_chars: int = 0
    reasoning_tokens: Optional[int] = None
    has_reasoning_text: bool = False
    has_reasoning_details: bool = False
    reasoning_detail_count: int = 0
    reasoning_present: bool = False
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def summarize_reasoning_payload(raw_response: Optional[Dict[str, Any]]) -> ReasoningSummary:
    summary = ReasoningSummary()

    if not isinstance(raw_response, dict):
        summary.error = "missing_response"
        return summary

    summary.provider = raw_response.get("provider")
    summary.response_model = raw_response.get("model")

    top_level_error = raw_response.get("error")
    if top_level_error:
        summary.error = str(top_level_error)

    choices = raw_response.get("choices")
    choice0 = choices[0] if isinstance(choices, list) and choices else {}
    if not isinstance(choice0, dict):
        choice0 = {}

    summary.finish_reason = choice0.get("finish_reason")
    if choice0.get("error") and not summary.error:
        summary.error = str(choice0.get("error"))

    message = choice0.get("message")
    if not isinstance(message, dict):
        message = {}

    content = message.get("content")
    if isinstance(content, str):
        summary.has_content = content != ""
        summary.content_chars = len(content)
    else:
        summary.has_content = bool(content)

    reasoning_text = message.get("reasoning")
    summary.has_reasoning_text = bool(reasoning_text)

    reasoning_details = message.get("reasoning_details")
    if isinstance(reasoning_details, list):
        summary.reasoning_detail_count = len(reasoning_details)
        summary.has_reasoning_details = len(reasoning_details) > 0
    elif reasoning_details:
        summary.reasoning_detail_count = 1
        summary.has_reasoning_details = True

    # Native Anthropic Messages responses expose content blocks at top level.
    content_blocks = raw_response.get("content")
    if isinstance(content_blocks, list):
        summary.finish_reason = summary.finish_reason or raw_response.get("stop_reason")
        content_chars = 0
        reasoning_detail_count = summary.reasoning_detail_count
        for block in content_blocks:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "text":
                text = block.get("text")
                if isinstance(text, str):
                    content_chars += len(text)
            elif block_type == "thinking":
                thinking = block.get("thinking") or block.get("text")
                if isinstance(thinking, str) and thinking.strip():
                    summary.has_reasoning_text = True
                reasoning_detail_count += 1
            elif block_type == "redacted_thinking":
                if block.get("data") or block.get("redacted_thinking"):
                    summary.has_reasoning_details = True
                reasoning_detail_count += 1
        if content_chars:
            summary.has_content = True
            summary.content_chars = content_chars
        summary.reasoning_detail_count = reasoning_detail_count

    usage = raw_response.get("usage")
    if isinstance(usage, dict):
        completion_details = usage.get("completion_tokens_details")
        if isinstance(completion_details, dict):
            reasoning_tokens = completion_details.get("reasoning_tokens")
            if isinstance(reasoning_tokens, (int, float)):
                summary.reasoning_tokens = int(reasoning_tokens)

    summary.reasoning_present = (
        summary.has_reasoning_text
        or summary.has_reasoning_details
        or (summary.reasoning_tokens not in (None, 0))
    )
    return summary
