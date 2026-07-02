from datetime import datetime, timezone

from ask import clean_frpe
from compliance.data import JSONLHandler, ModelResponse


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


def test_frpe_keeps_original_moderation_rows_but_retries_other_failures(tmp_path):
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

    assert [row.question_id for row in kept] == ["blocked", "ok"]
    reloaded = JSONLHandler.load_jsonl(responses_path, ModelResponse)
    assert [row.question_id for row in reloaded] == ["blocked", "ok"]
