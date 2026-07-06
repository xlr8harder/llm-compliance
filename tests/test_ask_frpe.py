from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

import ask
from ask import clean_frpe, response_payload_with_client_metadata, validate_new_response_metadata
from compliance.data import JSONLHandler, ModelResponse, Question, UnknownResponseMetadataError


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


def test_responses_lock_blocks_second_writer(tmp_path):
    if ask.fcntl is None:
        pytest.skip("fcntl locking unavailable")
    responses_path = tmp_path / "responses.jsonl"
    first_lock = ask.acquire_responses_lock(responses_path)
    try:
        with pytest.raises(SystemExit) as exc:
            ask.acquire_responses_lock(responses_path)
        assert exc.value.code == 1
    finally:
        first_lock.close()


def test_skip_lock_allows_manual_recovery_even_when_locked(tmp_path):
    if ask.fcntl is None:
        pytest.skip("fcntl locking unavailable")
    responses_path = tmp_path / "responses.jsonl"
    first_lock = ask.acquire_responses_lock(responses_path)
    try:
        assert ask.acquire_responses_lock(responses_path, skip_lock=True) is None
    finally:
        first_lock.close()


def test_frpe_cleanup_preserves_response_file_lock(tmp_path):
    if ask.fcntl is None:
        pytest.skip("fcntl locking unavailable")
    responses_path = tmp_path / "responses.jsonl"
    provider_error = _response("retry", {"error": {"code": 500, "message": "try again"}})
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
    JSONLHandler.save_jsonl([provider_error, ok], responses_path)

    first_lock = ask.acquire_responses_lock(responses_path)
    try:
        clean_frpe(responses_path)
        with pytest.raises(SystemExit):
            ask.acquire_responses_lock(responses_path)
    finally:
        first_lock.close()


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


def test_metadata_error_retry_requires_explicit_flag(tmp_path):
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

    kept = clean_frpe(responses_path, retry_metadata_errors=True)

    assert [row.question_id for row in kept] == ["ok"]
    reloaded = JSONLHandler.load_jsonl(responses_path, ModelResponse)
    assert [row.question_id for row in reloaded] == ["ok"]


def test_retry_truncations_only_removes_truncation_rows(tmp_path):
    responses_path = tmp_path / "responses.jsonl"
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
    provider_error = _response(
        "provider-error",
        {"error": {"type": "server_error", "message": "upstream failed"}},
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
    JSONLHandler.save_jsonl([max_tokens, provider_error, ok], responses_path)

    kept = clean_frpe(
        responses_path,
        retry_frpe=False,
        retry_truncations=True,
    )

    assert [row.question_id for row in kept] == ["provider-error", "ok"]
    reloaded = JSONLHandler.load_jsonl(responses_path, ModelResponse)
    assert [row.question_id for row in reloaded] == ["provider-error", "ok"]


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


def _write_questions(path, count: int) -> None:
    JSONLHandler.save_jsonl(
        [
            Question(
                id=f"q{i}",
                question=f"Question {i}?",
                category="us_hard",
                domain="test",
            )
            for i in range(count)
        ],
        path,
    )


def _truncation_api_response() -> SimpleNamespace:
    return SimpleNamespace(
        success=True,
        is_retryable=False,
        raw_provider_response={
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {"role": "assistant", "content": "partial answer"},
                }
            ]
        },
        standardized_response=None,
        error_info=None,
        request_format="chat_completions",
        raw_response_format="chat_completions",
    )


def test_main_truncation_guard_stops_new_admission_but_drains_in_flight(
    tmp_path,
    monkeypatch,
):
    questions_path = tmp_path / "questions.jsonl"
    responses_path = tmp_path / "responses.jsonl"
    catalog_path = tmp_path / "catalog.jsonl"
    _write_questions(questions_path, 10)
    calls = []

    def fake_request_model_response(**kwargs):
        calls.append(kwargs["context"]["qid"])
        return _truncation_api_response()

    monkeypatch.setenv("ALLOW_EXTERNAL_MODEL_APIS", "1")
    monkeypatch.setattr(ask, "request_model_response", fake_request_model_response)

    with pytest.raises(SystemExit) as exc:
        ask.main(
            [
                "--questions",
                str(questions_path),
                "--provider",
                "openrouter",
                "--model",
                "test/model",
                "--out",
                str(responses_path),
                "--catalog",
                str(catalog_path),
                "--workers",
                "4",
                "--max-truncations",
                "2",
                "--no-coherency",
            ]
        )

    assert exc.value.code == ask.TRUNCATION_ABORT_EXIT_CODE
    assert 2 <= len(calls) < 10
    rows = JSONLHandler.load_jsonl(responses_path, ModelResponse)
    assert len(rows) == len(calls)
    assert all(row.response_status == "truncation" for row in rows)


def test_main_truncation_guard_zero_threshold_stops_after_first_batch(
    tmp_path,
    monkeypatch,
):
    questions_path = tmp_path / "questions.jsonl"
    responses_path = tmp_path / "responses.jsonl"
    catalog_path = tmp_path / "catalog.jsonl"
    _write_questions(questions_path, 10)
    calls = []

    def fake_request_model_response(**kwargs):
        calls.append(kwargs["context"]["qid"])
        return _truncation_api_response()

    monkeypatch.setenv("ALLOW_EXTERNAL_MODEL_APIS", "1")
    monkeypatch.setattr(ask, "request_model_response", fake_request_model_response)

    with pytest.raises(SystemExit) as exc:
        ask.main(
            [
                "--questions",
                str(questions_path),
                "--provider",
                "openrouter",
                "--model",
                "test/model",
                "--out",
                str(responses_path),
                "--catalog",
                str(catalog_path),
                "--workers",
                "2",
                "--no-coherency",
            ]
        )

    assert exc.value.code == ask.TRUNCATION_ABORT_EXIT_CODE
    assert 1 <= len(calls) <= 2
    rows = JSONLHandler.load_jsonl(responses_path, ModelResponse)
    assert len(rows) == len(calls)
    assert all(row.response_status == "truncation" for row in rows)


def test_allow_truncations_continues_collection(tmp_path, monkeypatch):
    questions_path = tmp_path / "questions.jsonl"
    responses_path = tmp_path / "responses.jsonl"
    catalog_path = tmp_path / "catalog.jsonl"
    _write_questions(questions_path, 6)
    calls = []

    def fake_request_model_response(**kwargs):
        calls.append(kwargs["context"]["qid"])
        return _truncation_api_response()

    monkeypatch.setenv("ALLOW_EXTERNAL_MODEL_APIS", "1")
    monkeypatch.setattr(ask, "request_model_response", fake_request_model_response)

    ask.main(
        [
            "--questions",
            str(questions_path),
            "--provider",
            "openrouter",
            "--model",
            "test/model",
            "--out",
            str(responses_path),
            "--catalog",
            str(catalog_path),
            "--workers",
            "4",
            "--max-truncations",
            "2",
            "--allow-truncations",
            "--no-coherency",
        ]
    )

    rows = JSONLHandler.load_jsonl(responses_path, ModelResponse)
    assert len(calls) == 6
    assert len(rows) == 6
    assert all(row.response_status == "truncation" for row in rows)
