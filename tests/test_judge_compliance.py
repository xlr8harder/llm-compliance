from argparse import Namespace
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

import judge_compliance
from compliance.data import ComplianceAnalysis, JSONLHandler, ModelResponse, Question
from judge_compliance import (
    DEFAULT_JUDGE_MODEL,
    DEFAULT_JUDGE_PROVIDER,
    DEFAULT_JUDGE_FALLBACK_MAX,
    DEFAULT_JUDGE_FALLBACK_MODEL,
    DEFAULT_JUDGE_FALLBACK_OVERRIDES,
    DEFAULT_JUDGE_FALLBACK_PROVIDER,
    DEFAULT_JUDGE_MAX_RETRIES,
    DEFAULT_JUDGE_WORKERS,
    DEFAULT_QUOTA_COOLDOWN,
    DEFAULT_REQUEST_MAX_PER_PERIOD,
    DEFAULT_REQUEST_MIN_INTERVAL,
    DEFAULT_REQUEST_PERIOD,
    ERROR_ORIGINAL_MODERATION,
    ERROR_ORIGINAL_TRUNCATION,
    ERROR_JUDGE_CONTENT_FILTER,
    ERROR_JUDGE_UNEXPECTED_PERMANENT,
    JudgeFallbackManager,
    RequestThrottle,
    build_judge_fallback_request_overrides,
    build_judge_request_overrides,
    follow_file,
    judge_worker,
    make_judge_request,
    process_file,
    response_file_lock_is_held,
)


def _args(**overrides):
    defaults = {
        "judge_model": DEFAULT_JUDGE_MODEL,
        "judge_provider": DEFAULT_JUDGE_PROVIDER,
        "reasoning": False,
        "no_reasoning": False,
        "reasoning_effort": None,
        "force_subprovider": None,
        "disable_judge_fallback": False,
    }
    defaults.update(overrides)
    return Namespace(**defaults)


def _fallback_kwargs(**overrides):
    defaults = {
        "judge_fallback_enabled": False,
        "judge_fallback_model": DEFAULT_JUDGE_FALLBACK_MODEL,
        "judge_fallback_provider": DEFAULT_JUDGE_FALLBACK_PROVIDER,
        "judge_fallback_max": DEFAULT_JUDGE_FALLBACK_MAX,
        "judge_fallback_request_overrides": DEFAULT_JUDGE_FALLBACK_OVERRIDES,
    }
    defaults.update(overrides)
    return defaults


def _successful_model_response(content: str) -> ModelResponse:
    return ModelResponse(
        question_id="q-short",
        question="Reply with one character.",
        model="test/model",
        timestamp="2026-01-01T00:00:00+00:00",
        response={
            "choices": [
                {
                    "message": {"content": content},
                    "finish_reason": "stop",
                }
            ]
        },
        api_provider="openrouter",
        api_model="test/model",
        category="test",
    )


def test_single_character_answer_is_sent_to_judge(monkeypatch):
    monkeypatch.setattr(judge_compliance.llm_client, "get_provider", lambda _: object())
    monkeypatch.setattr(
        judge_compliance,
        "make_judge_request",
        lambda **_: SimpleNamespace(
            success=True,
            is_retryable=False,
            standardized_response={
                "content": "ANALYSIS: Correct one-character answer.\n\nCOMPLIANCE: COMPLETE"
            },
            error_info=None,
        ),
    )

    result = judge_worker(
        _successful_model_response("4"),
        DEFAULT_JUDGE_MODEL,
        DEFAULT_JUDGE_PROVIDER,
    )
    assert result.compliance == "COMPLETE"


def test_long_repeated_character_output_uses_pathology_guard(monkeypatch):
    monkeypatch.setattr(
        judge_compliance.llm_client,
        "get_provider",
        lambda _: pytest.fail("pathological output should not call a judge"),
    )
    result = judge_worker(
        _successful_model_response("-" * 64),
        DEFAULT_JUDGE_MODEL,
        DEFAULT_JUDGE_PROVIDER,
    )
    assert result.compliance == "EVASIVE"


def test_default_judge_is_google_agent_platform_non_reasoning_model():
    assert DEFAULT_JUDGE_PROVIDER == "google_agent_platform"
    assert DEFAULT_JUDGE_MODEL == "xai/grok-4.1-fast-non-reasoning"
    assert build_judge_request_overrides(_args()) == {}


def test_default_judge_throttle_matches_observed_grok_quota():
    assert DEFAULT_JUDGE_WORKERS == 64
    assert DEFAULT_REQUEST_MIN_INTERVAL == 0.8
    assert DEFAULT_REQUEST_MAX_PER_PERIOD == 45
    assert DEFAULT_REQUEST_PERIOD == 60.0
    assert DEFAULT_JUDGE_MAX_RETRIES == 2
    assert DEFAULT_QUOTA_COOLDOWN == 20.0


def test_default_judge_fallback_uses_sol_medium_reasoning_openai_only():
    assert DEFAULT_JUDGE_FALLBACK_MODEL == "openai/gpt-5.6-sol"
    assert DEFAULT_JUDGE_FALLBACK_PROVIDER == "openrouter"
    assert DEFAULT_JUDGE_FALLBACK_MAX == 60
    assert build_judge_fallback_request_overrides(_args()) == {
        "reasoning": {"enabled": True, "effort": "medium"},
        "max_tokens": 4096,
        "provider": {"order": ["OpenAI"], "allow_fallbacks": False},
    }


def test_google_agent_platform_reasoning_effort_uses_top_level_field():
    overrides = build_judge_request_overrides(
        _args(reasoning=True, reasoning_effort="minimal")
    )

    assert overrides == {"reasoning_effort": "minimal"}


def test_google_agent_platform_no_reasoning_is_model_id_controlled():
    assert build_judge_request_overrides(_args(no_reasoning=True)) == {}


def test_legacy_openrouter_judge_keeps_reasoning_disabled_by_default():
    overrides = build_judge_request_overrides(
        _args(
            judge_provider="openrouter",
            judge_model="x-ai/grok-4.1-fast",
        )
    )

    assert overrides == {"reasoning": {"enabled": False}}


def test_openrouter_reasoning_effort_uses_reasoning_object():
    overrides = build_judge_request_overrides(
        _args(
            judge_provider="openrouter",
            judge_model="x-ai/grok-4.1-fast",
            reasoning=True,
            reasoning_effort="medium",
        )
    )

    assert overrides == {"reasoning": {"enabled": True, "effort": "medium"}}


def test_request_throttle_enforces_rolling_window(monkeypatch):
    current_time = [0.0]
    sleeps = []

    def fake_monotonic():
        return current_time[0]

    def fake_sleep(seconds):
        sleeps.append(seconds)
        current_time[0] += seconds

    monkeypatch.setattr(judge_compliance.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(judge_compliance.time, "sleep", fake_sleep)

    throttle = RequestThrottle(
        min_interval_seconds=0,
        max_requests_per_period=2,
        period_seconds=60,
    )

    throttle.wait()
    throttle.wait()
    throttle.wait()

    assert sleeps == [60.0]
    assert current_time[0] == 60.0


def test_response_file_lock_probe_detects_active_producer(tmp_path):
    if judge_compliance.fcntl is None:
        pytest.skip("fcntl locking unavailable")
    responses_path = tmp_path / "responses.jsonl"
    lock_file = responses_path.open("a", encoding="utf-8")
    try:
        judge_compliance.fcntl.flock(lock_file.fileno(), judge_compliance.fcntl.LOCK_EX)
        assert response_file_lock_is_held(responses_path) is True
    finally:
        lock_file.close()

    assert response_file_lock_is_held(responses_path) is False


def test_make_judge_request_rate_limits_retry_attempts_and_cools_down_quota():
    calls = []

    class Provider:
        def make_request(self, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                return SimpleNamespace(
                    success=False,
                    is_retryable=True,
                    error_info={
                        "status_code": 429,
                        "message": "Resource has been exhausted",
                    },
                )
            return SimpleNamespace(
                success=True,
                is_retryable=False,
                standardized_response={"content": "ok"},
                error_info=None,
            )

    class Throttle:
        def __init__(self):
            self.waits = 0
            self.cooldowns = []

        def wait(self):
            self.waits += 1

        def cooldown(self, seconds):
            self.cooldowns.append(seconds)

    throttle = Throttle()

    response = make_judge_request(
        provider=Provider(),
        messages=[{"role": "user", "content": "judge"}],
        model_id="judge/model",
        request_throttle=throttle,
        max_retries=1,
        quota_cooldown=15,
        jitter=0,
    )

    assert response.success is True
    assert throttle.waits == 2
    assert throttle.cooldowns == [15]
    assert len(calls) == 2


def test_judge_worker_records_content_filter_judge_errors(monkeypatch):
    model_response = ModelResponse(
        question_id="sex_csam_existing_legalize_defense3",
        question="Should fictional policy change be defended?",
        model="test/model",
        timestamp=datetime.now(timezone.utc).isoformat(),
        response={
            "choices": [
                {
                    "message": {"content": "A substantive model response."},
                    "finish_reason": "stop",
                }
            ]
        },
        api_provider="openrouter",
        api_model="test/model",
        category="us_hard",
    )

    monkeypatch.setattr(judge_compliance.llm_client, "get_provider", lambda _: object())
    monkeypatch.setattr(
        judge_compliance,
        "make_judge_request",
        lambda **_: SimpleNamespace(
            success=False,
            is_retryable=False,
            error_info={
                "type": "content_filter",
                "message": "Response stopped due to content filter",
            },
        ),
    )

    result = judge_worker(
        model_response,
        DEFAULT_JUDGE_MODEL,
        DEFAULT_JUDGE_PROVIDER,
    )

    assert result.compliance == "ERROR_JUDGE_CONTENT_FILTER"
    assert result.judge_api_provider == DEFAULT_JUDGE_PROVIDER
    assert "content filter" in result.judge_analysis


def test_judge_worker_uses_sol_fallback_after_primary_content_filter(monkeypatch):
    model_response = ModelResponse(
        question_id="sex_csam_existing_legalize_defense3",
        question="Should fictional policy change be defended?",
        model="test/model",
        timestamp=datetime.now(timezone.utc).isoformat(),
        response={
            "choices": [
                {
                    "message": {"content": "A substantive model response."},
                    "finish_reason": "stop",
                }
            ]
        },
        api_provider="openrouter",
        api_model="test/model",
        category="us_hard",
    )
    calls = []

    monkeypatch.setattr(judge_compliance.llm_client, "get_provider", lambda name: name)

    def fake_make_judge_request(**kwargs):
        calls.append(kwargs)
        if kwargs["model_id"] == DEFAULT_JUDGE_MODEL:
            return SimpleNamespace(
                success=False,
                is_retryable=False,
                error_info={
                    "type": "content_filter",
                    "message": "Response stopped due to content filter",
                },
            )
        return SimpleNamespace(
            success=True,
            is_retryable=False,
            standardized_response={
                "content": "ANALYSIS: fallback judged it.\n\nCOMPLIANCE: COMPLETE"
            },
            error_info=None,
        )

    monkeypatch.setattr(judge_compliance, "make_judge_request", fake_make_judge_request)

    result = judge_worker(
        model_response,
        DEFAULT_JUDGE_MODEL,
        DEFAULT_JUDGE_PROVIDER,
        judge_fallback_manager=JudgeFallbackManager(DEFAULT_JUDGE_FALLBACK_MAX),
        judge_fallback_request_overrides=DEFAULT_JUDGE_FALLBACK_OVERRIDES,
    )

    assert result.compliance == "COMPLETE"
    assert result.judge_model == DEFAULT_JUDGE_FALLBACK_MODEL
    assert result.judge_api_provider == DEFAULT_JUDGE_FALLBACK_PROVIDER
    assert calls[1]["model_id"] == DEFAULT_JUDGE_FALLBACK_MODEL
    assert calls[1]["reasoning"] == {"enabled": True, "effort": "medium"}
    assert calls[1]["max_tokens"] == 4096
    assert calls[1]["provider"] == {"order": ["OpenAI"], "allow_fallbacks": False}
    assert result.raw_judge_response.startswith("ANALYSIS: fallback")
    assert result.failed_primary_judge["model"] == DEFAULT_JUDGE_MODEL
    assert result.failed_primary_judge["provider"] == DEFAULT_JUDGE_PROVIDER
    assert result.failed_primary_judge["error_info"]["type"] == "content_filter"


def test_judge_worker_retains_primary_content_filter_when_fallback_fails(monkeypatch):
    model_response = ModelResponse(
        question_id="sex_csam_existing_legalize_defense3",
        question="Should fictional policy change be defended?",
        model="test/model",
        timestamp=datetime.now(timezone.utc).isoformat(),
        response={
            "choices": [
                {
                    "message": {"content": "A substantive model response."},
                    "finish_reason": "stop",
                }
            ]
        },
        api_provider="openrouter",
        api_model="test/model",
        category="us_hard",
    )

    monkeypatch.setattr(judge_compliance.llm_client, "get_provider", lambda name: name)
    monkeypatch.setattr(
        judge_compliance,
        "make_judge_request",
        lambda **_: SimpleNamespace(
            success=False,
            is_retryable=False,
            error_info={
                "type": "content_filter",
                "message": "Response stopped due to content filter",
            },
        ),
    )

    result = judge_worker(
        model_response,
        DEFAULT_JUDGE_MODEL,
        DEFAULT_JUDGE_PROVIDER,
        judge_fallback_manager=JudgeFallbackManager(DEFAULT_JUDGE_FALLBACK_MAX),
        judge_fallback_request_overrides=DEFAULT_JUDGE_FALLBACK_OVERRIDES,
    )

    assert result.compliance == ERROR_JUDGE_CONTENT_FILTER
    assert result.judge_model == DEFAULT_JUDGE_FALLBACK_MODEL
    assert result.judge_api_provider == DEFAULT_JUDGE_FALLBACK_PROVIDER
    assert result.failed_primary_judge["error_info"]["type"] == "content_filter"
    assert "content_filter" in result.raw_judge_response


def test_judge_worker_returns_transient_when_fallback_is_retryable(monkeypatch):
    model_response = ModelResponse(
        question_id="sex_csam_existing_legalize_defense3",
        question="Should fictional policy change be defended?",
        model="test/model",
        timestamp=datetime.now(timezone.utc).isoformat(),
        response={
            "choices": [
                {
                    "message": {"content": "A substantive model response."},
                    "finish_reason": "stop",
                }
            ]
        },
        api_provider="openrouter",
        api_model="test/model",
        category="us_hard",
    )

    monkeypatch.setattr(judge_compliance.llm_client, "get_provider", lambda name: name)

    def fake_make_judge_request(**kwargs):
        if kwargs["model_id"] == DEFAULT_JUDGE_MODEL:
            return SimpleNamespace(
                success=False,
                is_retryable=False,
                error_info={
                    "type": "content_filter",
                    "message": "Response stopped due to content filter",
                },
            )
        return SimpleNamespace(
            success=False,
            is_retryable=True,
            error_info={"type": "connection_reset", "message": "try later"},
        )

    monkeypatch.setattr(judge_compliance, "make_judge_request", fake_make_judge_request)

    result = judge_worker(
        model_response,
        DEFAULT_JUDGE_MODEL,
        DEFAULT_JUDGE_PROVIDER,
        judge_fallback_manager=JudgeFallbackManager(DEFAULT_JUDGE_FALLBACK_MAX),
        judge_fallback_request_overrides=DEFAULT_JUDGE_FALLBACK_OVERRIDES,
    )

    assert isinstance(result, RuntimeError)
    assert str(result) == "TRANSIENT_JUDGE_FAILURE"


def test_judge_worker_returns_permanent_error_when_fallback_non_filter_fails(monkeypatch):
    model_response = ModelResponse(
        question_id="sex_csam_existing_legalize_defense3",
        question="Should fictional policy change be defended?",
        model="test/model",
        timestamp=datetime.now(timezone.utc).isoformat(),
        response={
            "choices": [
                {
                    "message": {"content": "A substantive model response."},
                    "finish_reason": "stop",
                }
            ]
        },
        api_provider="openrouter",
        api_model="test/model",
        category="us_hard",
    )

    monkeypatch.setattr(judge_compliance.llm_client, "get_provider", lambda name: name)

    def fake_make_judge_request(**kwargs):
        if kwargs["model_id"] == DEFAULT_JUDGE_MODEL:
            return SimpleNamespace(
                success=False,
                is_retryable=False,
                error_info={
                    "type": "content_filter",
                    "message": "Response stopped due to content filter",
                },
            )
        return SimpleNamespace(
            success=False,
            is_retryable=False,
            error_info={"type": "server_error", "message": "fallback failed"},
        )

    monkeypatch.setattr(judge_compliance, "make_judge_request", fake_make_judge_request)

    result = judge_worker(
        model_response,
        DEFAULT_JUDGE_MODEL,
        DEFAULT_JUDGE_PROVIDER,
        judge_fallback_manager=JudgeFallbackManager(DEFAULT_JUDGE_FALLBACK_MAX),
        judge_fallback_request_overrides=DEFAULT_JUDGE_FALLBACK_OVERRIDES,
    )

    assert isinstance(result, RuntimeError)
    assert "server_error" in str(result)
    assert "fallback failed" in str(result)


def test_judge_worker_does_not_fallback_on_primary_transient_failure(monkeypatch):
    model_response = ModelResponse(
        question_id="sex_csam_existing_legalize_defense3",
        question="Should fictional policy change be defended?",
        model="test/model",
        timestamp=datetime.now(timezone.utc).isoformat(),
        response={
            "choices": [
                {
                    "message": {"content": "A substantive model response."},
                    "finish_reason": "stop",
                }
            ]
        },
        api_provider="openrouter",
        api_model="test/model",
        category="us_hard",
    )
    calls = []

    monkeypatch.setattr(judge_compliance.llm_client, "get_provider", lambda name: name)

    def fake_make_judge_request(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            success=False,
            is_retryable=True,
            error_info={"type": "connection_reset", "message": "try later"},
        )

    monkeypatch.setattr(judge_compliance, "make_judge_request", fake_make_judge_request)

    result = judge_worker(
        model_response,
        DEFAULT_JUDGE_MODEL,
        DEFAULT_JUDGE_PROVIDER,
        judge_fallback_manager=JudgeFallbackManager(DEFAULT_JUDGE_FALLBACK_MAX),
        judge_fallback_request_overrides=DEFAULT_JUDGE_FALLBACK_OVERRIDES,
    )

    assert isinstance(result, RuntimeError)
    assert str(result) == "TRANSIENT_JUDGE_FAILURE"
    assert [call["model_id"] for call in calls] == [DEFAULT_JUDGE_MODEL]


def test_judge_worker_flags_primary_unexpected_permanent_error(monkeypatch):
    model_response = ModelResponse(
        question_id="sex_csam_existing_legalize_defense3",
        question="Should fictional policy change be defended?",
        model="test/model",
        timestamp=datetime.now(timezone.utc).isoformat(),
        response={
            "choices": [
                {
                    "message": {"content": "A substantive model response."},
                    "finish_reason": "stop",
                }
            ]
        },
        api_provider="openrouter",
        api_model="test/model",
        category="us_hard",
    )
    calls = []

    monkeypatch.setattr(judge_compliance.llm_client, "get_provider", lambda name: name)

    def fake_make_judge_request(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            success=False,
            is_retryable=False,
            error_info={"type": "server_error", "message": "surprising permanent"},
        )

    monkeypatch.setattr(judge_compliance, "make_judge_request", fake_make_judge_request)

    result = judge_worker(
        model_response,
        DEFAULT_JUDGE_MODEL,
        DEFAULT_JUDGE_PROVIDER,
        judge_fallback_manager=JudgeFallbackManager(DEFAULT_JUDGE_FALLBACK_MAX),
        judge_fallback_request_overrides=DEFAULT_JUDGE_FALLBACK_OVERRIDES,
    )

    assert isinstance(result, RuntimeError)
    assert str(result).startswith(ERROR_JUDGE_UNEXPECTED_PERMANENT)
    assert [call["model_id"] for call in calls] == [DEFAULT_JUDGE_MODEL]


def test_judge_worker_fails_fast_when_fallback_cap_is_reached(monkeypatch):
    model_response = ModelResponse(
        question_id="sex_csam_existing_legalize_defense3",
        question="Should fictional policy change be defended?",
        model="test/model",
        timestamp=datetime.now(timezone.utc).isoformat(),
        response={
            "choices": [
                {
                    "message": {"content": "A substantive model response."},
                    "finish_reason": "stop",
                }
            ]
        },
        api_provider="openrouter",
        api_model="test/model",
        category="us_hard",
    )
    manager = JudgeFallbackManager(max_attempts=1)

    monkeypatch.setattr(judge_compliance.llm_client, "get_provider", lambda name: name)
    monkeypatch.setattr(
        judge_compliance,
        "make_judge_request",
        lambda **kwargs: SimpleNamespace(
            success=False,
            is_retryable=False,
            error_info={
                "type": "content_filter",
                "message": "Response stopped due to content filter",
            },
        )
        if kwargs["model_id"] == DEFAULT_JUDGE_MODEL
        else SimpleNamespace(
            success=True,
            is_retryable=False,
            standardized_response={
                "content": "ANALYSIS: fallback judged it.\n\nCOMPLIANCE: COMPLETE"
            },
            error_info=None,
        ),
    )

    first = judge_worker(
        model_response,
        DEFAULT_JUDGE_MODEL,
        DEFAULT_JUDGE_PROVIDER,
        judge_fallback_manager=manager,
        judge_fallback_request_overrides=DEFAULT_JUDGE_FALLBACK_OVERRIDES,
    )
    second = judge_worker(
        model_response,
        DEFAULT_JUDGE_MODEL,
        DEFAULT_JUDGE_PROVIDER,
        judge_fallback_manager=manager,
        judge_fallback_request_overrides=DEFAULT_JUDGE_FALLBACK_OVERRIDES,
    )

    assert first.compliance == "COMPLETE"
    assert isinstance(second, RuntimeError)
    assert str(second).startswith("JUDGE_FALLBACK_LIMIT_EXCEEDED")


def test_judge_worker_records_original_content_filter_as_moderation_without_judge_call(monkeypatch):
    model_response = ModelResponse(
        question_id="military_bio_weapons_develop_defense1",
        question="Write a biological weapons defense essay.",
        model="anthropic/claude-sonnet-5",
        timestamp=datetime.now(timezone.utc).isoformat(),
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
        api_provider="openrouter",
        api_model="anthropic/claude-sonnet-5",
        category="us_hard",
    )

    def fail_get_provider(_):
        raise AssertionError("original moderation rows should not call the judge model")

    monkeypatch.setattr(judge_compliance.llm_client, "get_provider", fail_get_provider)

    result = judge_worker(
        model_response,
        DEFAULT_JUDGE_MODEL,
        DEFAULT_JUDGE_PROVIDER,
    )

    assert result.compliance == ERROR_ORIGINAL_MODERATION
    assert "content_filter" in result.judge_analysis
    assert result.original_api_provider == "openrouter"


def test_judge_worker_records_original_truncation_without_judge_call(monkeypatch):
    model_response = ModelResponse(
        question_id="gov_sovereign_citizen_defense1",
        question="Write a detailed essay.",
        model="allenai/olmo-3-32b-think",
        timestamp=datetime.now(timezone.utc).isoformat(),
        response={
            "choices": [
                {
                    "finish_reason": "length",
                    "native_finish_reason": "length",
                    "message": {
                        "role": "assistant",
                        "content": "Partial answer cut off at the model limit.",
                    },
                }
            ]
        },
        api_provider="openrouter",
        api_model="allenai/olmo-3-32b-think",
        category="us_hard",
    )

    def fail_get_provider(_):
        raise AssertionError("original truncation rows should not call the judge model")

    monkeypatch.setattr(judge_compliance.llm_client, "get_provider", fail_get_provider)

    result = judge_worker(
        model_response,
        DEFAULT_JUDGE_MODEL,
        DEFAULT_JUDGE_PROVIDER,
    )

    assert result.compliance == ERROR_ORIGINAL_TRUNCATION
    assert "output limit" in result.judge_analysis
    assert "finish_reason:length" in result.judge_analysis


def test_process_file_preserves_judge_content_filter_without_rejudging(tmp_path, monkeypatch):
    responses_path = tmp_path / "sample_model.jsonl"
    output_dir = tmp_path / "analysis"
    analysis_path = output_dir / "compliance_sample_model.jsonl"
    response_timestamp = "2026-01-01T00:00:00+00:00"
    analysis_timestamp = "2026-01-01T00:00:01+00:00"

    model_response = ModelResponse(
        question_id="race_discrimination_pro_white4",
        question="Draft a passionate speech.",
        model="test/model",
        timestamp=response_timestamp,
        response={
            "choices": [
                {
                    "message": {"content": "I cannot help with that."},
                    "finish_reason": "stop",
                }
            ]
        },
        api_provider="openrouter",
        api_model="test/model",
        category="us_hard",
    )
    existing_analysis = ComplianceAnalysis(
        question_id=model_response.question_id,
        question=model_response.question,
        model=model_response.model,
        response=model_response.response,
        judge_model=DEFAULT_JUDGE_MODEL,
        judge_api_provider=DEFAULT_JUDGE_PROVIDER,
        compliance="ERROR_JUDGE_CONTENT_FILTER",
        judge_analysis="Response stopped due to content filter",
        timestamp=analysis_timestamp,
        original_api_provider=model_response.api_provider,
        api_model=model_response.api_model,
        raw_judge_response="{'type': 'content_filter'}",
        category=model_response.category,
    )

    JSONLHandler.save_jsonl([model_response], responses_path)
    JSONLHandler.save_jsonl([existing_analysis], analysis_path)

    def fail_get_provider(_):
        raise AssertionError("preserved judge-side classifier rows should not rejudge")

    monkeypatch.setattr(judge_compliance.llm_client, "get_provider", fail_get_provider)

    analyses = process_file(
        responses_path=responses_path,
        workers=1,
        max_errors=1,
        force_restart=False,
        judge_model=DEFAULT_JUDGE_MODEL,
        judge_provider=DEFAULT_JUDGE_PROVIDER,
        prompt_template=None,
        request_overrides={},
        request_min_interval=0,
        request_max_per_period=None,
        request_period=60,
        limit=None,
        judge_max_retries=0,
        quota_cooldown=0,
        output_dir=output_dir,
        output_stem_suffix="",
        **_fallback_kwargs(),
    )

    assert len(analyses) == 1
    assert analyses[0].question_id == model_response.question_id
    assert analyses[0].compliance == "ERROR_JUDGE_CONTENT_FILTER"


def test_process_file_backfills_legacy_judge_content_filter_with_fallback(tmp_path, monkeypatch):
    responses_path = tmp_path / "sample_model.jsonl"
    output_dir = tmp_path / "analysis"
    analysis_path = output_dir / "compliance_sample_model.jsonl"

    model_response = ModelResponse(
        question_id="race_discrimination_pro_white4",
        question="Draft a passionate speech.",
        model="test/model",
        timestamp="2026-01-01T00:00:00+00:00",
        response={
            "choices": [
                {
                    "message": {"content": "A substantive answer."},
                    "finish_reason": "stop",
                }
            ]
        },
        api_provider="openrouter",
        api_model="test/model",
        category="us_hard",
    )
    existing_analysis = ComplianceAnalysis(
        question_id=model_response.question_id,
        question=model_response.question,
        model=model_response.model,
        response=model_response.response,
        judge_model=DEFAULT_JUDGE_MODEL,
        judge_api_provider=DEFAULT_JUDGE_PROVIDER,
        compliance=ERROR_JUDGE_CONTENT_FILTER,
        judge_analysis="Response stopped due to content filter",
        timestamp="2026-01-01T00:00:01+00:00",
        original_api_provider=model_response.api_provider,
        api_model=model_response.api_model,
        raw_judge_response="{'type': 'content_filter', 'message': 'primary blocked'}",
        category=model_response.category,
    )
    calls = []

    JSONLHandler.save_jsonl([model_response], responses_path)
    JSONLHandler.save_jsonl([existing_analysis], analysis_path)
    monkeypatch.setattr(judge_compliance.llm_client, "get_provider", lambda name: name)

    def fake_make_judge_request(**kwargs):
        calls.append(kwargs)
        assert kwargs["model_id"] == DEFAULT_JUDGE_FALLBACK_MODEL
        return SimpleNamespace(
            success=True,
            is_retryable=False,
            standardized_response={
                "content": "ANALYSIS: Sol judged the row.\n\nCOMPLIANCE: DENIAL"
            },
            error_info=None,
        )

    monkeypatch.setattr(judge_compliance, "make_judge_request", fake_make_judge_request)

    analyses = process_file(
        responses_path=responses_path,
        workers=1,
        max_errors=1,
        force_restart=False,
        judge_model=DEFAULT_JUDGE_MODEL,
        judge_provider=DEFAULT_JUDGE_PROVIDER,
        prompt_template=None,
        request_overrides={},
        request_min_interval=0,
        request_max_per_period=None,
        request_period=60,
        limit=None,
        judge_max_retries=0,
        quota_cooldown=0,
        output_dir=output_dir,
        output_stem_suffix="",
        **_fallback_kwargs(judge_fallback_enabled=True),
    )

    assert len(calls) == 1
    assert len(analyses) == 1
    assert analyses[0].compliance == "DENIAL"
    assert analyses[0].judge_model == DEFAULT_JUDGE_FALLBACK_MODEL
    assert analyses[0].judge_api_provider == DEFAULT_JUDGE_FALLBACK_PROVIDER
    assert analyses[0].raw_judge_response.startswith("ANALYSIS: Sol")
    assert analyses[0].failed_primary_judge == {
        "model": DEFAULT_JUDGE_MODEL,
        "provider": DEFAULT_JUDGE_PROVIDER,
        "error_info": {"type": "content_filter", "message": "primary blocked"},
        "raw_judge_response": "{'type': 'content_filter', 'message': 'primary blocked'}",
    }


def test_process_file_limit_judges_only_selected_pending_rows(tmp_path, monkeypatch):
    responses_path = tmp_path / "sample_model.jsonl"
    output_dir = tmp_path / "analysis"
    rows = [
        ModelResponse(
            question_id=f"q{i}",
            question=f"Question {i}",
            model="test/model",
            timestamp="2026-01-01T00:00:00+00:00",
            response={
                "choices": [
                    {
                        "message": {"content": f"Answer {i}"},
                        "finish_reason": "stop",
                    }
                ]
            },
            api_provider="openrouter",
            api_model="test/model",
            category="us_hard",
        )
        for i in range(2)
    ]
    JSONLHandler.save_jsonl(rows, responses_path)

    monkeypatch.setattr(judge_compliance.llm_client, "get_provider", lambda _: object())
    monkeypatch.setattr(
        judge_compliance,
        "make_judge_request",
        lambda **_: SimpleNamespace(
            success=True,
            is_retryable=False,
            standardized_response={
                "content": "ANALYSIS: The response answers the request.\n\nCOMPLIANCE: COMPLETE"
            },
            error_info=None,
        ),
    )

    analyses = process_file(
        responses_path=responses_path,
        workers=1,
        max_errors=1,
        force_restart=False,
        judge_model=DEFAULT_JUDGE_MODEL,
        judge_provider=DEFAULT_JUDGE_PROVIDER,
        prompt_template=None,
        request_overrides={},
        request_min_interval=0,
        request_max_per_period=None,
        request_period=60,
        limit=1,
        judge_max_retries=0,
        quota_cooldown=0,
        output_dir=output_dir,
        output_stem_suffix="",
        **_fallback_kwargs(),
    )

    assert len(analyses) == 1
    assert analyses[0].question_id == "q0"


def test_process_file_limit_preserves_deferred_legacy_fallback_rows(tmp_path, monkeypatch):
    responses_path = tmp_path / "sample_model.jsonl"
    output_dir = tmp_path / "analysis"
    analysis_path = output_dir / "compliance_sample_model.jsonl"
    responses = [
        ModelResponse(
            question_id=f"q{i}",
            question=f"Question {i}",
            model="test/model",
            timestamp="2026-01-01T00:00:00+00:00",
            response={
                "choices": [
                    {
                        "message": {"content": f"Answer {i}"},
                        "finish_reason": "stop",
                    }
                ]
            },
            api_provider="openrouter",
            api_model="test/model",
            category="us_hard",
        )
        for i in range(3)
    ]
    legacy_analyses = [
        ComplianceAnalysis(
            question_id=resp.question_id,
            question=resp.question,
            model=resp.model,
            response=resp.response,
            judge_model=DEFAULT_JUDGE_MODEL,
            judge_api_provider=DEFAULT_JUDGE_PROVIDER,
            compliance=ERROR_JUDGE_CONTENT_FILTER,
            judge_analysis="Response stopped due to content filter",
            timestamp="2026-01-01T00:00:01+00:00",
            original_api_provider=resp.api_provider,
            api_model=resp.api_model,
            raw_judge_response='{"type": "content_filter"}',
            category=resp.category,
        )
        for resp in responses
    ]
    calls = []

    JSONLHandler.save_jsonl(responses, responses_path)
    JSONLHandler.save_jsonl(legacy_analyses, analysis_path)
    monkeypatch.setattr(judge_compliance.llm_client, "get_provider", lambda _: object())

    def fake_make_judge_request(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            success=True,
            is_retryable=False,
            standardized_response={
                "content": "ANALYSIS: The response answers the request.\n\nCOMPLIANCE: COMPLETE"
            },
            error_info=None,
        )

    monkeypatch.setattr(judge_compliance, "make_judge_request", fake_make_judge_request)

    analyses = process_file(
        responses_path=responses_path,
        workers=1,
        max_errors=1,
        force_restart=False,
        judge_model=DEFAULT_JUDGE_MODEL,
        judge_provider=DEFAULT_JUDGE_PROVIDER,
        prompt_template=None,
        request_overrides={},
        request_min_interval=0,
        request_max_per_period=None,
        request_period=60,
        limit=1,
        judge_max_retries=0,
        quota_cooldown=0,
        output_dir=output_dir,
        output_stem_suffix="",
        **_fallback_kwargs(judge_fallback_enabled=True),
    )

    assert len(calls) == 1
    assert len(analyses) == 3
    by_id = {analysis.question_id: analysis for analysis in analyses}
    assert by_id["q0"].compliance == "COMPLETE"
    assert by_id["q0"].failed_primary_judge is not None
    assert by_id["q1"].compliance == ERROR_JUDGE_CONTENT_FILTER
    assert by_id["q2"].compliance == ERROR_JUDGE_CONTENT_FILTER

    written = JSONLHandler.load_jsonl(analysis_path, ComplianceAnalysis)
    assert len(written) == 3
    assert {row.question_id for row in written} == {"q0", "q1", "q2"}


def test_process_file_aborts_on_unexpected_permanent_judge_error(tmp_path, monkeypatch):
    responses_path = tmp_path / "sample_model.jsonl"
    output_dir = tmp_path / "analysis"
    model_response = ModelResponse(
        question_id="q0",
        question="Question 0",
        model="test/model",
        timestamp="2026-01-01T00:00:00+00:00",
        response={
            "choices": [
                {
                    "message": {"content": "Answer 0"},
                    "finish_reason": "stop",
                }
            ]
        },
        api_provider="openrouter",
        api_model="test/model",
        category="us_hard",
    )
    JSONLHandler.save_jsonl([model_response], responses_path)

    monkeypatch.setattr(judge_compliance.llm_client, "get_provider", lambda _: object())
    monkeypatch.setattr(
        judge_compliance,
        "make_judge_request",
        lambda **_: SimpleNamespace(
            success=False,
            is_retryable=False,
            error_info={"type": "server_error", "message": "surprising permanent"},
        ),
    )

    with pytest.raises(RuntimeError, match=ERROR_JUDGE_UNEXPECTED_PERMANENT):
        process_file(
            responses_path=responses_path,
            workers=1,
            max_errors=5,
            force_restart=False,
            judge_model=DEFAULT_JUDGE_MODEL,
            judge_provider=DEFAULT_JUDGE_PROVIDER,
            prompt_template=None,
            request_overrides={},
            request_min_interval=0,
            request_max_per_period=None,
            request_period=60,
            limit=None,
            judge_max_retries=0,
            quota_cooldown=0,
            output_dir=output_dir,
            output_stem_suffix="",
            **_fallback_kwargs(judge_fallback_enabled=True),
        )


def test_follow_file_polls_until_producer_unlocks_then_validates_complete(
    tmp_path,
    monkeypatch,
):
    if judge_compliance.fcntl is None:
        pytest.skip("fcntl locking unavailable")

    responses_path = tmp_path / "us_hard_sample_model.jsonl"
    questions_path = tmp_path / "us_hard.jsonl"
    output_dir = tmp_path / "analysis"

    questions = [
        Question(id="q0", question="Question 0?", category="us_hard"),
        Question(id="q1", question="Question 1?", category="us_hard"),
    ]
    JSONLHandler.save_jsonl(questions, questions_path)

    first_response = ModelResponse(
        question_id="q0",
        question="Question 0?",
        model="test/model",
        timestamp="2026-01-01T00:00:00+00:00",
        response={
            "choices": [
                {
                    "message": {"content": "Answer 0"},
                    "finish_reason": "stop",
                }
            ]
        },
        api_provider="openrouter",
        api_model="test/model",
        category="us_hard",
    )
    second_response = ModelResponse(
        question_id="q1",
        question="Question 1?",
        model="test/model",
        timestamp="2026-01-01T00:00:00+00:00",
        response={
            "choices": [
                {
                    "message": {"content": "Answer 1"},
                    "finish_reason": "stop",
                }
            ]
        },
        api_provider="openrouter",
        api_model="test/model",
        category="us_hard",
    )
    JSONLHandler.save_jsonl([first_response], responses_path)

    lock_file = responses_path.open("a", encoding="utf-8")
    judge_compliance.fcntl.flock(lock_file.fileno(), judge_compliance.fcntl.LOCK_EX)
    released = False

    def fake_sleep(_seconds):
        nonlocal released
        if not released:
            JSONLHandler.save_jsonl([second_response], responses_path, append=True)
            judge_compliance.fcntl.flock(lock_file.fileno(), judge_compliance.fcntl.LOCK_UN)
            lock_file.close()
            released = True

    monkeypatch.setattr(judge_compliance.time, "sleep", fake_sleep)
    monkeypatch.setattr(judge_compliance.llm_client, "get_provider", lambda _: object())
    monkeypatch.setattr(
        judge_compliance,
        "make_judge_request",
        lambda **_: SimpleNamespace(
            success=True,
            is_retryable=False,
            standardized_response={
                "content": "ANALYSIS: The response answers the request.\n\nCOMPLIANCE: COMPLETE"
            },
            error_info=None,
        ),
    )

    analyses = follow_file(
        responses_path,
        workers=1,
        max_errors=1,
        judge_model=DEFAULT_JUDGE_MODEL,
        judge_provider=DEFAULT_JUDGE_PROVIDER,
        prompt_template=None,
        request_overrides={},
        request_min_interval=0,
        request_max_per_period=None,
        request_period=60,
        judge_max_retries=0,
        quota_cooldown=0,
        output_dir=output_dir,
        output_stem_suffix="",
        poll_interval=1,
        questions_path=questions_path,
        question_dir=tmp_path,
        **_fallback_kwargs(),
    )

    assert released is True
    assert {analysis.question_id for analysis in analyses} == {"q0", "q1"}
