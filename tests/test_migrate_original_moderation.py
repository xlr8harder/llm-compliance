import json

from tools.migrate_original_moderation import (
    ERROR_ORIGINAL_MODERATION,
    migrate_file,
    recode_row,
)


def _row(compliance, response):
    return {
        "question_id": "q1",
        "question": "Question?",
        "model": "test/model",
        "response": response,
        "judge_model": "judge/model",
        "compliance": compliance,
        "judge_analysis": "old analysis",
        "raw_judge_response": "old judge output",
        "timestamp": "2026-01-01T00:00:00+00:00",
    }


def test_recode_row_marks_native_refusal_with_partial_text_as_original_moderation():
    row = _row(
        "COMPLETE",
        {
            "choices": [
                {
                    "finish_reason": "stop",
                    "native_finish_reason": "refusal",
                    "message": {"content": "partial text"},
                }
            ]
        },
    )

    updated, reason, old_label = recode_row(
        row,
        migrated_at="2026-07-02T00:00:00+00:00",
    )

    assert reason == "refusal"
    assert old_label == "COMPLETE"
    assert updated["compliance"] == ERROR_ORIGINAL_MODERATION
    assert updated["timestamp"] == "2026-07-02T00:00:00+00:00"
    assert "COMPLETE" in updated["judge_analysis"]
    assert "raw_judge_response" not in updated


def test_recode_row_leaves_normal_judged_response_unchanged():
    row = _row(
        "DENIAL",
        {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": "I cannot help with that."},
                }
            ]
        },
    )

    updated, reason, old_label = recode_row(
        row,
        migrated_at="2026-07-02T00:00:00+00:00",
    )

    assert updated is row
    assert reason is None
    assert old_label is None


def test_recode_row_marks_usage_policy_error_text_as_original_moderation():
    row = _row(
        "ERROR_ORIGINAL_RESPONSE",
        {
            "error": {
                "message": (
                    "Upstream error from OpenAI: Invalid prompt: your prompt "
                    "was flagged as potentially violating our usage policy."
                ),
                "type": "invalid_request_error",
            }
        },
    )

    updated, reason, old_label = recode_row(
        row,
        migrated_at="2026-07-02T00:00:00+00:00",
    )

    assert reason == "moderation_error_text"
    assert old_label == "ERROR_ORIGINAL_RESPONSE"
    assert updated["compliance"] == ERROR_ORIGINAL_MODERATION


def test_recode_row_does_not_mark_invalid_model_as_moderation():
    row = _row(
        "ERROR_ORIGINAL_RESPONSE",
        {
            "error": {
                "message": "invalid model ID",
                "type": "invalid_request_error",
            }
        },
    )

    updated, reason, old_label = recode_row(
        row,
        migrated_at="2026-07-02T00:00:00+00:00",
    )

    assert updated is row
    assert reason is None
    assert old_label is None


def test_migrate_file_rewrites_only_changed_rows(tmp_path):
    path = tmp_path / "compliance_sample.jsonl"
    moderated = _row(
        "EVASIVE",
        {
            "choices": [
                {
                    "finish_reason": "content_filter",
                    "message": {"content": "partial text"},
                }
            ]
        },
    )
    normal = _row(
        "COMPLETE",
        {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": "complete answer"},
                }
            ]
        },
    )
    normal["question_id"] = "q2"
    path.write_text(
        "\n".join(json.dumps(row) for row in [moderated, normal]) + "\n",
        encoding="utf-8",
    )

    stats = migrate_file(
        path,
        migrated_at="2026-07-02T00:00:00+00:00",
    )

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert stats.rows_changed == 1
    assert stats.changed_by_old_label["EVASIVE"] == 1
    assert rows[0]["compliance"] == ERROR_ORIGINAL_MODERATION
    assert rows[1]["compliance"] == "COMPLETE"
