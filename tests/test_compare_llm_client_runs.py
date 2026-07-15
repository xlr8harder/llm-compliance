from tools.compare_llm_client_runs import compare_runs


def _row(*, status="success", raw_extra=None, canonical=False):
    response = {
        "choices": [],
        "_llm_client": {
            "success": status == "success",
            "is_retryable": False,
            "standardized_response": {
                "content": "generated text",
                "finish_reason": "stop",
                "native_finish_reason": "completed",
            },
        },
        **(raw_extra or {}),
    }
    if canonical:
        response["_llm_client_v2"] = {"schema_version": 1}
    return {
        "question_id": "q1",
        "response": response,
        "response_status": status,
        "response_status_reason": "finish_reason:stop;native_finish_reason:completed",
        "request_format": "chat_completions",
        "raw_response_format": "openrouter.chat_completions",
    }


def test_comparison_accepts_canonical_addition():
    report = compare_runs({"q1": _row()}, {"q1": _row(canonical=True)})
    assert report["compatible"] is True
    assert report["rows"][0]["v2_canonical_present"] is True


def test_comparison_reports_classification_and_raw_shape_drift():
    report = compare_runs(
        {"q1": _row()},
        {"q1": _row(status="truncation", raw_extra={"future": True})},
    )
    assert report["compatible"] is False
    assert "response_status" in report["rows"][0]["mismatches"]
    assert "raw_response_keys" in report["rows"][0]["mismatches"]


def test_comparison_reports_standardized_shape_drift():
    v2 = _row(canonical=True)
    v2["response"]["_llm_client"]["standardized_response"]["reasoning"] = None
    report = compare_runs({"q1": _row()}, {"q1": v2})
    assert report["compatible"] is False
    assert "standardized_response.keys" in report["rows"][0]["mismatches"]
