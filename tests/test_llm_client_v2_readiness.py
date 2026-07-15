from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from concurrent.futures import ThreadPoolExecutor
import json
import threading
from types import SimpleNamespace

from ask import response_payload_with_client_metadata
from compliance.data import ModelResponse
from compliance.utils.llm_client_api import V2ProviderAdapter, _v2_options
from llm_client import Conversation


def _start_local_server(response_payload=None, status_code=200):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            return

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            requests.append(json.loads(self.rfile.read(length)))
            body = json.dumps(
                response_payload
                or {
                    "id": "local-response-1",
                    "object": "chat.completion",
                    "model": "test-model",
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "HELLO"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 4,
                        "completion_tokens": 1,
                        "total_tokens": 5,
                    },
                }
            ).encode()
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, requests


def test_v2_adapter_preserves_v1_projection_and_canonical_record(
    monkeypatch,
):
    server, requests = _start_local_server()
    monkeypatch.setenv(
        "LOCAL_LLM_BASE_URL", f"http://127.0.0.1:{server.server_port}/v1"
    )
    try:
        response = V2ProviderAdapter("local").make_request(
            messages=[{"role": "user", "content": "Say hello"}],
            model_id="test-model",
            context={"qid": "q1"},
            max_tokens=20,
        )
    finally:
        server.shutdown()
        server.server_close()

    assert response.success is True
    assert response.standardized_response["content"] == "HELLO"
    assert response.error_info is None
    assert response.request_format == "chat_completions"
    assert response.raw_response_format == "local.chat_completions"
    assert response.context == {"qid": "q1"}
    assert requests[0]["messages"] == [{"role": "user", "content": "Say hello"}]

    payload = response_payload_with_client_metadata(response)
    assert payload["_llm_client"] == {
        "success": True,
        "is_retryable": False,
        "standardized_response": response.standardized_response,
        "request_format": "chat_completions",
        "raw_response_format": "local.chat_completions",
    }
    canonical = payload["_llm_client_v2"]
    assert canonical["default_route"]["model"] == "local/unset/test-model"
    assert "127.0.0.1" not in json.dumps(canonical)
    assert Conversation.from_dict(canonical).to_dict() == canonical

    stored = ModelResponse(
        question_id="q1",
        question="Say hello",
        model="test-model",
        timestamp="2026-01-01T00:00:00+00:00",
        response=payload,
    )
    assert stored.final_content_text() == "HELLO"
    assert stored.classify_response_status()[0] == "success"


def test_v2_openrouter_option_translation():
    assert _v2_options(
        "openrouter",
        {
            "only": ["OpenAI"],
            "ignore_list": ["Other"],
            "provider": {"data_collection": "deny"},
            "max_tokens": 100,
        },
    ) == {
        "provider_options": {
            "data_collection": "deny",
            "order": ["OpenAI"],
            "allow_fallbacks": False,
            "ignore": ["Other"],
        },
        "max_tokens": 100,
    }


def test_unmigrated_provider_uses_legacy_transport_and_canonical_import(monkeypatch):
    calls = []
    legacy_response = SimpleNamespace(
        success=True,
        standardized_response={
            "content": "answer",
            "finish_reason": "stop",
            "native_finish_reason": "STOP",
        },
        error_info=None,
        raw_provider_response={"candidates": [{"finishReason": "STOP"}]},
        request_format="chat_completions",
        raw_response_format="google.response",
        is_retryable=False,
        context={"qid": "q-google"},
    )

    class LegacyProvider:
        def make_request(self, **kwargs):
            calls.append(kwargs)
            return legacy_response

    monkeypatch.setattr(
        "compliance.utils.llm_client_api.llm_client.get_provider",
        lambda name: LegacyProvider(),
    )
    response = V2ProviderAdapter("google_agent_platform").make_request(
        messages=[{"role": "user", "content": "question"}],
        model_id="publisher/model",
        context={"qid": "q-google"},
        timeout=90,
    )

    assert response is legacy_response
    assert calls[0]["model_id"] == "publisher/model"
    canonical = response.canonical_conversation
    assert canonical["default_route"]["model"] == (
        "google_agent_platform/publisher/model"
    )
    migration = canonical["operations"][0]["result"]["metadata"]["migration"]
    assert migration["source_format"] == "llm_client.LLMResponse"
    assert migration["source_record"]["context"] == {"qid": "q-google"}
    assert Conversation.from_dict(canonical).to_dict() == canonical


def test_v2_moderation_error_retains_legacy_classification(monkeypatch):
    server, _ = _start_local_server(
        {
            "error": {
                "code": "content_filter",
                "type": "moderation_error",
                "message": "blocked by moderation",
            }
        },
        status_code=400,
    )
    monkeypatch.setenv(
        "LOCAL_LLM_BASE_URL", f"http://127.0.0.1:{server.server_port}/v1"
    )
    try:
        response = V2ProviderAdapter("local").make_request(
            messages=[{"role": "user", "content": "blocked"}],
            model_id="test-model",
        )
    finally:
        server.shutdown()
        server.server_close()

    assert response.success is False
    assert response.is_retryable is False
    assert response.error_info["type"] == "content_filter"
    assert response.error_info["provider_type"] == "moderation_error"
    payload = response_payload_with_client_metadata(response)
    stored = ModelResponse(
        question_id="q-moderation",
        question="blocked",
        model="test-model",
        timestamp="2026-01-01T00:00:00+00:00",
        response=payload,
    )
    assert stored.is_original_moderation_error()
    assert (
        Conversation.from_dict(payload["_llm_client_v2"]).to_dict()
        == payload["_llm_client_v2"]
    )


def test_v2_truncation_and_retryable_error_project_unambiguously(monkeypatch):
    cases = [
        (
            {
                "id": "truncated",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "partial"},
                        "finish_reason": "length",
                    }
                ],
                "usage": {},
            },
            200,
            "truncation",
            False,
        ),
        (
            {"error": {"code": "overloaded", "message": "try later"}},
            503,
            "provider_error",
            True,
        ),
    ]
    for index, (raw, status_code, expected_status, retryable) in enumerate(cases):
        server, _ = _start_local_server(raw, status_code=status_code)
        monkeypatch.setenv(
            "LOCAL_LLM_BASE_URL", f"http://127.0.0.1:{server.server_port}/v1"
        )
        try:
            response = V2ProviderAdapter("local").make_request(
                messages=[{"role": "user", "content": "test"}],
                model_id="test-model",
            )
        finally:
            server.shutdown()
            server.server_close()
        payload = response_payload_with_client_metadata(response)
        stored = ModelResponse(
            question_id=f"q-{index}",
            question="test",
            model="test-model",
            timestamp="2026-01-01T00:00:00+00:00",
            response=payload,
        )
        assert stored.classify_response_status()[0] == expected_status
        assert response.is_retryable is retryable


def test_v2_adapter_supports_parallel_independent_requests(monkeypatch):
    server, requests = _start_local_server()
    monkeypatch.setenv(
        "LOCAL_LLM_BASE_URL", f"http://127.0.0.1:{server.server_port}/v1"
    )

    def send(index):
        return V2ProviderAdapter("local").make_request(
            messages=[{"role": "user", "content": f"question-{index}"}],
            model_id="test-model",
            context={"qid": f"q-{index}"},
        )

    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            responses = list(pool.map(send, range(20)))
    finally:
        server.shutdown()
        server.server_close()

    assert len(requests) == 20
    assert {response.context["qid"] for response in responses} == {
        f"q-{index}" for index in range(20)
    }
    assert all(response.success for response in responses)


def test_responses_payload_projects_text_for_existing_status_classifier():
    response = SimpleNamespace(
        error=None,
        standardized_response={
            "content": "HELLO",
            "reasoning": None,
            "finish_reason": "stop",
            "native_finish_reason": "completed",
            "usage": {"input_tokens": 4, "output_tokens": 1},
        },
        raw_provider_response={
            "id": "resp-1",
            "object": "response",
            "status": "completed",
            "model": "openai/test",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "HELLO"}],
                }
            ],
            "usage": {"input_tokens": 4, "output_tokens": 1},
        },
    )
    from compliance.utils.llm_client_api import _legacy_standardized_projection

    standardized = _legacy_standardized_projection(
        "openrouter",
        "responses",
        response.raw_provider_response,
        response.standardized_response,
        None,
    )
    assert standardized["content"] == "HELLO"
    payload = dict(response.raw_provider_response)
    payload["_llm_client"] = {
        "success": True,
        "is_retryable": False,
        "standardized_response": standardized,
    }
    stored = ModelResponse(
        question_id="q-responses",
        question="Say hello",
        model="openai/test",
        timestamp="2026-01-01T00:00:00+00:00",
        response=payload,
    )
    assert stored.final_content_text() == "HELLO"
    assert stored.classify_response_status()[0] == "success"
