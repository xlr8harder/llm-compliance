from types import SimpleNamespace

import pytest

from compliance.utils import llm_requests


class DummyProvider:
    api_base = "https://old.example/v1"


class NoApiBaseProvider:
    pass


def test_request_model_response_applies_api_base_override(monkeypatch):
    provider = DummyProvider()
    captured = {}

    monkeypatch.setattr(llm_requests.llm_client, "get_provider", lambda _name: provider)

    def fake_retry_request(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(success=True)

    monkeypatch.setattr(llm_requests, "retry_request", fake_retry_request)

    result = llm_requests.request_model_response(
        provider_name="custom-provider",
        api_model="example/custom-model",
        prompt="hello",
        overrides={
            "api_base": "https://api.example-override.invalid/v1",
            "max_tokens": 123,
        },
    )

    assert result.success is True
    assert provider.api_base == "https://api.example-override.invalid/v1"
    assert captured["provider"] is provider
    assert captured["model_id"] == "example/custom-model"
    assert captured["max_tokens"] == 123
    assert "api_base" not in captured


def test_request_model_response_rejects_api_base_override_for_unsupported_provider(
    monkeypatch,
):
    monkeypatch.setattr(
        llm_requests.llm_client,
        "get_provider",
        lambda _name: NoApiBaseProvider(),
    )

    with pytest.raises(ValueError, match="does not support api_base override"):
        llm_requests.request_model_response(
            provider_name="unsupported",
            api_model="model",
            prompt="hello",
            overrides={"api_base": "https://api.example/v1"},
        )
