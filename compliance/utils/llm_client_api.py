"""Explicit legacy/V2 llm_client selection with a stable V1 projection."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

import llm_client


SUPPORTED_CLIENT_APIS = ("legacy", "v2")
NATIVE_V2_PROVIDERS = {"openrouter", "openai", "codex", "local"}


def get_provider(provider_name: str, client_api: str = "legacy") -> Any:
    if client_api == "legacy":
        return llm_client.get_provider(provider_name)
    if client_api == "v2":
        return V2ProviderAdapter(provider_name)
    raise ValueError(
        f"Unknown llm_client API {client_api!r}; expected one of {SUPPORTED_CLIENT_APIS}"
    )


@dataclass
class V2LegacyProjection:
    """Expose V2 evidence through the attributes consumed by V1 pipelines."""

    success: bool
    standardized_response: dict[str, Any]
    error_info: dict[str, Any] | None
    raw_provider_response: Any
    request_format: str
    raw_response_format: str
    is_retryable: bool
    context: Any
    canonical_conversation: dict[str, Any]


class V2ProviderAdapter:
    """Present a V1 provider interface while executing through llm_client V2."""

    def __init__(self, provider_name: str):
        self.provider_name = provider_name

    def make_chat_completion_request(
        self, messages, model_id, context=None, **options
    ) -> V2LegacyProjection:
        return self.make_request(
            messages=messages,
            model_id=model_id,
            context=context,
            request_format="chat_completions",
            **options,
        )

    def make_request(
        self,
        messages,
        model_id,
        context=None,
        request_format="chat_completions",
        **options,
    ) -> V2LegacyProjection:
        if self.provider_name not in NATIVE_V2_PROVIDERS:
            return self._make_legacy_import(
                messages=messages,
                model_id=model_id,
                context=context,
                request_format=request_format,
                **options,
            )
        protocol = _protocol_name(request_format)
        timeout = options.pop("timeout", 60)
        options = _v2_options(self.provider_name, options)
        local_endpoint = (
            os.getenv("LOCAL_LLM_BASE_URL") if self.provider_name == "local" else None
        )
        route = (
            f"local/unset/{model_id}"
            if self.provider_name == "local"
            else f"{self.provider_name}/{model_id}"
        )

        with llm_client.Client(
            timeout=timeout,
            max_retries=0,
            local_endpoint=local_endpoint,
        ) as client:
            model = client.model(route, protocol=protocol)
            conversation = model.conversation(messages=messages)
            response = conversation.send_pending(**options)

        return V2LegacyProjection(
            success=response.ok,
            standardized_response=_legacy_standardized_projection(
                self.provider_name,
                protocol,
                response.raw_provider_response,
                response.standardized_response,
                context,
            ),
            error_info=_legacy_error_info(response),
            raw_provider_response=response.raw_provider_response,
            request_format=_legacy_request_format(protocol),
            raw_response_format=_raw_response_format(
                self.provider_name, protocol, response.raw_provider_response
            ),
            is_retryable=bool(response.error and response.error.retryable),
            context=context,
            canonical_conversation=conversation.to_dict(),
        )

    def _make_legacy_import(
        self,
        *,
        messages,
        model_id,
        context,
        request_format,
        **options,
    ):
        provider = llm_client.get_provider(self.provider_name)
        response = provider.make_request(
            messages=messages,
            model_id=model_id,
            context=context,
            request_format=request_format,
            **options,
        )
        conversation = llm_client.Conversation.import_legacy(
            response,
            messages=messages,
            model=f"{self.provider_name}/{model_id}",
            preserve_source="full",
        )
        response.canonical_conversation = conversation.to_dict()
        return response


def _protocol_name(request_format: str | None) -> str:
    normalized = (request_format or "chat_completions").replace("-", "_")
    aliases = {
        "chat_completion": "chat_completions",
        "anthropic_message": "messages",
        "anthropic_messages": "messages",
        "anthropic_messages_api": "messages",
        "response": "responses",
        "responses_api": "responses",
    }
    return aliases.get(normalized, normalized)


def _legacy_request_format(protocol: str) -> str:
    return "anthropic_messages" if protocol == "messages" else protocol


def _v2_options(provider_name: str, options: dict[str, Any]) -> dict[str, Any]:
    converted = dict(options)
    converted.pop("context", None)
    transport = converted.pop("transport", None)
    if transport == "stream":
        converted["stream"] = True

    if provider_name == "openrouter":
        routing = dict(converted.pop("provider", {}) or {})
        only = converted.pop("only", None)
        allow_list = converted.pop("allow_list", None)
        ignore_list = converted.pop("ignore_list", None)
        if only or allow_list:
            routing["order"] = list(only or allow_list)
            routing["allow_fallbacks"] = False
        if ignore_list:
            routing["ignore"] = list(ignore_list)
        if routing:
            converted["provider_options"] = routing
    return converted


def _legacy_error_info(response) -> dict[str, Any] | None:
    if response.error is None:
        return None
    if response.error.category == "truncation":
        return None
    error = response.error.to_dict()
    error["type"] = (
        "content_filter"
        if response.error.category == "moderation"
        else response.error.provider_type or response.error.category
    )
    if response.error.provider_code is not None:
        error["code"] = response.error.provider_code
    return error


def _legacy_standardized_projection(
    provider_name: str,
    protocol: str,
    raw: Any,
    fallback: dict[str, Any],
    context: Any,
) -> dict[str, Any]:
    """Reuse legacy pure normalizers so the compatibility sidecar stays exact."""
    if not isinstance(raw, dict):
        return {key: value for key, value in fallback.items() if value is not None}
    provider = llm_client.get_provider(provider_name)
    try:
        if protocol == "messages" and hasattr(
            provider, "_standardize_anthropic_messages_response"
        ):
            return provider._standardize_anthropic_messages_response(raw)
        if protocol == "responses" and hasattr(provider, "_response_from_payload"):
            projected = provider._response_from_payload(raw, context)
            if isinstance(projected.standardized_response, dict):
                return projected.standardized_response
        if hasattr(provider, "_standardize_response"):
            return provider._standardize_response(raw)
    except (KeyError, TypeError, ValueError):
        pass
    return {key: value for key, value in fallback.items() if value is not None}


def _raw_response_format(provider: str, protocol: str, raw: Any) -> str:
    if isinstance(raw, dict) and raw.get("error") is not None:
        return f"{provider}.error"
    suffix = {
        "chat_completions": "chat_completions",
        "messages": "anthropic_messages",
        "responses": "responses",
    }.get(protocol, f"{protocol}.unknown")
    return f"{provider}.{suffix}"
