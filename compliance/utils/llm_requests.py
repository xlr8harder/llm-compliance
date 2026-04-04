"""
Shared helpers for issuing model requests through llm_client.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import llm_client
from llm_client.retry import retry_request

from ..models import ModelCatalog


def resolve_catalog_entry(
    catalog: ModelCatalog,
    canonical_name: Optional[str],
    provider: str,
    api_model: str,
) -> tuple[Optional[str], str, str]:
    canonical, provider_resolved, api_model_resolved = catalog.resolve_model(
        canonical_name=canonical_name,
        provider=provider,
        provider_model_id=api_model,
    )
    if not provider_resolved or not api_model_resolved:
        raise RuntimeError("Cannot resolve provider / model – please provide explicit flags")
    return canonical or canonical_name, provider_resolved, api_model_resolved


def request_model_response(
    *,
    provider_name: str,
    api_model: str,
    prompt: str,
    ignore_list: Optional[List[str]] = None,
    overrides: Optional[Dict[str, Any]] = None,
    system_prompt: Optional[str] = None,
    force_subprovider: Optional[str] = None,
    timeout: int = 180,
    context: Optional[Dict[str, Any]] = None,
):
    provider = llm_client.get_provider(provider_name)

    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    kwargs: Dict[str, Any] = {}
    if overrides:
        kwargs.update(overrides)
    kwargs.setdefault("timeout", timeout)

    if provider_name == "openrouter":
        if force_subprovider:
            kwargs["only"] = [force_subprovider]
        elif ignore_list:
            kwargs["ignore_list"] = list(ignore_list)

    return retry_request(
        provider=provider,
        messages=messages,
        model_id=api_model,
        max_retries=4,
        context=context or {},
        **kwargs,
    )
