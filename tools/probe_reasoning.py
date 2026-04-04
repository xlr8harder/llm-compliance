#!/usr/bin/env python3
"""
Probe a model's reasoning configuration and recommend run flags/canonical naming.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add the parent directory to the path so we can import compliance
script_dir = Path(__file__).parent.resolve()
parent_dir = script_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

import llm_client

from compliance.models import ModelCatalog
from compliance.utils.llm_requests import request_model_response, resolve_catalog_entry
from compliance.utils.reasoning import summarize_reasoning_payload


LOGGER = logging.getLogger("probe_reasoning")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

DEFAULT_PROMPT = "Explain briefly why the sky appears blue."


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Probe reasoning support for a model and recommend standardized run flags."
    )
    parser.add_argument("--provider", choices=llm_client.PROVIDER_MAP.keys(), required=True, help="API provider")
    parser.add_argument("--model", required=True, help="provider-specific model id")
    parser.add_argument("--canonical-name", help="canonical model name for reporting")
    parser.add_argument("--catalog", type=Path, default=Path("model_catalog.jsonl"))
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="probe prompt")
    parser.add_argument("--system-prompt", help="system prompt to include")
    parser.add_argument(
        "--force-subprovider",
        help="OpenRouter only: restrict to a single subprovider (uses OpenRouter 'only').",
    )
    parser.add_argument(
        "-t",
        "--timeout-seconds",
        type=int,
        default=180,
        help="request timeout in seconds (default: 180)",
    )
    parser.add_argument("--json", action="store_true", help="emit a single JSON object")
    return parser


def build_probe_overrides(base_overrides: Dict[str, Any], probe: str) -> Dict[str, Any]:
    request_overrides = {k: v for k, v in base_overrides.items() if k != "reasoning"}

    if probe == "reasoning":
        request_overrides["reasoning"] = {"enabled": True}
    elif probe == "reasoning-medium":
        request_overrides["reasoning"] = {"enabled": True, "effort": "medium"}
    elif probe == "no-reasoning":
        request_overrides["reasoning"] = {"enabled": False}
    elif probe != "default":
        raise ValueError(f"unknown probe type: {probe}")

    return request_overrides


def run_probe(
    *,
    probe: str,
    prompt: str,
    provider_name: str,
    api_model: str,
    base_overrides: Dict[str, Any],
    system_prompt: Optional[str],
    force_subprovider: Optional[str],
    timeout_seconds: int,
) -> Dict[str, Any]:
    request_overrides = build_probe_overrides(base_overrides, probe)
    try:
        api_response = request_model_response(
            provider_name=provider_name,
            api_model=api_model,
            prompt=prompt,
            overrides=request_overrides if request_overrides else None,
            system_prompt=system_prompt,
            force_subprovider=force_subprovider,
            timeout=timeout_seconds,
            context={"probe": probe},
        )
        raw_response = api_response.raw_provider_response
        summary = summarize_reasoning_payload(raw_response)
        return {
            "probe": probe,
            "request_overrides": request_overrides,
            "summary": summary.to_dict(),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "probe": probe,
            "request_overrides": request_overrides,
            "summary": None,
            "error": str(exc),
        }


def reasoning_present(result: Optional[Dict[str, Any]]) -> Optional[bool]:
    if not result:
        return None
    if result.get("error"):
        return None
    summary = result.get("summary")
    if not isinstance(summary, dict):
        return None
    return bool(summary.get("reasoning_present"))


def recommend_configuration(canonical_name: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_probe = {result["probe"]: result for result in results}

    default_present = reasoning_present(by_probe.get("default"))
    reasoning_present_default = reasoning_present(by_probe.get("reasoning"))
    reasoning_present_medium = reasoning_present(by_probe.get("reasoning-medium"))
    no_reasoning_present = reasoning_present(by_probe.get("no-reasoning"))

    reasoning_probe = None
    reasoning_flags: Optional[List[str]] = None
    if reasoning_present_default is True:
        reasoning_probe = "reasoning"
        reasoning_flags = ["--reasoning"]
    elif reasoning_present_medium is True:
        reasoning_probe = "reasoning-medium"
        reasoning_flags = ["--reasoning", "--reasoning-effort", "medium"]

    recommendation: Dict[str, Any] = {
        "mode": "indeterminate",
        "base_canonical_name": canonical_name,
        "reasoning_canonical_name": None,
        "base_run_flags": None,
        "reasoning_run_flags": None,
        "notes": "Probe did not produce a clear recommendation.",
    }

    if default_present is False and reasoning_flags:
        recommendation.update(
            {
                "mode": "paired_modes",
                "reasoning_canonical_name": f"{canonical_name}-reasoning",
                "base_run_flags": ["--no-reasoning"],
                "reasoning_run_flags": reasoning_flags,
                "notes": "Reasoning traces appear only when reasoning is explicitly enabled.",
            }
        )
        return recommendation

    if default_present is True and no_reasoning_present is False:
        recommendation.update(
            {
                "mode": "paired_modes",
                "reasoning_canonical_name": f"{canonical_name}-reasoning",
                "base_run_flags": ["--no-reasoning"],
                "reasoning_run_flags": reasoning_flags or ["--reasoning"],
                "notes": "Default responses include reasoning traces, but explicit --no-reasoning suppresses them.",
            }
        )
        return recommendation

    if default_present is False and reasoning_flags is None:
        recommendation.update(
            {
                "mode": "single_mode_base_only",
                "base_run_flags": ["--no-reasoning"],
                "notes": "No reasoning traces were observed, even when reasoning was explicitly enabled.",
            }
        )
        return recommendation

    if default_present is True and no_reasoning_present is True:
        recommendation.update(
            {
                "mode": "single_mode_reasoning_only",
                "base_run_flags": ["--reasoning"],
                "notes": "Reasoning traces were present in default and explicit --no-reasoning probes, so the provider appears to expose only one reasoning-like mode.",
            }
        )
        return recommendation

    return recommendation


def print_human_report(
    *,
    canonical_name: str,
    provider_name: str,
    api_model: str,
    prompt: str,
    results: List[Dict[str, Any]],
    recommendation: Dict[str, Any],
) -> None:
    print(f"MODEL {canonical_name}")
    print(f"provider={provider_name} api_model={api_model}")
    print(f"prompt={prompt!r}")
    for result in results:
        probe = result["probe"]
        if result.get("error"):
            print(f"probe={probe} error={result['error']}")
            continue
        summary = result["summary"]
        print(
            "probe={probe} reasoning_present={reasoning_present} reasoning_tokens={reasoning_tokens} "
            "provider={provider} response_model={response_model} finish_reason={finish_reason}".format(
                probe=probe,
                reasoning_present=summary.get("reasoning_present"),
                reasoning_tokens=summary.get("reasoning_tokens"),
                provider=summary.get("provider"),
                response_model=summary.get("response_model"),
                finish_reason=summary.get("finish_reason"),
            )
        )
    print(f"recommendation_mode={recommendation['mode']}")
    print(f"base_canonical_name={recommendation['base_canonical_name']}")
    print(f"base_run_flags={recommendation.get('base_run_flags')}")
    print(f"reasoning_canonical_name={recommendation.get('reasoning_canonical_name')}")
    print(f"reasoning_run_flags={recommendation.get('reasoning_run_flags')}")
    print(f"notes={recommendation['notes']}")


def main(argv: Optional[List[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)

    catalog = ModelCatalog(args.catalog)
    canonical_name, provider_name, api_model = resolve_catalog_entry(
        catalog,
        args.canonical_name,
        args.provider,
        args.model,
    )
    base_name = canonical_name or args.canonical_name or api_model

    catalog_entry = catalog.get_model(base_name) if base_name else None
    base_overrides: Dict[str, Any] = dict(catalog_entry.get_request_overrides()) if catalog_entry else {}

    results: List[Dict[str, Any]] = []
    results.append(
        run_probe(
            probe="default",
            prompt=args.prompt,
            provider_name=provider_name,
            api_model=api_model,
            base_overrides=base_overrides,
            system_prompt=args.system_prompt,
            force_subprovider=args.force_subprovider,
            timeout_seconds=args.timeout_seconds,
        )
    )
    results.append(
        run_probe(
            probe="reasoning",
            prompt=args.prompt,
            provider_name=provider_name,
            api_model=api_model,
            base_overrides=base_overrides,
            system_prompt=args.system_prompt,
            force_subprovider=args.force_subprovider,
            timeout_seconds=args.timeout_seconds,
        )
    )

    if reasoning_present(results[-1]) is False:
        results.append(
            run_probe(
                probe="reasoning-medium",
                prompt=args.prompt,
                provider_name=provider_name,
                api_model=api_model,
                base_overrides=base_overrides,
                system_prompt=args.system_prompt,
                force_subprovider=args.force_subprovider,
                timeout_seconds=args.timeout_seconds,
            )
        )

    if reasoning_present(results[0]) is True:
        results.append(
            run_probe(
                probe="no-reasoning",
                prompt=args.prompt,
                provider_name=provider_name,
                api_model=api_model,
                base_overrides=base_overrides,
                system_prompt=args.system_prompt,
                force_subprovider=args.force_subprovider,
                timeout_seconds=args.timeout_seconds,
            )
        )

    recommendation = recommend_configuration(base_name, results)
    payload = {
        "canonical_name": base_name,
        "provider": provider_name,
        "api_model": api_model,
        "prompt": args.prompt,
        "results": results,
        "recommendation": recommendation,
    }

    if args.json:
        print(json.dumps(payload, ensure_ascii=False))
        return

    print_human_report(
        canonical_name=base_name,
        provider_name=provider_name,
        api_model=api_model,
        prompt=args.prompt,
        results=results,
        recommendation=recommendation,
    )


if __name__ == "__main__":
    main()
