"""Shape Codex Responses tools and request output options.

This module constructs wire payload fields without opening a connection.
"""

from __future__ import annotations

import copy
from typing import Any


def lite_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """tools/tool_spec.rs::create_tools_json_for_responses_lite."""
    namespace = {
        "type": "namespace",
        "name": "functions",
        "description": "",
        "tools": [],
    }
    result, index = [], None
    for tool in copy.deepcopy(tools):
        if tool["type"] in {"function", "custom"}:
            namespace["tools"].append(tool)
        elif tool["type"] == "namespace" and tool["name"] == "functions":
            if tool.get("description", "").strip():
                namespace["description"] = tool["description"]
            namespace["tools"].extend(tool["tools"])
        else:
            result.append(tool)
            continue
        if index is None:
            index = len(result)
    if index is not None and namespace["tools"]:
        result.insert(index, namespace)
    return result


def apply_output_options(
    payload: dict[str, Any],
    settings: dict[str, Any],
    info: dict[str, Any],
    reasoning: dict[str, Any],
    *,
    is_openai: bool,
) -> None:
    verbosity = (
        settings.get("model_verbosity") or info.get("default_verbosity")
        if info["support_verbosity"]
        else None
    )
    text = {}
    if verbosity is not None:
        text["verbosity"] = verbosity
    if settings.get("output_schema") is not None:
        text["format"] = {
            "type": "json_schema",
            "strict": settings.get("output_schema_strict", True),
            "schema": settings["output_schema"],
            "name": "codex_output_schema",
        }
    if text:
        payload["text"] = text
    service_tier = settings.get("service_tier")
    if service_tier != "default" and any(
        tier["id"] == service_tier for tier in info["service_tiers"]
    ):
        payload["service_tier"] = service_tier
    features = settings.get("features", {})
    if (
        features.get("concurrent_reasoning_summaries")
        and is_openai
        and "summary" in reasoning
    ):
        payload["stream_options"] = {"reasoning_summary_delivery": "sequential_cutoff"}
