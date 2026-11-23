"""Transport helpers shared by the native harnesses."""

import copy
import json

from flashrsi.core.infra import InfraError


def is_deepseek(model):
    return str(model).split("/")[-1].lower().startswith("deepseek")


async def read_anthropic_sse(response, notify=None, *, require_stop=True):
    """Rebuild the ordered native content blocks, including signed thinking."""
    content = {}
    partial_inputs = {}
    usage = {}
    stop_reason = None
    stopped = False
    events = []
    delta_fields = {
        "text_delta": ("text", "text"),
        "thinking_delta": ("thinking", "thinking"),
        "signature_delta": ("thinking", "signature"),
    }
    async for line in response.aiter_lines():
        if not line.startswith("data:"):
            continue
        try:
            event = json.loads(line[5:])
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        if "error" in event:
            raise InfraError("incomplete_stream")
        events.append(event)
        if notify is not None:
            await notify(event)
        kind = event.get("type")
        index = event.get("index", 0)
        if kind == "content_block_start":
            content[index] = copy.deepcopy(event.get("content_block") or {})
        elif kind == "content_block_delta":
            delta = event.get("delta") or {}
            delta_type = delta.get("type")
            if delta_type == "input_json_delta":
                partial_inputs[index] = partial_inputs.get(index, "") + delta.get("partial_json", "")
            elif delta_type in delta_fields:
                block_type, field = delta_fields[delta_type]
                block = content.setdefault(index, {"type": block_type})
                block[field] = (block.get(field) or "") + (delta.get(field) or "")
        elif kind == "message_start":
            usage.update((event.get("message") or {}).get("usage") or {})
        elif kind == "message_delta":
            stop_reason = (event.get("delta") or {}).get("stop_reason") or stop_reason
            usage.update(event.get("usage") or {})
        elif kind == "message_stop":
            stopped = True
            break
    if not (stopped if require_stop else (stopped or stop_reason)):
        raise InfraError("incomplete_stream")
    for index, value in partial_inputs.items():
        if value:
            content[index]["input"] = json.loads(value)
    return {
        "content": [content[index] for index in sorted(content)],
        "stop_reason": stop_reason,
        "usage": usage,
        "events": events,
    }
