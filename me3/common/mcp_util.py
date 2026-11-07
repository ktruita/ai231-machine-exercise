"""Helpers for reading MCP tool results regardless of how the server encoded them."""

from __future__ import annotations

import json
from typing import Any


class ToolError(RuntimeError):
    pass


def tool_value(result) -> Any:
    """Return a tool call's value: structured content if present, else parsed JSON/text."""
    if result.is_error:
        raise ToolError(" ".join(getattr(c, "text", "") for c in result.content))
    if result.structured_content is not None:
        sc = result.structured_content
        return sc["result"] if set(sc) == {"result"} else sc
    texts = [c.text for c in result.content if getattr(c, "type", None) == "text"]
    if len(texts) != 1:
        return texts
    try:
        return json.loads(texts[0])
    except json.JSONDecodeError:
        return texts[0]
