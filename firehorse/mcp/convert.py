from __future__ import annotations

import re

from mcp.types import Tool, TextContent, ImageContent

from openreward.api.environments.types import ToolSpec, ToolOutput


_OR_REWARD_RE = re.compile(r'(?:\\n|\s)*\[OR_REWARD:\{[^}]+\}\]')


def strip_or_reward_marker(text: str) -> str:
    """Remove [OR_REWARD:{...}] markers injected by the MCP bridge.

    The bridge appends these so agents can round-trip the reward/finished
    signal through stdio. Once parsed into structured fields, the marker is
    a transport artifact and should not appear in persisted traces.
    Handles both raw text and JSON-escaped forms (for stripping raw
    stream-json lines before writing to disk).
    """
    return _OR_REWARD_RE.sub("", text)


def toolspec_to_mcp(spec: ToolSpec, description_override: str | None = None) -> Tool:
    return Tool(
        name=spec.name,
        description=description_override if description_override is not None else spec.description,
        inputSchema=dict(spec.input_schema) if spec.input_schema else {"type": "object", "properties": {}},
    )


def tooloutput_to_mcp(output: ToolOutput) -> list[TextContent | ImageContent]:
    contents: list[TextContent | ImageContent] = []
    for block in output.blocks:
        if block.type == "text":
            contents.append(TextContent(type="text", text=block.text))
        elif block.type == "image":
            contents.append(ImageContent(
                type="image",
                data=block.data,
                mimeType=block.mimeType,
            ))
    return contents
