from __future__ import annotations

from mcp.types import Tool, TextContent, ImageContent

from openreward.api.environments.types import ToolSpec, ToolOutput


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
