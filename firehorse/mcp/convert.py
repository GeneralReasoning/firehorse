from __future__ import annotations

import json
import re

from mcp.types import Tool, TextContent, ImageContent

from openreward.api.environments.types import ToolSpec, ToolOutput


_OR_REWARD_RE = re.compile(r'(?:\\n|\s)*\[OR_REWARD:(\{[^}]+\})\]')
_EPISODE_COMPLETE_RE = re.compile(r'(?:\\n|\s)*\[EPISODE COMPLETE\][^\n]*')


def strip_or_reward_marker(text: str) -> str:
    """Remove [OR_REWARD:{...}] markers injected by the MCP bridge.

    The bridge appends these so agents can round-trip the reward/finished
    signal through stdio. Once parsed into structured fields, the marker is
    a transport artifact and should not appear in persisted traces.
    Handles both raw text and JSON-escaped forms (for stripping raw
    stream-json lines before writing to disk).
    """
    return _OR_REWARD_RE.sub("", text)


def strip_bridge_markers(text: str) -> str:
    """Remove every bridge-injected control marker from a string.

    Strips both ``[OR_REWARD:{...}]`` and ``[EPISODE COMPLETE]...`` sentences.
    Used by the bridge to scrub env-supplied tool-result text *before* it
    appends its own markers — otherwise a model could ask an env tool to echo
    its input and smuggle a fake reward/termination signal through to the
    harness rollout logger. See ``firehorse/mcp/bridge.py`` for the
    spoofing-mitigation contract.
    """
    return _EPISODE_COMPLETE_RE.sub("", _OR_REWARD_RE.sub("", text))


def parse_or_reward_marker(text: str) -> tuple[float | None, bool]:
    """Parse the bridge's last ``[OR_REWARD:{...}]`` marker out of text.

    Returns ``(reward, is_finished)`` — both default to ``(None, False)`` if no
    marker is found. The bridge always appends its real marker at the end of
    a tool-result text (after stripping any pre-existing markers from
    env-supplied content), so matching the *last* marker gives us the
    bridge-authoritative value rather than anything a model may have injected
    earlier in the string.

    Wrong marker bodies (malformed JSON, missing fields) yield ``(None, False)``
    silently — the alternative is an exception inside every harness's rollout
    logger, which is worse than a missing attribution.
    """
    matches = _OR_REWARD_RE.findall(text)
    if not matches:
        return None, False
    try:
        payload = json.loads(matches[-1])
    except (json.JSONDecodeError, ValueError):
        return None, False
    reward = payload.get("r")
    if reward is not None and not isinstance(reward, (int, float)):
        reward = None
    return reward, bool(payload.get("f", False))


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
