"""Shared helpers for subprocess-based agents (openclaw, hermes)."""
from __future__ import annotations

from typing import Callable

# Env tools that should not be exposed via MCP — the agent's own
# planning/tracking tools are preferred.
ALWAYS_USE_BUILTIN = {"todo_write", "todowrite"}

# Submission tool names to remind the agent about.
SUBMISSION_TOOL_NAMES = {"submit", "answer", "submit_answer"}

# Buffer limit for subprocess stdout/stderr line reading.
SUBPROCESS_LINE_LIMIT = 10 * 1024 * 1024  # 10 MB

McpNameFormatter = Callable[[str, str], str]


def format_mcp_name_double(server: str, tool: str) -> str:
    """OpenClaw-style MCP tool name: ``mcp__server__tool``."""
    return f"mcp__{server}__{tool}"


def format_mcp_name_single(server: str, tool: str) -> str:
    """Hermes-style MCP tool name: ``mcp_server_tool``."""
    return f"mcp_{server}_{tool}"


def sanitize_prompt(text: str) -> str:
    """Ensure prompt text doesn't start with '-' (subprocess arg parsing)."""
    if text.startswith("-"):
        return " " + text
    return text


def build_mcp_prompt_section(
    env_tool_names: list[str],
    format_mcp_name: McpNameFormatter,
    mcp_server_name: str = "openreward",
) -> str:
    """Build the MCP tool instructions block appended to the user prompt."""
    if not env_tool_names:
        return ""

    lines = [
        "",
        "# OpenReward Environment Tools",
        "",
        "You are solving a task in an OpenReward environment. The environment provides "
        "additional tools via an MCP server named 'openreward'. Use your built-in tools "
        "normally for file operations, terminal commands, web search, etc. The MCP tools "
        "below are for environment-specific actions (e.g. submitting answers).",
        "",
        "Available MCP tools:",
    ]
    for name in env_tool_names:
        if name.lower() in ALWAYS_USE_BUILTIN:
            continue
        lines.append(f"- `{format_mcp_name(mcp_server_name, name)}`")

    lines.extend([
        "",
        "## Termination",
        "",
        "When a tool result contains [EPISODE COMPLETE], stop working immediately — "
        "the task is done. Do not make any more tool calls after seeing [EPISODE COMPLETE].",
    ])
    return "\n".join(lines)


def build_submission_reminder(
    env_tool_names: list[str],
    format_mcp_name: McpNameFormatter,
    mcp_server_name: str = "openreward",
) -> str:
    """Build a reminder prompt telling the agent to call submission tools."""
    submission_tools = [
        format_mcp_name(mcp_server_name, name)
        for name in env_tool_names
        if name.lower() in SUBMISSION_TOOL_NAMES
    ]
    if not submission_tools:
        return ""
    tool_list = ", ".join(f"`{t}`" for t in submission_tools)
    return (
        "Submission Reminder:\n"
        f"This environment has a submission tool: {tool_list}. "
        "You MUST call this tool to submit your final answer before you finish. "
        "If you stop without calling it, your work will not be scored."
    )
