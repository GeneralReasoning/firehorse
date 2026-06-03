"""Shared prompt fragments injected by agent harnesses.

Each agent CLI exposes MCP tools under a different prefix
(``mcp__openreward__`` for claude-code, ``openreward_`` for gemini,
``mcp_openreward_`` for hermes), so helpers here take the prefix as a
parameter rather than hard-coding it.
"""
from __future__ import annotations

# Canonical sandbox/system tool names. When the env exposes any of these
# (case-insensitive), we tell the agent that its built-in Bash/Read/Write/...
# operate on the host and won't affect the sandbox.
_SANDBOX_TOOL_HINTS = {
    "bash", "shell", "terminal", "run_shell_command", "execute_command",
    "read", "read_file", "write", "write_file",
    "edit", "edit_file", "patch", "apply_patch",
    "grep", "ripgrep", "search_files",
    "glob", "find", "ls", "list_directory",
}


def build_env_preference_rule(
    env_tool_names: list[str],
    mcp_prefix: str,
) -> str:
    """Build a prompt fragment telling the agent to prefer env tools.

    Always emits a soft directive: only env tools change task state, built-in
    tools are useful for thinking but don't register actions. When the env
    advertises canonical sandbox/system tools, adds a stronger warning that
    built-in shell/file tools target the host rather than the sandbox.
    """
    base = (
        "# Tool Preference\n"
        "\n"
        f"Only `{mcp_prefix}*` tool calls affect the task environment or "
        "count toward your reward. Your built-in tools may help you reason, "
        "plan, or compute, but they do NOT modify task state — call the "
        "environment's tools to take action."
    )
    env_lower = {n.lower() for n in env_tool_names}
    if env_lower & _SANDBOX_TOOL_HINTS:
        return (
            base
            + "\n\n"
            "This environment runs in a sandbox. Your built-in Bash, Read, "
            "Write, Edit, Grep, Glob, and code-execution tools operate on the "
            "host machine — they cannot see or modify the sandbox. If you "
            f"need to read, write, or run code, use the `{mcp_prefix}*` "
            "equivalents instead."
        )
    return base
