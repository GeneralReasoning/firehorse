"""Codex-specific tool descriptions for MCP-overridden tools.

Only bash is needed — the harness excludes other filesystem tools
(read, write, edit, grep, glob) for Codex via _compute_codex_excluded_tools.

The bash description is the real upstream Codex shell_command description from
codex-rs/tools/src/local_tool.rs (v0.118).

Source: https://github.com/openai/codex (codex-rs/)
"""
from __future__ import annotations

CODEX_DESCRIPTIONS: dict[str, str] = {
    "bash": (
        "Runs a shell command and returns its output. "
        "Always set workdir param; avoid cd unless absolutely necessary."
    ),
}
