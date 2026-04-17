"""Pretty-print JSONL trajectory files using rich."""
from __future__ import annotations

import json
import os
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text


# File extension to language mapping for syntax highlighting
_EXT_TO_LANG = {
    ".py": "python", ".js": "javascript", ".ts": "typescript",
    ".r": "r", ".R": "r", ".rb": "ruby", ".go": "go",
    ".rs": "rust", ".c": "c", ".cpp": "cpp", ".h": "c",
    ".java": "java", ".sh": "bash", ".bash": "bash",
    ".json": "json", ".yaml": "yaml", ".yml": "yaml",
    ".toml": "toml", ".sql": "sql", ".html": "html",
    ".css": "css", ".xml": "xml", ".md": "markdown",
}


def _truncate(text: str, limit: int, full: bool, tail: int = 150) -> tuple[str, int]:
    """Truncate text keeping start and end. Returns (text, chars_truncated)."""
    if full or len(text) <= limit:
        return text, 0
    head = limit - tail
    if head < 100:
        head = limit
        tail = 0
    middle_len = len(text) - head - tail
    if tail > 0:
        return f"{text[:head]}\n\n[...{middle_len} chars truncated...]\n\n{text[-tail:]}", middle_len
    return text[:limit], len(text) - limit


def _lang_for_tool(tool_name: str, arguments: dict | None = None) -> str:
    """Guess syntax language from tool name and arguments."""
    if tool_name == "bash":
        return "bash"
    if tool_name in ("write", "edit", "read") and arguments:
        path = arguments.get("file_path", "") or arguments.get("path", "")
        ext = os.path.splitext(path)[1]
        return _EXT_TO_LANG.get(ext, "text")
    return "json"


def _render_openreward_prompt(console: Console, event: dict, full: bool) -> None:
    system = event.get("system_prompt", "")
    prompt = event.get("environment_prompt", "")

    if system and system != "default":
        text, trunc = _truncate(system, 300, full)
        console.print(Panel(Text(text), title="System Prompt", border_style="dim"))

    if prompt:
        text, trunc = _truncate(prompt, 1000, full)
        console.print(Panel(Text(text), title="Task", border_style="blue"))


def _render_system_or_user(console: Console, event: dict, full: bool) -> None:
    role = event.get("type", "user").capitalize()
    content_str = event.get("content", "")
    text, trunc = _truncate(content_str, 500, full)
    style = "blue" if role == "User" else "dim"
    console.print(Panel(Text(text), title=role, border_style=style))


def _render_assistant(console: Console, event: dict, full: bool) -> None:
    # Build subtitle with token info
    parts = []
    input_tok = event.get("input_tokens")
    output_tok = event.get("output_tokens")
    if input_tok is not None:
        parts.append(f"{input_tok:,} in")
    if output_tok is not None:
        parts.append(f"{output_tok:,} out")
    if event.get("cache_hit"):
        parts.append("cache hit")

    subtitle = " | ".join(parts) if parts else None

    # Text content
    text_content = event.get("text_content") or ""
    reasoning = event.get("reasoning_content") or ""

    # Tool calls
    tool_calls = event.get("tool_calls", [])

    # For react agents, content is in "raw" field
    raw = event.get("raw")
    if raw and isinstance(raw, dict):
        # Anthropic react format: raw is {"role": "assistant", "content": [...]}
        raw_content = raw.get("content")
        if isinstance(raw_content, list):
            for block in raw_content:
                if isinstance(block, dict):
                    if block.get("type") == "text" and not text_content:
                        text_content = block.get("text", "")
                    elif block.get("type") == "tool_use":
                        tool_calls.append({
                            "name": block.get("name", ""),
                            "arguments": block.get("input", {}),
                        })
                    elif block.get("type") == "thinking" and not reasoning:
                        reasoning = block.get("thinking", "")
        elif isinstance(raw_content, str) and not text_content:
            text_content = raw_content
        # OpenRouter react format: raw is {"role": "assistant", "content": "...", "tool_calls": [...]}
        if raw.get("tool_calls") and not tool_calls:
            for tc in raw["tool_calls"]:
                func = tc.get("function", {})
                args = func.get("arguments", "{}")
                try:
                    args = json.loads(args) if isinstance(args, str) else args
                except json.JSONDecodeError:
                    pass
                tool_calls.append({"name": func.get("name", ""), "arguments": args})
            if raw.get("content") and not text_content:
                text_content = raw["content"]
    elif raw and isinstance(raw, list):
        # OpenAI react format: raw is a list of output items
        for item in raw:
            if isinstance(item, dict):
                if item.get("type") == "function_call":
                    args = item.get("arguments", "{}")
                    try:
                        args = json.loads(args) if isinstance(args, str) else args
                    except json.JSONDecodeError:
                        pass
                    tool_calls.append({
                        "name": item.get("name", ""),
                        "arguments": args,
                    })
                elif item.get("type") == "message" or item.get("role") == "assistant":
                    if item.get("content") and not text_content:
                        text_content = item["content"]

    lines = []

    if reasoning:
        r_text, _ = _truncate(reasoning, 300, full)
        lines.append(Text.assemble(("Thinking: ", "dim italic"), (r_text, "dim")))

    if text_content:
        t_text, _ = _truncate(text_content, 500, full)
        lines.append(Text(t_text))

    if tool_calls:
        for tc in tool_calls:
            name = tc.get("name", "?")
            args = tc.get("arguments", {})
            if isinstance(args, str):
                args_str = args
            else:
                args_str = json.dumps(args, indent=2)
            a_text, a_trunc = _truncate(args_str, 200, full)
            lines.append(Text.assemble(
                ("Tool: ", "bold"),
                (name, "cyan bold"),
                ("(", ""),
                (a_text, ""),
                (")" if not a_trunc else f"...) [{a_trunc} more chars]", "dim" if a_trunc else ""),
            ))

    if not lines:
        lines.append(Text("[no content]", style="dim"))

    from rich.console import Group
    content = Group(*lines)
    console.print(Panel(content, title="Assistant", subtitle=subtitle, border_style="green"))


def _render_tool_call(console: Console, event: dict, full: bool) -> None:
    name = event.get("tool_name", "?")
    args = event.get("arguments", {})

    lang = _lang_for_tool(name, args if isinstance(args, dict) else None)

    if isinstance(args, dict):
        if name == "bash" and "command" in args:
            code = args["command"]
            lang = "bash"
        elif name in ("write",) and "content" in args:
            path = args.get("file_path", "")
            ext = os.path.splitext(path)[1]
            lang = _EXT_TO_LANG.get(ext, "text")
            code = args["content"]
            name = f"{name}: {path}"
        else:
            code = json.dumps(args, indent=2)
            lang = "json"
    else:
        code = str(args)

    code, trunc = _truncate(code, 500, full)
    syntax = Syntax(code, lang, theme="monokai", word_wrap=True)

    content = syntax

    console.print(Panel(content, title=f"[cyan]{name}[/]", border_style="cyan"))


def _render_tool_result(console: Console, event: dict, full: bool) -> None:
    reward = event.get("reward")
    finished = event.get("finished", False)
    output = event.get("output", "")

    # React agents put output in "raw" field
    if not output:
        raw = event.get("raw")
        if isinstance(raw, dict):
            output = raw.get("output", "") or raw.get("content", "")

    title_parts = ["Tool Result"]
    if reward is not None:
        title_parts.append(f"reward={reward}")
    if finished:
        title_parts.append("FINISHED")

    title = " | ".join(title_parts)
    border = "yellow" if not finished else "bold green"

    text, _ = _truncate(str(output), 500, full)
    console.print(Panel(Text(text), title=title, border_style=border))


def _render_compaction(console: Console, event: dict) -> None:
    method = event.get("method", event.get("type", "compaction"))
    count = event.get("compaction_count", "")
    msg_count = event.get("original_message_count", "?")
    new_count = event.get("new_message_count", "?")
    proactive = " (proactive)" if event.get("proactive") else ""

    text = f"{method}{proactive}: {msg_count} -> {new_count} messages"
    if count:
        text += f" (compaction #{count})"

    console.print(Panel(Text(text, style="dim"), title="Compaction", border_style="dim"))


def _render_summary(console: Console, event: dict) -> None:
    reward = event.get("final_reward")
    finished = event.get("finished", False)
    usage = event.get("usage", {})

    lines = []
    if reward is not None:
        lines.append(f"Reward: {reward}")
    lines.append(f"Finished: {finished}")
    if usage:
        input_tok = usage.get("input_tokens")
        output_tok = usage.get("output_tokens")
        duration = usage.get("duration_ms")
        if input_tok is not None:
            lines.append(f"Input tokens: {input_tok:,}")
        if output_tok is not None:
            lines.append(f"Output tokens: {output_tok:,}")
        if duration is not None:
            lines.append(f"Duration: {duration / 1000:.1f}s")

    env = event.get("env", "")
    model = event.get("model", "")
    if env:
        lines.append(f"Environment: {env}")
    if model:
        lines.append(f"Model: {model}")

    content = Text("\n".join(lines))
    style = "bold green" if finished else "bold yellow"
    console.print(Panel(content, title="Summary", border_style=style))


def view_trace(path: Path, full: bool = False) -> None:
    """Pretty-print a JSONL trajectory file."""
    console = Console()

    if not path.exists():
        console.print(f"[red]File not found: {path}[/]")
        return

    events = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    if not events:
        console.print(f"[yellow]No events found in {path}[/]")
        return

    console.print(f"\n[bold]Trace: {path.name}[/] ({len(events)} events)\n")

    for event in events:
        etype = event.get("type", "unknown")

        if etype == "openreward_prompt":
            _render_openreward_prompt(console, event, full)
        elif etype in ("system", "user"):
            _render_system_or_user(console, event, full)
        elif etype == "assistant":
            _render_assistant(console, event, full)
        elif etype == "tool_call":
            _render_tool_call(console, event, full)
        elif etype == "tool_result":
            _render_tool_result(console, event, full)
        elif etype in ("compaction", "micro_compaction"):
            _render_compaction(console, event)
        elif etype == "openreward_summary":
            _render_summary(console, event)

    console.print()


def view_directory(path: Path, full: bool = False) -> None:
    """View all trial JSONL files in a directory."""
    jsonl_files = sorted(path.glob("trial_*.jsonl"))

    if not jsonl_files:
        console = Console()
        console.print(f"[yellow]No trial_*.jsonl files found in {path}[/]")
        return

    for jsonl_path in jsonl_files:
        view_trace(jsonl_path, full=full)
