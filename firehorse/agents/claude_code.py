from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

from openreward import (
    AssistantMessage,
    ReasoningItem,
    ToolCall,
    ToolResult,
    UserMessage,
)

from firehorse.agents.base import BaseAgent, AgentResult, TrialContext

# Map of lowercase env tool names -> Claude built-in tool names they should replace
ENV_TO_BUILTIN = {
    "bash": "Bash",
    "read": "Read",
    "write": "Write",
    "edit": "Edit",
    "grep": "Grep",
    "glob": "Glob",
    "notebookedit": "NotebookEdit",
}

# Filesystem built-ins are always disabled — the agent should only access
# files through environment-provided sandbox tools (if any).
FILESYSTEM_BUILTINS = ["Bash", "Read", "Write", "Edit", "Grep", "Glob"]

# Env tools with these names are always excluded from MCP —
# Claude's own built-in versions are preferred for planning/tracking.
ALWAYS_USE_BUILTIN = {"todo_write", "todowrite"}

# Tool names (exact, lowercased) that represent submission/answer tools.
# Used to remind the agent to call them before finishing.
SUBMISSION_TOOL_NAMES = {"submit", "answer", "submit_answer"}

# Buffer limit for subprocess stdout/stderr line reading.
# asyncio defaults to 64KB which is too small for large JSONL lines
# (e.g. thinking blocks, large tool outputs).
_SUBPROCESS_LINE_LIMIT = 10 * 1024 * 1024  # 10 MB

def _openrouter_env(or_key: str) -> dict[str, str]:
    """Build env vars to route Claude Code through OpenRouter's Anthropic-compatible endpoint."""
    return {
        "ANTHROPIC_BASE_URL": "https://openrouter.ai/api",
        "ANTHROPIC_AUTH_TOKEN": or_key,
        "ANTHROPIC_API_KEY": or_key,
    }


def _require_openrouter_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        raise ValueError("OPENROUTER_API_KEY environment variable is required")
    return key


def _resolve_model(model: str, provider_url: str | None) -> tuple[str, dict[str, str], bool]:
    """Map model string to Claude CLI model flag, extra env vars, and whether it's Anthropic.

    Returns (model_name, extra_env, is_anthropic).

    Supported prefixes:
      anthropic/   — Direct Anthropic API (requires ANTHROPIC_API_KEY)
      openrouter/  — Any model via OpenRouter (requires OPENROUTER_API_KEY)
      openai/      — OpenAI models via OpenRouter (requires OPENROUTER_API_KEY)
      google/      — Google Gemini models via OpenRouter (requires OPENROUTER_API_KEY)

    Claude Code speaks the Anthropic Messages API. Non-Anthropic models must go
    through a proxy that accepts Anthropic-format requests (e.g. OpenRouter).
    """
    if model.startswith("anthropic/"):
        return model.split("/", 1)[1], {}, True

    elif model.startswith("openrouter/"):
        name = model.split("/", 1)[1]
        or_key = _require_openrouter_key()
        is_anthropic = name.startswith("anthropic/")
        return name, _openrouter_env(or_key), is_anthropic

    elif provider_url:
        # Custom Anthropic-compatible proxy (e.g. LiteLLM, vLLM with Anthropic support)
        return model, {"ANTHROPIC_BASE_URL": provider_url}, False

    else:
        raise ValueError(
            f"Model {model!r} needs a provider prefix or --provider-url.\n"
            f"  Anthropic:    anthropic/claude-sonnet-4-5\n"
            f"  OpenRouter:   openrouter/{model}\n"
            f"  OpenAI (OR):  openrouter/openai/gpt-4.1\n"
            f"  Google (OR):  openrouter/google/gemini-2.5-pro\n"
            f"  Custom proxy: --provider-url https://my-proxy/v1"
        )


def _compute_disallowed_builtins(env_tool_names: list[str]) -> list[str]:
    """Given environment tool names, return built-in tools that should be disabled.

    Filesystem built-ins are always disabled — the agent should only access
    files through environment-provided sandbox tools (if any). Non-filesystem
    built-ins (WebSearch, WebFetch, TodoWrite, etc.) remain available.
    """
    disallow = list(FILESYSTEM_BUILTINS)
    # Also disable non-filesystem built-ins that the env replaces (e.g. NotebookEdit)
    for env_name in env_tool_names:
        builtin = ENV_TO_BUILTIN.get(env_name.lower())
        if builtin and builtin not in disallow:
            disallow.append(builtin)
    return disallow


def _build_tool_mapping_prompt(env_tool_names: list[str], mcp_server_name: str = "openreward") -> str:
    """Build system prompt section listing available MCP tools and mapping built-in names.

    Claude Code's default system prompt references built-in tools (Read, Edit, etc.)
    that are disabled when the environment provides replacements via MCP. This prompt
    section tells the model the actual MCP tool names to use.
    """
    if not env_tool_names:
        return ""

    lines = [
        "# Available Tools",
        "",
        "The following tools are available via the MCP server. Use these tool names exactly.",
        "The built-in file/shell tools (Read, Edit, Write, Bash, Grep, Glob) have been replaced",
        "by MCP equivalents below. Other built-in tools (WebSearch, WebFetch, TodoWrite, Agent, etc.)",
        "remain available as normal.",
        "",
    ]

    mapped = []
    other = []
    for env_name in env_tool_names:
        if env_name.lower() in ALWAYS_USE_BUILTIN:
            continue
        mcp_name = f"mcp__{mcp_server_name}__{env_name}"
        builtin = ENV_TO_BUILTIN.get(env_name.lower())
        if builtin:
            mapped.append(f"- `{mcp_name}` — use this instead of `{builtin}`")
        else:
            other.append(f"- `{mcp_name}`")

    if mapped:
        lines.append("Built-in replacements:")
        lines.extend(mapped)
        lines.append("")

    if other:
        lines.append("Additional environment tools:")
        lines.extend(other)
        lines.append("")

    lines.append("# Important")
    lines.append("")
    lines.append("Do NOT use ToolSearch to find environment tools. ToolSearch only indexes")
    lines.append("Claude Code's built-in deferred tools and cannot find MCP tools.")
    lines.append("The MCP tools listed above are your only environment tools. If they return")
    lines.append("'No such tool' errors, the MCP server failed to connect — report the error")
    lines.append("instead of searching for alternatives.")
    lines.append("")

    return "\n".join(lines)


def _build_submission_reminder(env_tool_names: list[str], mcp_server_name: str = "openreward") -> str:
    """Build a prompt reminder telling the agent to call submission tools before finishing.

    Returns empty string if no submission tools are found.
    """
    submission_tools = []
    for name in env_tool_names:
        if name.lower() in SUBMISSION_TOOL_NAMES:
            submission_tools.append(f"mcp__{mcp_server_name}__{name}")

    if not submission_tools:
        return ""

    tool_list = ", ".join(f"`{t}`" for t in submission_tools)
    return (
        f"Submission Reminder:\n"
        f"This environment has a submission tool: {tool_list}. "
        f"You MUST call this tool to submit your final answer before you finish. "
        f"If you stop without calling it, your work will not be scored. "
        f"Do not assume that writing to a file or printing output is sufficient — "
        f"you must explicitly call the submission tool."
    )


def _sanitize_prompt(text: str) -> str:
    """Ensure prompt text doesn't start with '-' which CLI parsers may interpret as a flag."""
    if text.startswith("-"):
        return " " + text
    return text


def _log_event_to_rollout(event: dict, rollout: Any) -> None:
    """Parse a Claude stream-json event and log it to an OpenReward rollout."""
    event_type = event.get("type")
    msg = event.get("message", {})
    if not isinstance(msg, dict):
        return

    content_blocks = msg.get("content", [])
    if not isinstance(content_blocks, list):
        return

    for block in content_blocks:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")

        if event_type == "assistant":
            if btype == "text":
                text = block.get("text", "")
                if text:
                    rollout.log(AssistantMessage(content=text))
            elif btype == "thinking":
                thinking = block.get("thinking", "")
                summary = block.get("summary", "")
                if thinking:
                    rollout.log(ReasoningItem(content=thinking, summary=summary))
            elif btype == "redacted_thinking":
                # OpenRouter encodes reasoning as base64 (openrouter.reasoning:<b64>).
                # Anthropic models use encrypted data that can't be decoded.
                # Try to decode; fall back to opaque marker.
                data = block.get("data", "")
                content = "[redacted thinking]"
                if data.startswith("openrouter.reasoning:"):
                    try:
                        decoded = json.loads(base64.b64decode(data.split(":", 1)[1]))
                        content = decoded.get("text", content)
                    except (ValueError, json.JSONDecodeError):
                        pass
                if content:
                    rollout.log(ReasoningItem(content=content))
            elif btype == "tool_use":
                rollout.log(ToolCall(
                    name=block.get("name", ""),
                    content=json.dumps(block.get("input", {})),
                    call_id=block.get("id", ""),
                ))

        elif event_type == "user":
            if btype == "tool_result":
                content = block.get("content", "")
                if isinstance(content, list):
                    content = "\n".join(
                        b.get("text", "") for b in content if isinstance(b, dict)
                    )
                content_str = str(content)[:10000]

                # Extract reward/finished from [OR_REWARD:{"r":...,"f":...}] tag
                reward = None
                is_finished = False
                m = re.search(r'\[OR_REWARD:(\{[^}]+\})\]', content_str)
                if m:
                    try:
                        rd = json.loads(m.group(1))
                        reward = rd.get("r")
                        is_finished = rd.get("f", False)
                    except (json.JSONDecodeError, AttributeError):
                        pass

                rollout.log(
                    ToolResult(
                        content=content_str,
                        call_id=block.get("tool_use_id", ""),
                    ),
                    reward=reward,
                    is_finished=is_finished,
                )


class ClaudeCodeAgent(BaseAgent):

    @property
    def name(self) -> str:
        return "claude-code"

    async def setup(self) -> None:
        proc = await asyncio.create_subprocess_exec(
            "claude", "--version",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(
                "Claude Code CLI not found. Install via: npm install -g @anthropic-ai/claude-code"
            )
        version = stdout.decode().strip()
        print(f"Claude Code version: {version}", file=sys.stderr)

    async def run(self, ctx: TrialContext) -> AgentResult:
        with tempfile.TemporaryDirectory(prefix="orwd-trial-") as tmpdir:
            tmppath = Path(tmpdir)
            result_file = tmppath / "result.json"
            trial_id = ctx.task_spec.get("id", ctx.task_spec.get("index", ctx.task_index if ctx.task_index is not None else "unknown"))
            log_dir = Path(ctx.output_dir) if ctx.output_dir else None

            env_tool_names = [t.name for t in ctx.tools]

            # Build MCP config
            session_task = ctx.session.task
            mcp_env: dict[str, str] = {
                "OPENREWARD_API_KEY": os.environ.get("OPENREWARD_API_KEY", ""),
                "OPENREWARD_ENV_NAME": ctx.env_name,
                "OPENREWARD_TASK_SPEC": json.dumps(dict(ctx.task_spec)),
                "OPENREWARD_TASK_SERVER_NAME": session_task.server_name,
                "OPENREWARD_TASK_ENV_NAME": session_task.environment_name,
                "OPENREWARD_TASK_NAMESPACE": session_task.namespace or "",
                "OPENREWARD_RESULT_FILE": str(result_file),
            }
            if ctx.toolset_name:
                mcp_env["OPENREWARD_TOOLSET_NAME"] = ctx.toolset_name
            else:
                mcp_env["OPENREWARD_TOOL_DESCRIPTIONS"] = "claude" if ctx.use_builtin_descriptions else "env"
            if os.environ.get("OPENREWARD_URL"):
                mcp_env["OPENREWARD_URL"] = os.environ["OPENREWARD_URL"]
            if ctx.secrets:
                mcp_env["OPENREWARD_SESSION_SECRETS"] = json.dumps(ctx.secrets)

            # Rewards sidecar JSONL
            if log_dir:
                rewards_path = log_dir / f"trial_{trial_id}_rewards.jsonl"
                mcp_env["OPENREWARD_REWARDS_FILE"] = str(rewards_path)

            # Exclude env tools that duplicate Claude's built-in planning tools
            exclude_tools = [n for n in env_tool_names if n.lower() in ALWAYS_USE_BUILTIN]
            if exclude_tools:
                mcp_env["OPENREWARD_EXCLUDE_TOOLS"] = ",".join(exclude_tools)

            mcp_config = {
                "mcpServers": {
                    "openreward": {
                        "command": sys.executable,
                        "args": ["-m", "firehorse.mcp"],
                        "env": mcp_env,
                    }
                }
            }
            mcp_config_path = tmppath / "mcp.json"
            mcp_config_path.write_text(json.dumps(mcp_config))

            # Resolve model and provider
            model_name, extra_env, is_anthropic = _resolve_model(ctx.model, ctx.provider_url)

            # Compute which built-in tools to disable
            auto_disallow = _compute_disallowed_builtins(env_tool_names)
            all_disallow = list(set(auto_disallow + ctx.disable_builtin_tools))

            if auto_disallow:
                print(
                    f"[claude-code] Disabling built-in tools: "
                    f"{', '.join(sorted(auto_disallow))}",
                    file=sys.stderr,
                )

            # Termination instructions appended to user prompt
            termination_instructions = (
                "When a tool result contains [EPISODE COMPLETE], stop working immediately — the task is done. "
                "Do not make any more tool calls after seeing [EPISODE COMPLETE]."
            )

            # Build command
            cmd = [
                "claude",
                "--print",
                "--verbose",
                "--output-format", "stream-json",
                "--model", model_name,
                "--mcp-config", str(mcp_config_path),
                "--permission-mode", "plan" if ctx.plan_mode else "bypassPermissions",
                "--no-session-persistence",
            ]

            # Only add --effort for Anthropic models (extended thinking is Anthropic-specific)
            if is_anthropic:
                cmd.extend(["--effort", ctx.effort])

            if all_disallow:
                cmd.extend(["--disallowed-tools", ",".join(all_disallow)])

            # Append tool name mapping to system prompt so the model uses MCP names
            tool_mapping_prompt = _build_tool_mapping_prompt(env_tool_names)
            if tool_mapping_prompt:
                cmd.extend(["--append-system-prompt", tool_mapping_prompt])

            if ctx.max_turns:
                budget = max(1.0, ctx.max_turns * 0.10)
                cmd.extend(["--max-budget-usd", str(budget)])

            submission_reminder = _build_submission_reminder(env_tool_names)
            full_prompt = f"{ctx.prompt_text}\n\nTermination Instructions:\n{termination_instructions}"
            if submission_reminder:
                full_prompt = f"{full_prompt}\n\n{submission_reminder}"
            full_prompt = _sanitize_prompt(full_prompt)
            cmd.extend(["-p", full_prompt])
            proc_env = {**os.environ, **extra_env}

            print(f"[claude-code] Launching with model={model_name}", file=sys.stderr)

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=proc_env,
                limit=_SUBPROCESS_LINE_LIMIT,
            )

            # --- Set up JSONL logging ---
            main_log = None
            subagent_logs: dict[str, Any] = {}  # parent_tool_use_id -> file handle

            if log_dir:
                main_log = open(log_dir / f"trial_{trial_id}.jsonl", "w")

            def _get_log(event: dict):
                if main_log is None:
                    return None
                parent = event.get("parent_tool_use_id")
                if parent:
                    if parent not in subagent_logs:
                        subagent_logs[parent] = open(
                            log_dir / f"trial_{trial_id}_subagent_{parent}.jsonl", "w"
                        )
                    return subagent_logs[parent]
                return main_log

            # --- Set up OpenReward rollout logging ---
            main_rollout = None
            subagent_rollouts: dict[str, Any] = {}  # parent_tool_use_id -> Rollout

            if ctx.logging and ctx.rollout_client:
                try:
                    main_rollout = ctx.rollout_client.rollout.create(
                        run_name=ctx.run_name,
                        rollout_name=f"trial_{trial_id}",
                        environment=ctx.env_name,
                        split=ctx.split,
                        task_spec=ctx.task_spec,
                        metadata={"model": ctx.model, "agent": "claude-code", "effort": ctx.effort},
                    )
                    print(f"[claude-code] Rollout: https://openreward.ai/rollout/{main_rollout.event_id}", file=sys.stderr)
                except Exception as e:
                    print(f"[claude-code] Failed to create rollout: {e}", file=sys.stderr)

            def _get_rollout(event: dict):
                if main_rollout is None:
                    return None
                parent = event.get("parent_tool_use_id")
                if parent:
                    if parent not in subagent_rollouts and ctx.rollout_client:
                        subagent_rollouts[parent] = ctx.rollout_client.rollout.create(
                            run_name=ctx.run_name,
                            rollout_name=f"trial_{trial_id}_subagent_{parent}",
                            environment=ctx.env_name,
                            metadata={
                                "parent_rollout": main_rollout.event_id,
                                "parent_tool_use_id": parent,
                            },
                        )
                    return subagent_rollouts.get(parent)
                return main_rollout

            # --- Log prompt ---
            # System prompt is Claude Code's default (not captured here).
            # Termination instructions are appended to the user prompt.
            prompt_event = {
                "type": "openreward_prompt",
                "system_prompt": "default",
                "environment_prompt": full_prompt,
            }
            if main_log:
                main_log.write(json.dumps(prompt_event) + "\n")

            if main_rollout:
                main_rollout.log(UserMessage(content=full_prompt))

            # --- Read stdout and stderr concurrently ---
            turns_used = 0
            stdout_lines: list[str] = []
            result_event: dict | None = None  # Claude's final 'result' event
            mcp_failed = False
            mcp_failed_tool_calls = 0
            mcp_tool_use_ids: dict[str, str] = {}  # tool_use_id -> mcp tool name
            mcp_result_count = 0

            async def read_stdout():
                nonlocal turns_used, result_event, mcp_failed, mcp_failed_tool_calls, mcp_result_count
                assert proc.stdout is not None
                async for line in proc.stdout:
                    line_str = line.decode(errors="replace").strip()
                    if not line_str:
                        continue
                    stdout_lines.append(line_str)
                    try:
                        event = json.loads(line_str)
                    except json.JSONDecodeError:
                        if main_log:
                            main_log.write(line_str + "\n")
                        continue

                    # Write to JSONL
                    log_file = _get_log(event)
                    if log_file:
                        log_file.write(line_str + "\n")

                    # Log to OpenReward rollout
                    rollout = _get_rollout(event)
                    if rollout:
                        _log_event_to_rollout(event, rollout)

                    # Detect MCP server failure from system.init event
                    if event.get("type") == "system" and event.get("subtype") == "init":
                        for srv in event.get("mcp_servers", []):
                            if srv.get("name") == "openreward" and srv.get("status") != "connected":
                                mcp_failed = True
                                print(
                                    f"[claude-code] MCP 'openreward' status={srv.get('status')}"
                                    f" — will abort after first futile calls",
                                    file=sys.stderr,
                                )

                    # Track the result event (has cost/token data)
                    if event.get("type") == "result":
                        result_event = event

                    # Track tool use events
                    if event.get("type") == "assistant" and isinstance(event.get("message"), dict):
                        content = event["message"].get("content", [])
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "tool_use":
                                turns_used += 1
                                # Track MCP tool_use_ids for reward correlation
                                tool_name = block.get("name", "")
                                tool_use_id = block.get("id", "")
                                if tool_name.startswith("mcp__openreward__"):
                                    mcp_tool_use_ids[tool_use_id] = tool_name
                                # Abort if MCP failed and agent is making futile calls
                                if mcp_failed:
                                    mcp_failed_tool_calls += 1
                                    if mcp_failed_tool_calls >= 2:
                                        print(
                                            "[claude-code] MCP failed, agent making futile calls — terminating",
                                            file=sys.stderr,
                                        )
                                        proc.kill()
                                        return

                    # Annotate MCP tool results with call_count for reward file correlation.
                    # Only count successful calls (is_error=False) since the bridge only
                    # increments call_count on success — errors (validation, post-episode)
                    # are returned with isError=True and don't increment.
                    if event.get("type") == "user" and log_file:
                        msg_content = (event.get("message") or {}).get("content", [])
                        if isinstance(msg_content, list):
                            for block in msg_content:
                                if isinstance(block, dict) and block.get("type") == "tool_result":
                                    tuid = block.get("tool_use_id", "")
                                    if tuid not in mcp_tool_use_ids:
                                        continue
                                    is_error = block.get("is_error", False)
                                    if not is_error:
                                        mcp_result_count += 1
                                    annotation = {
                                        "type": "openreward_tool_correlation",
                                        "tool_use_id": tuid,
                                        "mcp_call_count": mcp_result_count if not is_error else None,
                                        "tool_name": mcp_tool_use_ids[tuid],
                                        "is_error": is_error,
                                    }
                                    log_file.write(json.dumps(annotation) + "\n")

            async def read_stderr():
                assert proc.stderr is not None
                async for line in proc.stderr:
                    line_str = line.decode(errors="replace").strip()
                    if line_str and (
                        "[openreward-bridge]" in line_str
                        or "Error" in line_str
                        or "error" in line_str
                    ):
                        print(f"  {line_str}", file=sys.stderr)

            try:
                await asyncio.gather(read_stdout(), read_stderr())
                await proc.wait()
            except Exception as e:
                if proc.returncode is None:
                    proc.kill()
                    await proc.wait()
                return AgentResult(success=False, error=str(e))
            finally:
                # Write summary and close logs
                result_data = None
                if result_file.exists():
                    try:
                        result_data = json.loads(result_file.read_text())
                    except json.JSONDecodeError:
                        pass

                # Extract cost/token data from Claude's result event
                cost_usd = None
                input_tokens = None
                output_tokens = None
                duration_ms = None
                if result_event:
                    cost_usd = result_event.get("total_cost_usd")
                    duration_ms = result_event.get("duration_ms")
                    usage = result_event.get("usage", {})
                    if isinstance(usage, dict):
                        input_tokens = usage.get("input_tokens")
                        output_tokens = usage.get("output_tokens")

                if main_log:
                    summary_event = {
                        "type": "openreward_summary",
                        "task_spec": ctx.task_spec,
                        "env": ctx.env_name,
                        "model": ctx.model,
                        "bridge_result": result_data,
                        "mcp_failed": mcp_failed,
                        "usage": {
                            "cost_usd": cost_usd,
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                            "duration_ms": duration_ms,
                        },
                    }
                    main_log.write(json.dumps(summary_event) + "\n")
                    main_log.close()
                for f in subagent_logs.values():
                    f.close()

                # Write per-trial result.json
                if log_dir:
                    trial_result = {
                        "task_id": trial_id,
                        "task_spec": ctx.task_spec,
                        "environment": ctx.env_name,
                        "agent": "claude-code",
                        "model": ctx.model,
                        "split": ctx.split,
                        "final_reward": result_data.get("last_reward") if result_data else None,
                        "finished": result_data.get("finished", False) if result_data else False,
                        "total_reward": result_data.get("total_reward") if result_data else None,
                        "tool_calls": result_data.get("calls") if result_data else turns_used,
                        "duration_seconds": (duration_ms / 1000) if duration_ms else None,
                        "usage": {
                            "cost_usd": cost_usd,
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                        },
                        "error": "MCP server 'openreward' failed to connect — no environment tools available" if mcp_failed else None,
                        "rollout_url": f"https://openreward.ai/rollout/{main_rollout.event_id}" if main_rollout else None,
                    }
                    result_json_path = log_dir / f"trial_{trial_id}_result.json"
                    result_json_path.write_text(json.dumps(trial_result, indent=2))

            # Build AgentResult
            if mcp_failed:
                return AgentResult(
                    success=False,
                    error="MCP server 'openreward' failed to connect — no environment tools available",
                    turns_used=turns_used,
                    cost_usd=cost_usd,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    duration_ms=duration_ms,
                )

            if result_data is not None:
                finished = result_data.get("finished", False)
                return AgentResult(
                    success=True,
                    reward=result_data.get("last_reward"),
                    finished=finished,
                    turns_used=result_data.get("calls", turns_used),
                    cost_usd=cost_usd,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    duration_ms=duration_ms,
                )

            # No result file
            if stdout_lines:
                print(f"[claude-code] stdout ({len(stdout_lines)} lines), last 3:", file=sys.stderr)
                for line in stdout_lines[-3:]:
                    print(f"  {line[:300]}", file=sys.stderr)

            return AgentResult(
                success=False,
                error=f"No result file produced. Exit code: {proc.returncode}. stdout_lines: {len(stdout_lines)}",
                turns_used=turns_used,
                cost_usd=cost_usd,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                duration_ms=duration_ms,
            )
