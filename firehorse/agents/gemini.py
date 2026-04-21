from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from openreward import (
    AssistantMessage,
    ToolCall,
    ToolResult,
    UserMessage,
)
from openreward.models import RolloutInfo

from firehorse.agents.base import BaseAgent, AgentResult, TrialContext
from firehorse.mcp.convert import strip_or_reward_marker

# Generic env tool names to exclude when gemini-cli toolset is active.
# The toolset provides both generic names (bash, read, write, ...) and
# gemini-native names (run_shell_command, read_file, write_file, ...).
# We exclude the generic names so only gemini-native names are exposed.
_GENERIC_TOOL_NAMES = {"bash", "edit", "grep", "read", "write", "glob", "todo_write"}


def _compute_gemini_excluded_tools(
    env_tool_names: list[str],
    use_all_filesystem_tools: bool = False,
) -> list[str]:
    """Return env tool names that should be excluded from MCP for Gemini.

    Excludes generic tool names (bash, read, write, etc.) so only the
    gemini-native names from the gemini-cli toolset are exposed
    (run_shell_command, read_file, write_file, etc.).
    """
    if use_all_filesystem_tools:
        return []

    return [n for n in env_tool_names if n.lower() in _GENERIC_TOOL_NAMES]


# Gemini CLI auto-prefixes MCP tools as "{server_name}_".
_GEMINI_MCP_SERVER_NAME = "openreward"


def _build_gemini_mcp_prompt(
    env_tool_names: list[str],
    excluded: list[str],
) -> str:
    """Build the MCP tools section appended to the Gemini system prompt.

    Tool names are prefixed with ``openreward_`` to match how the Gemini CLI
    exposes MCP tools (it auto-namespaces them as ``{server}_{tool}``).
    """
    excluded_lower = {n.lower() for n in excluded}
    available = [n for n in env_tool_names if n.lower() not in excluded_lower]

    if not available:
        return ""

    prefix = f"{_GEMINI_MCP_SERVER_NAME}_"
    has_shell = any(n.lower() == "run_shell_command" for n in available)
    others = [n for n in available if n.lower() != "run_shell_command"]

    lines = [
        "",
        "# Available MCP Tools",
        "",
        "The following tools are provided via the 'openreward' MCP server. "
        "Use ONLY these tools for all environment interactions.",
        "",
    ]

    if has_shell:
        lines.append(
            f"**Primary tool**: `{prefix}run_shell_command` — use this for ALL shell commands."
        )
        lines.append("")

    if others:
        lines.append("**Additional tools**:")
        for name in others:
            lines.append(f"- `{prefix}{name}`")
        lines.append("")

    lines.extend([
        "When a tool result contains [EPISODE COMPLETE], stop working immediately — "
        "the task is done. Do not make any more tool calls after seeing [EPISODE COMPLETE].",
    ])

    return "\n".join(lines)


def _resolve_model_gemini(model: str) -> str:
    """Map model string to Gemini CLI model name.

    Strips the ``google/`` prefix. Gemini CLI talks directly to Google's API
    so no provider routing is needed.
    """
    if model.startswith("google/"):
        return model.split("/", 1)[1]
    raise ValueError(
        f"Model {model!r} requires 'google/' prefix for the gemini agent. "
        f"Tip: use google/{model}"
    )


_UPSTREAM_SYSTEM_PROMPT = """\
You are a coding agent solving a task in an OpenReward environment. \
Built-in tools have been disabled. Use the MCP tools provided by the 'openreward' server for all interactions.

Keep going until the task is completely resolved.\
"""


def _log_gemini_event_to_rollout(
    event: dict,
    rollout: Any,
    accumulated_text: list[str],
) -> None:
    """Parse a Gemini stream-json event and log it to an OpenReward rollout.

    ``accumulated_text`` is a mutable list holding accumulated assistant message
    deltas. It is flushed (logged as AssistantMessage) before tool_use events
    and at the end of the stream.
    """
    event_type = event.get("type")

    if event_type == "message":
        role = event.get("role")
        content = event.get("content", "")

        if role == "assistant":
            # Accumulate deltas
            if content:
                accumulated_text.append(content)

    elif event_type == "tool_use":
        # Flush accumulated assistant text before tool call
        if accumulated_text:
            rollout.log(AssistantMessage(content="".join(accumulated_text)))
            accumulated_text.clear()

        rollout.log(ToolCall(
            name=event.get("tool_name", ""),
            content=json.dumps(event.get("parameters", {})),
            call_id=event.get("tool_id", ""),
        ))

    elif event_type == "tool_result":
        # Gemini CLI puts tool output in "output" field, not "content"
        content_str = event.get("output", event.get("content", event.get("status", "")))
        if isinstance(content_str, dict):
            content_str = json.dumps(content_str)
        content_str = str(content_str)

        # Parse [OR_REWARD:{...}] tag from MCP bridge output
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
                content=strip_or_reward_marker(content_str),
                call_id=event.get("tool_id", ""),
            ),
            reward=reward,
            is_finished=is_finished,
        )


_EFFORT_TO_THINKING_BUDGET = {"low": 1024, "medium": 5000, "high": 16000, "max": 24576}


def _build_gemini_settings(
    mcp_env: dict[str, str],
    max_turns: int | None = None,
    effort: str | None = None,
) -> dict[str, Any]:
    """Build the .gemini/settings.json content.

    Excludes built-in tools by name. MCP tools are prefixed with ``mcp_``
    (via OPENREWARD_TOOL_PREFIX) so they don't collide with the excluded names.
    """
    settings: dict[str, Any] = {
        "mcpServers": {
            "openreward": {
                "command": sys.executable,
                "args": ["-m", "firehorse.mcp"],
                "env": mcp_env,
            }
        },
        # No tools.exclude — MCP tools with mcp_ prefix don't collide.
    }
    if max_turns:
        settings["max_turns"] = min(max_turns, 100)
    if effort:
        settings["thinkingBudget"] = _EFFORT_TO_THINKING_BUDGET.get(effort, 16000)
    return settings


class GeminiAgent(BaseAgent):

    @property
    def name(self) -> str:
        return "gemini"

    async def setup(self) -> None:
        proc = await asyncio.create_subprocess_exec(
            "gemini", "--version",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(
                "Gemini CLI not found. Install via: npm install -g @google/gemini-cli"
            )
        version = stdout.decode().strip()
        print(f"Gemini CLI version: {version}", file=sys.stderr)

    async def run(self, ctx: TrialContext) -> AgentResult:
        start_time = time.monotonic()

        with tempfile.TemporaryDirectory(prefix="orwd-gemini-trial-") as tmpdir:
            tmppath = Path(tmpdir)
            result_file = tmppath / "result.json"
            trial_id = ctx.task_spec.get("id", ctx.task_spec.get("index", ctx.task_index))
            log_dir = Path(ctx.output_dir) if ctx.output_dir else None

            env_tool_names = [t.name for t in ctx.tools]

            # Build MCP environment variables (same as ClaudeCodeAgent/Codex)
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
                mcp_env["OPENREWARD_TOOL_DESCRIPTIONS"] = "env"
            # Pre-build tool specs so the MCP bridge can respond to list_tools
            # instantly (avoids Gemini CLI's MCP timeout during tool discovery).
            prebuilt = []
            excluded_lower = {n.lower() for n in _compute_gemini_excluded_tools(
                env_tool_names, ctx.use_all_filesystem_tools)}
            for t in ctx.tools:
                if t.name.lower() not in excluded_lower:
                    prebuilt.append({
                        "name": t.name,
                        "description": t.description or "",
                        "inputSchema": dict(t.input_schema) if t.input_schema else {"type": "object", "properties": {}},
                    })
            mcp_env["OPENREWARD_PREBUILT_TOOLS"] = json.dumps(prebuilt)

            if os.environ.get("OPENREWARD_URL"):
                mcp_env["OPENREWARD_URL"] = os.environ["OPENREWARD_URL"]
            if ctx.secrets:
                mcp_env["OPENREWARD_SESSION_SECRETS"] = json.dumps(ctx.secrets)

            # Rewards sidecar JSONL
            if log_dir:
                rewards_path = log_dir / f"trial_{trial_id}_rewards.jsonl"
                mcp_env["OPENREWARD_REWARDS_FILE"] = str(rewards_path.resolve())

            # Compute which env tools to exclude from MCP
            exclude_tools = _compute_gemini_excluded_tools(
                env_tool_names, ctx.use_all_filesystem_tools,
            )
            if exclude_tools:
                mcp_env["OPENREWARD_EXCLUDE_TOOLS"] = ",".join(exclude_tools)
                print(
                    f"[gemini] Excluding env tools from MCP: {', '.join(sorted(exclude_tools))}",
                    file=sys.stderr,
                )

            # Resolve model name
            model_name = _resolve_model_gemini(ctx.model)

            # Build .gemini/settings.json with MCP config
            gemini_config_dir = tmppath / ".gemini"
            gemini_config_dir.mkdir()

            settings = _build_gemini_settings(mcp_env, ctx.max_turns, ctx.effort)

            settings_json = json.dumps(settings)
            (gemini_config_dir / "settings.json").write_text(settings_json)
            print(f"[gemini] Settings: {len(settings_json)} bytes, env keys: {list(mcp_env.keys())}", file=sys.stderr)

            # Build system prompt + MCP tools section
            mcp_section = _build_gemini_mcp_prompt(env_tool_names, exclude_tools)
            system_prompt = _UPSTREAM_SYSTEM_PROMPT + mcp_section

            # Prepend system prompt to user prompt (Gemini CLI has no --system-prompt flag)
            full_prompt = f"{system_prompt}\n\n---\n\n{ctx.prompt_text}"

            # Build command
            cmd = [
                "gemini",
                "-p", full_prompt,
                "-o", "stream-json",
                "-m", model_name,
                "-y",  # auto-approve all tool calls (YOLO mode)
            ]

            # Subprocess environment — ensure Gemini API key is available
            proc_env = {**os.environ}
            # Gemini CLI uses GEMINI_API_KEY; also accept GOOGLE_API_KEY
            if not proc_env.get("GEMINI_API_KEY") and proc_env.get("GOOGLE_API_KEY"):
                proc_env["GEMINI_API_KEY"] = proc_env["GOOGLE_API_KEY"]

            print(f"[gemini] Launching with model={model_name}", file=sys.stderr)

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=proc_env,
                cwd=str(tmppath),
                limit=10 * 1024 * 1024,  # 10MB buffer for large JSONL lines
            )

            # --- JSONL logging ---
            main_log = None
            if log_dir:
                main_log = open(log_dir / f"trial_{trial_id}.jsonl", "w")

            # --- Rollout logging ---
            main_rollout = None
            if ctx.logging and ctx.rollout_client:
                try:
                    main_rollout = ctx.rollout_client.rollout.create(
                        run_name=ctx.run_name,
                        rollout_name=f"trial_{trial_id}",
                        environment=ctx.env_name,
                        variant=ctx.variant,
                        split=ctx.split,
                        task_spec=ctx.task_spec,
                        metadata={"effort": ctx.effort},
                    )
                    print(
                        f"[gemini] Rollout: https://openreward.ai/rollout/{main_rollout.event_id}",
                        file=sys.stderr,
                    )
                except Exception as e:
                    print(f"[gemini] Failed to create rollout: {e}", file=sys.stderr)

            # Log system prompt and environment prompt
            prompt_event = {
                "type": "openreward_prompt",
                "system_prompt": system_prompt,
                "environment_prompt": ctx.prompt_text,
            }
            if main_log:
                main_log.write(json.dumps(prompt_event) + "\n")

            if main_rollout:
                main_rollout.log(
                    UserMessage(content=full_prompt),
                    rollout_info=RolloutInfo(task_index=ctx.task_index, harness="gemini"),
                )

            # --- Read stdout and stderr concurrently ---
            turns_used = 0
            stdout_lines: list[str] = []
            result_stats: dict | None = None
            mcp_failed = False
            accumulated_text: list[str] = []

            async def read_stdout():
                nonlocal turns_used, result_stats, mcp_failed
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
                            main_log.write(strip_or_reward_marker(line_str) + "\n")
                        continue

                    # Write to JSONL
                    if main_log:
                        main_log.write(strip_or_reward_marker(line_str) + "\n")

                    # Log to OpenReward rollout
                    if main_rollout:
                        _log_gemini_event_to_rollout(event, main_rollout, accumulated_text)

                    # Track tool calls as turns
                    if event.get("type") == "tool_use":
                        turns_used += 1

                    # Capture result stats
                    if event.get("type") == "result":
                        result_stats = event.get("stats", {})

            async def read_stderr():
                nonlocal mcp_failed
                assert proc.stderr is not None
                async for line in proc.stderr:
                    line_str = line.decode(errors="replace").strip()
                    if not line_str:
                        continue
                    # Detect MCP failure
                    if "MCP issues detected" in line_str:
                        mcp_failed = True
                    if line_str and (
                        "[openreward-bridge]" in line_str
                        or "MCP" in line_str
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
                # Flush any remaining accumulated assistant text
                if accumulated_text and main_rollout:
                    main_rollout.log(AssistantMessage(content="".join(accumulated_text)))
                    accumulated_text.clear()

                duration_ms = int((time.monotonic() - start_time) * 1000)

                result_data = None
                if result_file.exists():
                    try:
                        result_data = json.loads(result_file.read_text())
                    except json.JSONDecodeError:
                        pass

                # Extract token usage from result stats
                input_tokens = None
                output_tokens = None
                stats_duration_ms = None
                if result_stats:
                    input_tokens = result_stats.get("input_tokens")
                    output_tokens = result_stats.get("output_tokens")
                    stats_duration_ms = result_stats.get("duration_ms")

                if main_log:
                    summary_event = {
                        "type": "openreward_summary",
                        "task_spec": ctx.task_spec,
                        "env": ctx.env_name,
                        "model": ctx.model,
                        "bridge_result": result_data,
                        "mcp_failed": mcp_failed,
                        "usage": {
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                            "duration_ms": stats_duration_ms or duration_ms,
                        },
                    }
                    main_log.write(json.dumps(summary_event) + "\n")
                    main_log.close()

                # Build MCP error string (keeps the canonical substring so trial.py retry fires)
                mcp_error: str | None = None
                if mcp_failed:
                    mcp_error = "MCP server 'openreward' failed to connect — no environment tools available"

                # Write per-trial result.json
                if log_dir:
                    trial_result = {
                        "task_id": trial_id,
                        "task_spec": ctx.task_spec,
                        "environment": ctx.env_name,
                        "agent": "gemini",
                        "model": ctx.model,
                        "split": ctx.split,
                        "final_reward": result_data.get("last_reward") if result_data else None,
                        "finished": result_data.get("finished", False) if result_data else False,
                        "total_reward": result_data.get("total_reward") if result_data else None,
                        "tool_calls": result_data.get("calls") if result_data else turns_used,
                        "duration_seconds": duration_ms / 1000,
                        "usage": {
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                        },
                        "error": mcp_error,
                        "rollout_url": (
                            f"https://openreward.ai/rollout/{main_rollout.event_id}"
                            if main_rollout else None
                        ),
                    }
                    result_json_path = log_dir / f"trial_{trial_id}_result.json"
                    result_json_path.write_text(json.dumps(trial_result, indent=2))

            # Build AgentResult
            if mcp_failed:
                return AgentResult(
                    success=False,
                    error=mcp_error,
                    turns_used=turns_used,
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
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    duration_ms=duration_ms,
                )

            # No result file
            if stdout_lines:
                print(f"[gemini] stdout ({len(stdout_lines)} lines), last 3:", file=sys.stderr)
                for line in stdout_lines[-3:]:
                    print(f"  {line[:300]}", file=sys.stderr)

            return AgentResult(
                success=False,
                error=f"No result file produced. Exit code: {proc.returncode}. stdout_lines: {len(stdout_lines)}",
                turns_used=turns_used,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                duration_ms=duration_ms,
            )
