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

# Env tools with these names are always excluded from MCP —
# the agent's own built-in versions are preferred for planning/tracking.
ALWAYS_USE_BUILTIN = {"todo_write", "todowrite"}

# Filesystem tool names that are redundant when the env provides bash.
# By default, only bash is exposed via MCP for Gemini (bash can do everything).
GEMINI_FILESYSTEM_TOOLS = {"read", "write", "edit", "grep", "glob"}


def _compute_gemini_excluded_tools(
    env_tool_names: list[str],
    use_all_filesystem_tools: bool = False,
) -> list[str]:
    """Return env tool names that should be excluded from MCP for Gemini.

    When the environment provides ``bash``, the other filesystem tools
    (read, write, edit, grep, glob) are redundant — bash can do everything.
    By default they are excluded from MCP so the model only sees bash.
    Pass ``use_all_filesystem_tools=True`` to expose all of them.
    """
    exclude = [n for n in env_tool_names if n.lower() in ALWAYS_USE_BUILTIN]

    if use_all_filesystem_tools:
        return exclude

    names_lower = {n.lower() for n in env_tool_names}
    if "bash" in names_lower:
        for n in env_tool_names:
            if n.lower() in GEMINI_FILESYSTEM_TOOLS:
                exclude.append(n)

    return exclude


def _build_gemini_mcp_prompt(env_tool_names: list[str], excluded: list[str]) -> str:
    """Build the MCP tools section appended to the Gemini system prompt."""
    excluded_lower = {n.lower() for n in excluded}
    available = [n for n in env_tool_names if n.lower() not in excluded_lower]

    if not available:
        return ""

    lines = [
        "",
        "# OpenReward Environment Tools",
        "",
        "You are solving a task in an OpenReward environment. The environment provides "
        "tools via an MCP server named 'openreward'. Use these MCP tools for ALL "
        "environment interactions.",
        "",
        "IMPORTANT: Do NOT use your built-in tools (read_file, write_file, edit_file, "
        "list_directory, run_shell_command, etc.) — they operate on a scratch directory, "
        "NOT the actual environment. Only use the MCP tools listed below.",
        "",
    ]

    has_bash = any(n.lower() == "bash" for n in available)
    others = [n for n in available if n.lower() != "bash"]

    if has_bash:
        lines.append(
            "**Primary tool**: The environment provides a `bash` tool via MCP. "
            "Use this instead of your built-in shell for ALL commands. "
            "Your built-in shell is sandboxed (read-only) and must NOT be used "
            "for environment interactions."
        )
        lines.append("")

    if others:
        lines.append("**Additional MCP tools**:")
        for name in others:
            lines.append(f"- `{name}`")
        lines.append("")

    lines.extend([
        "## Termination",
        "",
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
You are an AI coding agent using Gemini CLI to solve tasks in an OpenReward environment. You have access to tools provided via an MCP server.

# How you work

## Task execution

You are a coding agent. Please keep going until the query is completely resolved, before ending your turn. Only terminate your turn when you are sure that the problem is solved. Autonomously resolve the query to the best of your ability, using the tools available to you. Do NOT guess or make up an answer.

## Important constraints

- Use ONLY the MCP tools provided by the 'openreward' server for ALL environment interactions.
- Do NOT use your built-in file system tools (read_file, write_file, edit_file, list_directory) — they operate on a scratch directory, not the actual environment.
- Do NOT use your built-in shell/terminal tool (run_shell_command) — use the MCP bash tool instead.
- Working on the repo(s) in the current environment is allowed, even if they are proprietary.
- Analyzing code for vulnerabilities is allowed.

## Validating your work

If the codebase has tests or the ability to build or run, consider using them to verify that your work is complete.\
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
        content_str = event.get("content", event.get("status", ""))
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
            if os.environ.get("OPENREWARD_URL"):
                mcp_env["OPENREWARD_URL"] = os.environ["OPENREWARD_URL"]
            if ctx.secrets:
                mcp_env["OPENREWARD_SESSION_SECRETS"] = json.dumps(ctx.secrets)

            # Rewards sidecar JSONL
            if log_dir:
                rewards_path = log_dir / f"trial_{trial_id}_rewards.jsonl"
                mcp_env["OPENREWARD_REWARDS_FILE"] = str(rewards_path)

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

            settings = {
                "mcpServers": {
                    "openreward": {
                        "command": sys.executable,
                        "args": ["-m", "firehorse.mcp"],
                        "env": mcp_env,
                    }
                },
            }
            if ctx.max_turns:
                settings["max_turns"] = min(ctx.max_turns, 100)

            (gemini_config_dir / "settings.json").write_text(json.dumps(settings))

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
