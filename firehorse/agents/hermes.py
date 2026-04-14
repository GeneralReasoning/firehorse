"""Hermes Agent — subprocess-based runner using the Hermes CLI.

Launches ``hermes chat -q`` as a subprocess with an MCP bridge
to route tool calls through an OpenReward session sandbox.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from firehorse.agents.base import BaseAgent, AgentResult, TrialContext

# Env tools that should not be exposed via MCP — the agent's own
# planning/tracking tools are preferred.
ALWAYS_USE_BUILTIN = {"todo_write", "todowrite"}

# Submission tool names to remind the agent about.
SUBMISSION_TOOL_NAMES = {"submit", "answer", "submit_answer"}

# Buffer limit for subprocess stdout/stderr line reading.
_SUBPROCESS_LINE_LIMIT = 10 * 1024 * 1024  # 10 MB


def _build_hermes_prompt(
    env_tool_names: list[str],
    mcp_server_name: str = "openreward",
) -> str:
    """Build MCP tool instructions appended to the user prompt."""
    if not env_tool_names:
        return ""

    lines = [
        "",
        "# OpenReward Environment Tools",
        "",
        "You are solving a task in an OpenReward environment. The environment provides "
        "tools via an MCP server named 'openreward'. Use these MCP tools for ALL "
        "environment interactions instead of your built-in terminal, read_file, "
        "write_file, and patch tools.",
        "",
        "Available MCP tools:",
    ]

    for name in env_tool_names:
        if name.lower() in ALWAYS_USE_BUILTIN:
            continue
        lines.append(f"- `mcp_{mcp_server_name}_{name}`")

    lines.append("")
    lines.append("## Termination")
    lines.append("")
    lines.append(
        "When a tool result contains [EPISODE COMPLETE], stop working immediately — "
        "the task is done. Do not make any more tool calls after seeing [EPISODE COMPLETE]."
    )

    return "\n".join(lines)


def _build_submission_reminder(
    env_tool_names: list[str], mcp_server_name: str = "openreward",
) -> str:
    """Build a prompt reminder telling the agent to call submission tools."""
    submission_tools = []
    for name in env_tool_names:
        if name.lower() in SUBMISSION_TOOL_NAMES:
            submission_tools.append(f"mcp_{mcp_server_name}_{name}")

    if not submission_tools:
        return ""

    tool_list = ", ".join(f"`{t}`" for t in submission_tools)
    return (
        f"Submission Reminder:\n"
        f"This environment has a submission tool: {tool_list}. "
        f"You MUST call this tool to submit your final answer before you finish. "
        f"If you stop without calling it, your work will not be scored."
    )


def _sanitize_prompt(text: str) -> str:
    """Ensure prompt text doesn't start with '-'."""
    if text.startswith("-"):
        return " " + text
    return text


def _resolve_model_hermes(
    model: str, provider_url: str | None,
) -> tuple[str, str | None]:
    """Map model string to Hermes --model flag and optional --provider flag.

    Returns (model_name, provider_or_none).
    """
    if "/" in model:
        provider, model_name = model.split("/", 1)
        return model_name, provider
    return model, None


class HermesAgent(BaseAgent):

    @property
    def name(self) -> str:
        return "hermes"

    async def setup(self) -> None:
        proc = await asyncio.create_subprocess_exec(
            "hermes", "--version",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(
                "Hermes Agent CLI not found. Install via: "
                "curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash"
            )
        version = stdout.decode().strip()
        print(f"Hermes Agent version: {version}", file=sys.stderr)

    async def run(self, ctx: TrialContext) -> AgentResult:
        start_time = time.monotonic()

        with tempfile.TemporaryDirectory(prefix="orwd-hermes-trial-") as tmpdir:
            tmppath = Path(tmpdir)
            result_file = tmppath / "result.json"
            trial_id = ctx.task_spec.get(
                "id", ctx.task_spec.get(
                    "index", ctx.task_index if ctx.task_index is not None else "unknown",
                ),
            )
            log_dir = Path(ctx.output_dir) if ctx.output_dir else None

            env_tool_names = [t.name for t in ctx.tools]

            # Build MCP environment variables
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

            # Exclude env tools that duplicate the agent's built-in planning tools
            exclude_tools = [n for n in env_tool_names if n.lower() in ALWAYS_USE_BUILTIN]
            if exclude_tools:
                mcp_env["OPENREWARD_EXCLUDE_TOOLS"] = ",".join(exclude_tools)

            # Write Hermes config with MCP server to temp directory.
            # Hermes reads config from HERMES_HOME or ~/.hermes.
            hermes_home = tmppath / ".hermes"
            hermes_home.mkdir()
            (hermes_home / "sessions").mkdir()

            # Hermes config.yaml with MCP server and disabled built-in toolsets
            hermes_config = {
                "mcp_servers": {
                    "openreward": {
                        "command": sys.executable,
                        "args": ["-m", "firehorse.mcp"],
                        "env": mcp_env,
                        "enabled": True,
                    }
                },
                # Disable built-in terminal/file tools — use MCP equivalents
                "disabled_toolsets": ["terminal", "file_operations"],
            }
            config_path = hermes_home / "config.yaml"
            # YAML is a superset of JSON, so JSON content works in a .yaml file
            config_path.write_text(json.dumps(hermes_config, indent=2))

            # Resolve model
            model_name, provider = _resolve_model_hermes(ctx.model, ctx.provider_url)

            # Build prompt
            mcp_section = _build_hermes_prompt(env_tool_names)
            termination = (
                "When a tool result contains [EPISODE COMPLETE], stop working immediately — "
                "the task is done. Do not make any more tool calls after seeing [EPISODE COMPLETE]."
            )
            full_prompt = f"{ctx.prompt_text}\n\nTermination Instructions:\n{termination}"
            if mcp_section:
                full_prompt = f"{full_prompt}\n{mcp_section}"
            submission_reminder = _build_submission_reminder(env_tool_names)
            if submission_reminder:
                full_prompt = f"{full_prompt}\n\n{submission_reminder}"
            full_prompt = _sanitize_prompt(full_prompt)

            # Build command
            cmd = [
                "hermes", "chat",
                "-q", full_prompt,
                "-Q",  # Quiet/programmatic mode
                "--model", model_name,
                "--yolo",  # Skip approval prompts
            ]

            if provider:
                cmd.extend(["--provider", provider])

            if ctx.max_turns:
                cmd.extend(["--max-turns", str(ctx.max_turns)])

            proc_env = {
                **os.environ,
                "HERMES_HOME": str(hermes_home),
            }

            print(f"[hermes] Launching with model={model_name}, config at {config_path}", file=sys.stderr)

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=proc_env,
                limit=_SUBPROCESS_LINE_LIMIT,
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
                        split=ctx.split,
                        task_spec=ctx.task_spec,
                        metadata={"model": ctx.model, "agent": "hermes"},
                    )
                    print(
                        f"[hermes] Rollout: https://openreward.ai/rollout/{main_rollout.event_id}",
                        file=sys.stderr,
                    )
                except Exception as e:
                    print(f"[hermes] Failed to create rollout: {e}", file=sys.stderr)

            # Log prompt
            prompt_event = {
                "type": "openreward_prompt",
                "system_prompt": "default",
                "environment_prompt": full_prompt,
            }
            if main_log:
                main_log.write(json.dumps(prompt_event) + "\n")

            if main_rollout:
                from openreward import UserMessage
                try:
                    main_rollout.log(UserMessage(content=full_prompt))
                except Exception:
                    pass

            # --- Read stdout and stderr concurrently ---
            turns_used = 0
            stdout_lines: list[str] = []

            async def read_stdout():
                nonlocal turns_used
                assert proc.stdout is not None
                async for line in proc.stdout:
                    line_str = line.decode(errors="replace").strip()
                    if not line_str:
                        continue
                    stdout_lines.append(line_str)
                    if main_log:
                        main_log.write(line_str + "\n")
                    try:
                        event = json.loads(line_str)
                    except json.JSONDecodeError:
                        continue

                    # Count tool uses
                    event_type = event.get("type", "")
                    if event_type in ("tool_use", "tool_call", "mcp_tool_call_begin"):
                        turns_used += 1

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
                duration_ms = int((time.monotonic() - start_time) * 1000)

                result_data = None
                if result_file.exists():
                    try:
                        result_data = json.loads(result_file.read_text())
                    except json.JSONDecodeError:
                        pass

                if main_log:
                    summary_event = {
                        "type": "openreward_summary",
                        "task_spec": ctx.task_spec,
                        "env": ctx.env_name,
                        "model": ctx.model,
                        "bridge_result": result_data,
                        "usage": {"duration_ms": duration_ms},
                    }
                    main_log.write(json.dumps(summary_event) + "\n")
                    main_log.close()

                # Write per-trial result.json
                if log_dir:
                    trial_result = {
                        "task_id": trial_id,
                        "task_spec": ctx.task_spec,
                        "environment": ctx.env_name,
                        "agent": "hermes",
                        "model": ctx.model,
                        "split": ctx.split,
                        "final_reward": result_data.get("last_reward") if result_data else None,
                        "finished": result_data.get("finished", False) if result_data else False,
                        "total_reward": result_data.get("total_reward") if result_data else None,
                        "tool_calls": result_data.get("calls") if result_data else turns_used,
                        "duration_seconds": duration_ms / 1000,
                        "error": None,
                        "rollout_url": (
                            f"https://openreward.ai/rollout/{main_rollout.event_id}"
                            if main_rollout else None
                        ),
                    }
                    result_json_path = log_dir / f"trial_{trial_id}_result.json"
                    result_json_path.write_text(json.dumps(trial_result, indent=2))

            # Build AgentResult
            if result_data is not None:
                finished = result_data.get("finished", False)
                return AgentResult(
                    success=True,
                    reward=result_data.get("last_reward"),
                    finished=finished,
                    turns_used=result_data.get("calls", turns_used),
                    duration_ms=duration_ms,
                )

            if stdout_lines:
                print(f"[hermes] stdout ({len(stdout_lines)} lines), last 3:", file=sys.stderr)
                for line in stdout_lines[-3:]:
                    print(f"  {line[:300]}", file=sys.stderr)

            return AgentResult(
                success=False,
                error=f"No result file produced. Exit code: {proc.returncode}. stdout_lines: {len(stdout_lines)}",
                turns_used=turns_used,
                duration_ms=duration_ms,
            )
