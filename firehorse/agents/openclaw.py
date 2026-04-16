"""OpenClaw agent — subprocess-based runner using the OpenClaw CLI.

Launches ``openclaw agent --local`` as a subprocess with an MCP bridge
to route tool calls through an OpenReward session sandbox.
"""
from __future__ import annotations

import asyncio
import glob
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from openreward import AssistantMessage, ToolCall, ToolResult, UserMessage
from openreward.api.rollouts.serializers.base import ReasoningItem

from firehorse.agents.base import BaseAgent, AgentResult, TrialContext
from firehorse.mcp.convert import strip_or_reward_marker
from firehorse.agents._subprocess_common import (
    ALWAYS_USE_BUILTIN,
    SUBPROCESS_LINE_LIMIT,
    build_mcp_prompt_section,
    build_submission_reminder,
    format_mcp_name_double,
    sanitize_prompt,
)


def _resolve_openclaw_model(model: str) -> tuple[str, str]:
    """Split 'provider/model-name' into (provider, model_name).

    OpenClaw uses provider/model format natively (e.g. "anthropic/claude-sonnet-4-6").
    """
    if "/" in model:
        provider, model_name = model.split("/", 1)
        return provider, model_name
    return "openai", model


# Map provider names to the env var that holds their API key.
_PROVIDER_KEY_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}


def _build_auth_profiles(provider: str) -> dict:
    """Build an OpenClaw auth-profiles.json with the API key for the given provider."""
    env_var = _PROVIDER_KEY_ENV.get(provider)
    api_key = os.environ.get(env_var, "") if env_var else ""
    if not api_key:
        # Try common fallbacks
        api_key = os.environ.get("ANTHROPIC_API_KEY", "") or os.environ.get("OPENAI_API_KEY", "")

    return {
        "version": 1,
        "profiles": {
            "default": {
                "type": "api_key",
                "provider": provider,
                "key": api_key,
            }
        },
    }


def _log_openclaw_from_session(
    session_jsonl_path: Path,
    toolcalls_path: Path,
    rollout: Any,
) -> None:
    """Log structured rollout from an OpenClaw session transcript JSONL.

    OpenClaw stores session transcripts as append-only JSONL with entries::

        {"type":"message","id":"...","parentId":"...","message":{...}}

    where ``message`` has ``role`` = "user" | "assistant" | "toolResult".
    Assistant messages have ``content`` as a list of blocks:
    ``{"type":"thinking","thinking":"..."}`` / ``{"type":"text","text":"..."}``
    / ``{"type":"toolCall","id":"...","name":"...","arguments":{...}}``.
    """
    if not session_jsonl_path.exists():
        print(f"[openclaw] Session file not found: {session_jsonl_path}", file=sys.stderr)
        _log_toolcalls_fallback(toolcalls_path, rollout)
        return

    # Build sidecar reward lookup
    sidecar_events: list[dict] = []
    if toolcalls_path.exists():
        for line in toolcalls_path.read_text().strip().splitlines():
            sidecar_events.append(json.loads(line))
    sidecar_idx = 0

    # Parse session JSONL entries (linear scan — good enough for single-turn agents)
    entries: list[dict] = []
    for line in session_jsonl_path.read_text().strip().splitlines():
        entry = json.loads(line)
        if entry.get("type") == "message":
            entries.append(entry)

    print(f"[openclaw] Session has {len(entries)} messages", file=sys.stderr)

    for entry in entries:
        msg = entry.get("message", {})
        role = msg.get("role", "")

        if role == "user":
            # Skip — prompt already logged
            continue

        elif role == "assistant":
            content_blocks = msg.get("content", [])
            if isinstance(content_blocks, str):
                # Simple text content
                rollout.log(AssistantMessage(content=content_blocks))
                continue

            for block in content_blocks:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type", "")

                if btype == "thinking":
                    thinking = block.get("thinking", "")
                    if thinking:
                        rollout.log(ReasoningItem(content=thinking))

                elif btype == "text":
                    text = block.get("text", "")
                    if text:
                        rollout.log(AssistantMessage(content=text))

                elif btype == "toolCall":
                    call_id = block.get("id", "")
                    name = block.get("name", "")
                    args = block.get("arguments", {})
                    rollout.log(ToolCall(
                        name=name,
                        content=json.dumps(args) if isinstance(args, dict) else str(args),
                        call_id=call_id,
                    ))

        elif role == "toolResult":
            call_id = msg.get("toolCallId", "")
            content_blocks = msg.get("content", [])
            if isinstance(content_blocks, list):
                text_parts = []
                for block in content_blocks:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                content_str = "\n".join(text_parts)
            else:
                content_str = str(content_blocks)

            # Match with sidecar for reward/finished
            reward = None
            is_finished = False
            if sidecar_idx < len(sidecar_events):
                se = sidecar_events[sidecar_idx]
                reward = se.get("reward")
                is_finished = se.get("finished", False)
                sidecar_idx += 1
            else:
                m = re.search(r'\[OR_REWARD:(\{[^}]+\})\]', content_str)
                if m:
                    try:
                        rd = json.loads(m.group(1))
                        reward = rd.get("r")
                        is_finished = rd.get("f", False)
                    except (json.JSONDecodeError, KeyError):
                        pass

            rollout.log(
                ToolResult(content=strip_or_reward_marker(content_str), call_id=call_id),
                reward=reward,
                is_finished=is_finished,
            )


def _log_toolcalls_fallback(toolcalls_path: Path, rollout: Any) -> None:
    """Fallback: log only sidecar tool calls when session JSONL is unavailable."""
    if not toolcalls_path.exists():
        return
    for line in toolcalls_path.read_text().strip().splitlines():
        event = json.loads(line)
        call_id = event.get("call_id", "")
        rollout.log(ToolCall(
            name=event.get("tool", ""),
            content=json.dumps(event.get("arguments", {})),
            call_id=call_id,
        ))
        rollout.log(
            ToolResult(content=event.get("result", ""), call_id=call_id),
            reward=event.get("reward"),
            is_finished=event.get("finished", False),
        )


class OpenClawAgent(BaseAgent):

    @property
    def name(self) -> str:
        return "openclaw"

    async def setup(self) -> None:
        proc = await asyncio.create_subprocess_exec(
            "openclaw", "--version",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(
                "OpenClaw CLI not found. Install via: npm install -g openclaw@latest"
            )
        version = stdout.decode().strip()
        print(f"OpenClaw version: {version}", file=sys.stderr)

    async def run(self, ctx: TrialContext) -> AgentResult:
        start_time = time.monotonic()

        with tempfile.TemporaryDirectory(prefix="orwd-openclaw-trial-") as tmpdir:
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
            # OpenClaw uses its own native tools; env tools come through MCP
            # with their original descriptions from the environment.
            mcp_env["OPENREWARD_TOOL_DESCRIPTIONS"] = "env"
            if os.environ.get("OPENREWARD_URL"):
                mcp_env["OPENREWARD_URL"] = os.environ["OPENREWARD_URL"]
            if ctx.secrets:
                mcp_env["OPENREWARD_SESSION_SECRETS"] = json.dumps(ctx.secrets)

            # Rewards sidecar JSONL
            if log_dir:
                rewards_path = log_dir / f"trial_{trial_id}_rewards.jsonl"
                mcp_env["OPENREWARD_REWARDS_FILE"] = str(rewards_path)

            # Tool-call log for rollout reconstruction
            toolcalls_path = tmppath / "toolcalls.jsonl"
            mcp_env["OPENREWARD_TOOLCALLS_FILE"] = str(toolcalls_path)

            # Exclude env tools that duplicate the agent's built-in planning tools
            exclude_tools = [n for n in env_tool_names if n.lower() in ALWAYS_USE_BUILTIN]
            if exclude_tools:
                mcp_env["OPENREWARD_EXCLUDE_TOOLS"] = ",".join(exclude_tools)

            # Resolve model: OpenClaw uses "provider/model" format natively
            oc_model = ctx.model  # e.g. "anthropic/claude-sonnet-4-6"
            oc_provider, oc_model_name = _resolve_openclaw_model(ctx.model)

            # Write OpenClaw config with MCP server to temp directory.
            # OpenClaw reads config from $OPENCLAW_HOME/.openclaw/openclaw.json.
            # We set OPENCLAW_HOME=tmppath so config lives at tmppath/.openclaw/openclaw.json.
            openclaw_dir = tmppath / ".openclaw"
            openclaw_dir.mkdir()

            # Write auth profile so OpenClaw can authenticate with the provider.
            # Auth profiles live at .openclaw/agents/main/agent/auth-profiles.json.
            agent_dir = openclaw_dir / "agents" / "main" / "agent"
            agent_dir.mkdir(parents=True)
            auth_profiles = _build_auth_profiles(oc_provider)
            (agent_dir / "auth-profiles.json").write_text(json.dumps(auth_profiles, indent=2))

            openclaw_config: dict[str, Any] = {
                "agents": {
                    "defaults": {
                        "model": oc_model,
                    }
                },
                "auth": {
                    "profiles": {
                        "default": {
                            "provider": oc_provider,
                            "mode": "api_key",
                        }
                    },
                    "order": {
                        oc_provider: ["default"],
                    },
                },
                "mcp": {
                    "servers": {
                        "openreward": {
                            "command": sys.executable,
                            "args": ["-m", "firehorse.mcp"],
                            "env": mcp_env,
                            "transport": "stdio",
                        }
                    }
                },
            }
            config_path = openclaw_dir / "openclaw.json"
            config_path.write_text(json.dumps(openclaw_config, indent=2))

            # Build prompt: user prompt + MCP tool section (which includes
            # termination instructions) + optional submission reminder.
            mcp_section = build_mcp_prompt_section(env_tool_names, format_mcp_name_double)
            submission_reminder = build_submission_reminder(env_tool_names, format_mcp_name_double)
            full_prompt = ctx.prompt_text
            if mcp_section:
                full_prompt = f"{full_prompt}\n{mcp_section}"
            if submission_reminder:
                full_prompt = f"{full_prompt}\n\n{submission_reminder}"
            full_prompt = sanitize_prompt(full_prompt)

            # Build command
            cmd = [
                "openclaw", "agent",
                "--local",
                "--json",
                "--session-id", f"orwd-{trial_id}",
                "-m", full_prompt,
            ]

            proc_env = {
                **os.environ,
                "OPENCLAW_HOME": str(tmppath),
            }

            print(f"[openclaw] Launching with config at {config_path}", file=sys.stderr)

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=proc_env,
                limit=SUBPROCESS_LINE_LIMIT,
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
                        metadata={"model": ctx.model, "agent": "openclaw"},
                    )
                    print(
                        f"[openclaw] Rollout: https://openreward.ai/rollout/{main_rollout.event_id}",
                        file=sys.stderr,
                    )
                except Exception as e:
                    print(f"[openclaw] Failed to create rollout: {e}", file=sys.stderr)

            # Log prompt
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
            # OpenClaw outputs its JSON result to stderr (via runtime.log),
            # mixed with other log lines. We collect both streams.
            turns_used = 0
            stdout_chunks: list[bytes] = []
            stderr_lines: list[str] = []

            async def read_stdout():
                assert proc.stdout is not None
                while True:
                    chunk = await proc.stdout.read(65536)
                    if not chunk:
                        break
                    stdout_chunks.append(chunk)

            # Strip ANSI escape codes for cleaner log parsing.
            _ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

            async def read_stderr():
                nonlocal turns_used
                assert proc.stderr is not None
                async for line in proc.stderr:
                    line_str = line.decode(errors="replace").strip()
                    if not line_str:
                        continue
                    clean = _ANSI_RE.sub("", line_str)
                    stderr_lines.append(clean)
                    # Show MCP bridge messages, errors, and tool-call progress
                    if "[openreward-bridge]" in clean:
                        print(f"  {clean}", file=sys.stderr)
                    elif "tool_call" in clean.lower() or "mcp" in clean.lower():
                        turns_used += 1
                        print(f"  [openclaw] {clean[:200]}", file=sys.stderr)
                    elif "Error" in clean or "error" in clean or "FailoverError" in clean:
                        print(f"  [openclaw] {clean[:300]}", file=sys.stderr)
                    elif "[diagnostic]" in clean and "lane" in clean:
                        pass  # too noisy
                    elif "[model-fallback" in clean or "[agent/embedded]" in clean:
                        pass  # noisy

            try:
                await asyncio.gather(read_stdout(), read_stderr())
                await proc.wait()
            except Exception as e:
                if proc.returncode is None:
                    proc.kill()
                    await proc.wait()
                return AgentResult(success=False, error=str(e))

            duration_ms = int((time.monotonic() - start_time) * 1000)

            # Parse OpenClaw's JSON output.
            # OpenClaw outputs pretty-printed JSON to stderr (via console.log).
            stdout_raw = b"".join(stdout_chunks).decode(errors="replace").strip()
            openclaw_output = None
            if stdout_raw:
                try:
                    openclaw_output = json.loads(stdout_raw)
                except json.JSONDecodeError:
                    pass

            if openclaw_output is None and stderr_lines:
                # Find the '{' line before '"payloads"', collect until braces balance.
                json_start = None
                for i, line in enumerate(stderr_lines):
                    if '"payloads"' in line:
                        for j in range(i - 1, max(i - 3, -1), -1):
                            if stderr_lines[j].strip() == '{':
                                json_start = j
                                break
                        if json_start is None and line.strip().startswith('{'):
                            json_start = i
                        break
                if json_start is not None:
                    depth = 0
                    json_lines = []
                    for k in range(json_start, len(stderr_lines)):
                        json_lines.append(stderr_lines[k])
                        depth += stderr_lines[k].count('{') - stderr_lines[k].count('}')
                        if depth <= 0:
                            break
                    try:
                        openclaw_output = json.loads("\n".join(json_lines))
                    except json.JSONDecodeError:
                        pass

            if openclaw_output:
                print(f"[openclaw] Captured agent output ({len(openclaw_output.get('payloads', []))} payloads)", file=sys.stderr)
            else:
                print(f"[openclaw] No agent output captured (stdout={len(stdout_raw)}b, stderr={len(stderr_lines)} lines)", file=sys.stderr)

            result_data = None
            if result_file.exists():
                try:
                    result_data = json.loads(result_file.read_text())
                except json.JSONDecodeError:
                    pass

            # Log structured session data to rollout.
            # OpenClaw stores session transcripts as JSONL at
            # $OPENCLAW_HOME/.openclaw/agents/main/sessions/<session-id>.jsonl
            if main_rollout:
                session_jsonl = openclaw_dir / "agents" / "main" / "sessions" / f"orwd-{trial_id}.jsonl"
                # Glob in case the filename varies
                if not session_jsonl.exists():
                    candidates = glob.glob(str(openclaw_dir / "agents" / "*" / "sessions" / f"*orwd*{trial_id}*.jsonl"))
                    if candidates:
                        session_jsonl = Path(candidates[0])
                _log_openclaw_from_session(session_jsonl, toolcalls_path, main_rollout)

            # Log to JSONL
            if main_log:
                if openclaw_output:
                    main_log.write(json.dumps({"type": "openclaw_output", **openclaw_output}) + "\n")
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
                    "agent": "openclaw",
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

            if stdout_raw:
                print(f"[openclaw] stdout: {stdout_raw[:500]}", file=sys.stderr)

            return AgentResult(
                success=False,
                error=f"No result file produced. Exit code: {proc.returncode}.",
                turns_used=turns_used,
                duration_ms=duration_ms,
            )
