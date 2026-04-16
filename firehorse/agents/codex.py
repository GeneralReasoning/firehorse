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
    ReasoningItem,
    SystemMessage,
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
# By default, only bash is exposed via MCP for Codex (bash can do everything).
CODEX_FILESYSTEM_TOOLS = {"read", "write", "edit", "grep", "glob"}

# Upstream Codex system prompt from codex-rs/core/prompt.md (v0.118).
# We use this as the base and append MCP-specific instructions.
_UPSTREAM_SYSTEM_PROMPT = """\
You are a coding agent running in the Codex CLI, a terminal-based coding assistant. Codex CLI is an open source project led by OpenAI. You are expected to be precise, safe, and helpful.

Your capabilities:

- Receive user prompts and other context provided by the harness, such as files in the workspace.
- Communicate with the user by streaming thinking & responses, and by making & updating plans.
- Emit function calls to run terminal commands and apply patches. Depending on how this specific run is configured, you can request that these function calls be escalated to the user for approval before running. More on this in the "Sandbox and approvals" section.

Within this context, Codex refers to the open-source agentic coding interface (not the old Codex language model built by OpenAI).

# How you work

## Personality

Your default personality and tone is concise, direct, and friendly. You communicate efficiently, always keeping the user clearly informed about ongoing actions without unnecessary detail. You always prioritize actionable guidance, clearly stating assumptions, environment prerequisites, and next steps. Unless explicitly asked, you avoid excessively verbose explanations about your work.

## Task execution

You are a coding agent. Please keep going until the query is completely resolved, before ending your turn and yielding back to the user. Only terminate your turn when you are sure that the problem is solved. Autonomously resolve the query to the best of your ability, using the tools available to you, before coming back to the user. Do NOT guess or make up an answer.

You MUST adhere to the following criteria when solving queries:

- Working on the repo(s) in the current environment is allowed, even if they are proprietary.
- Analyzing code for vulnerabilities is allowed.
- Showing user code and tool call details is allowed.
- Use the `apply_patch` tool to edit files (NEVER try `applypatch` or `apply-patch`, only `apply_patch`): {"command":["apply_patch","*** Begin Patch\\n*** Update File: path/to/file.py\\n@@ def example():\\n- pass\\n+ return 123\\n*** End Patch"]}

If completing the user's task requires writing or modifying files, your code and final answer should follow these coding guidelines, though user instructions (i.e. AGENTS.md) may override these guidelines:

- Fix the problem at the root cause rather than applying surface-level patches, when possible.
- Avoid unneeded complexity in your solution.
- Do not attempt to fix unrelated bugs or broken tests. It is not your responsibility to fix them. (You may mention them to the user in your final message though.)
- Keep changes consistent with the style of the existing codebase. Changes should be minimal and focused on the task.
- NEVER add copyright or license headers unless specifically requested.
- Do not `git commit` your changes or create new git branches unless explicitly requested.

## Validating your work

If the codebase has tests or the ability to build or run, consider using them to verify that your work is complete.

When testing, your philosophy should be to start as specific as possible to the code you changed so that you can catch issues efficiently, then make your way to broader tests as you build confidence.

## Shell commands

When using the shell, you must adhere to the following guidelines:

- When searching for text or files, prefer using `rg` or `rg --files` respectively because `rg` is much faster than alternatives like `grep`. (If the `rg` command is not found, then use alternatives.)
- Do not use python scripts to attempt to output larger chunks of a file.\
"""


def _compute_codex_excluded_tools(
    env_tool_names: list[str],
    use_all_filesystem_tools: bool = False,
) -> list[str]:
    """Return env tool names that should be excluded from MCP for Codex.

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
            if n.lower() in CODEX_FILESYSTEM_TOOLS:
                exclude.append(n)

    return exclude


def _build_codex_mcp_prompt(env_tool_names: list[str], excluded: list[str]) -> str:
    """Build the MCP tools section appended to the Codex system prompt."""
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


def _resolve_model_codex(
    model: str, provider_url: str | None,
) -> tuple[str, dict[str, str] | None]:
    """Map model string to Codex CLI model name and optional model_provider config.

    Returns ``(model_name, provider_config_or_none)``.

    When ``provider_config`` is ``None`` we let Codex use its default OpenAI
    provider. Otherwise the dict is merged into ``-c model_providers.fh=...``
    and ``-c model_provider="fh"`` so Codex talks to the given endpoint.

    We set ``support_namespaces=false`` on custom providers so Codex falls back
    to individual ``function`` tools for MCP servers — the ``namespace`` tool
    shape it uses by default isn't accepted by the OpenRouter Responses API
    (see https://github.com/openai/codex Responses API spec), and breaks the
    tool router when rewritten client-side.
    """
    if model.startswith("openai/"):
        return model.split("/", 1)[1], None
    elif model.startswith("openrouter/"):
        name = model.split("/", 1)[1]
        if not os.environ.get("OPENROUTER_API_KEY"):
            raise ValueError("OPENROUTER_API_KEY environment variable required for openrouter/ models")
        return name, {
            "name": "OpenRouter",
            "base_url": "https://openrouter.ai/api/v1",
            "env_key": "OPENROUTER_API_KEY",
            "wire_api": "responses",
            "support_namespaces": False,
        }
    elif provider_url:
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable required for --provider-url")
        return model, {
            "name": "Custom",
            "base_url": provider_url.rstrip("/"),
            "env_key": "OPENAI_API_KEY",
            "wire_api": "responses",
            "support_namespaces": False,
        }
    else:
        raise ValueError(
            f"Model {model!r} requires 'openai/' prefix for the codex agent. "
            f"Tip: use openai/{model}"
        )


def _extract_mcp_text(result_data: Any) -> str:
    """Extract text content from an MCP tool result (possibly structured)."""
    if isinstance(result_data, dict):
        content_parts = result_data.get("content", [])
        if isinstance(content_parts, list):
            texts = []
            for part in content_parts:
                if isinstance(part, dict) and part.get("type") == "text":
                    texts.append(part.get("text", ""))
            if texts:
                return "\n".join(texts)
        return json.dumps(result_data)
    return str(result_data)


def _log_codex_event_to_rollout(event: dict, rollout: Any) -> None:
    """Parse a Codex stream-json event and log it to an OpenReward rollout."""
    # v0.39: {"id":"0","msg":{"type":"...",...}}
    # v0.118: {"type":"...",...}  (flat structure)
    msg = event.get("msg")
    if isinstance(msg, dict):
        event_type = msg.get("type")
    elif "type" in event:
        msg = event
        event_type = event.get("type")
    else:
        return

    if event_type == "agent_message":
        text = msg.get("message", "")
        if text:
            rollout.log(AssistantMessage(content=text))

    elif event_type == "agent_reasoning":
        text = msg.get("text", "")
        summary = msg.get("summary", "")
        if text or summary:
            rollout.log(ReasoningItem(content=text, summary=summary))

    elif event_type == "mcp_tool_call_begin":
        invocation = msg.get("invocation", {})
        if not isinstance(invocation, dict):
            return
        rollout.log(ToolCall(
            name=invocation.get("tool", invocation.get("tool_name", invocation.get("name", ""))),
            content=json.dumps(invocation.get("arguments", invocation.get("input", {}))),
            call_id=msg.get("call_id", invocation.get("call_id", "")),
        ))

    elif event_type == "mcp_tool_call_end":
        result_data = msg.get("result", "")
        # Codex wraps MCP results in a Rust Result: {"Ok": {"content": [...], "isError": false}}
        if isinstance(result_data, dict) and "Ok" in result_data:
            result_data = result_data["Ok"]
        content_str = _extract_mcp_text(result_data)

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

        call_id = msg.get("call_id", "")
        if not call_id:
            invocation = msg.get("invocation", {})
            if isinstance(invocation, dict):
                call_id = invocation.get("call_id", "")

        rollout.log(
            ToolResult(content=strip_or_reward_marker(content_str), call_id=call_id),
            reward=reward,
            is_finished=is_finished,
        )

    elif event_type == "exec_command_begin":
        command = msg.get("command", [])
        cmd_str = " ".join(command) if isinstance(command, list) else str(command)
        rollout.log(ToolCall(
            name="shell",
            content=json.dumps({"command": cmd_str, "cwd": msg.get("cwd", "")}),
            call_id=msg.get("call_id", ""),
        ))

    elif event_type == "exec_command_end":
        output = msg.get("aggregated_output", msg.get("stdout", ""))
        content_str = str(output)
        rollout.log(ToolResult(
            content=content_str,
            call_id=msg.get("call_id", ""),
        ))


# Codex v0.121 introduced a `namespace` tool shape for MCP servers that the
# OpenRouter Responses API rejects ("No matching discriminator" at tools[N]).
# Until upstream ships a way to disable it, cap at the last known-good release.
_MAX_SUPPORTED_CODEX_VERSION = (0, 120)


def _parse_codex_version(version: str) -> tuple[int, int, int] | None:
    """Parse ``codex-cli 0.120.0`` (or ``codex 0.120.0``) into a tuple."""
    for token in version.split():
        parts = token.split(".")
        if len(parts) >= 3 and all(p.isdigit() for p in parts[:3]):
            return int(parts[0]), int(parts[1]), int(parts[2])
    return None


def _warn_if_unsupported_codex_version(version: str) -> None:
    parsed = _parse_codex_version(version)
    if parsed is None:
        return
    if parsed[:2] > _MAX_SUPPORTED_CODEX_VERSION:
        print(
            f"[codex] WARNING: codex-cli {parsed[0]}.{parsed[1]}.{parsed[2]} "
            f"is newer than the last tested version "
            f"({_MAX_SUPPORTED_CODEX_VERSION[0]}.{_MAX_SUPPORTED_CODEX_VERSION[1]}.x). "
            "Non-OpenAI providers (e.g. openrouter/*) may fail because Codex "
            "now sends `namespace` tools that OpenRouter's Responses API "
            "rejects. Pin via: bun add -g @openai/codex@0.120",
            file=sys.stderr,
        )


class CodexAgent(BaseAgent):

    @property
    def name(self) -> str:
        return "codex"

    async def setup(self) -> None:
        proc = await asyncio.create_subprocess_exec(
            "codex", "--version",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(
                "Codex CLI not found. Install via: npm install -g @openai/codex"
            )
        version = stdout.decode().strip()
        print(f"Codex CLI version: {version}", file=sys.stderr)
        _warn_if_unsupported_codex_version(version)

    async def run(self, ctx: TrialContext) -> AgentResult:
        start_time = time.monotonic()

        with tempfile.TemporaryDirectory(prefix="orwd-codex-trial-") as tmpdir:
            tmppath = Path(tmpdir)
            result_file = tmppath / "result.json"
            trial_id = ctx.task_spec.get("id", ctx.task_spec.get("index", ctx.task_index))
            log_dir = Path(ctx.output_dir) if ctx.output_dir else None

            env_tool_names = [t.name for t in ctx.tools]

            # Build MCP environment variables (same as ClaudeCodeAgent)
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
                mcp_env["OPENREWARD_TOOL_DESCRIPTIONS"] = "codex" if ctx.use_builtin_descriptions else "env"
            if os.environ.get("OPENREWARD_URL"):
                mcp_env["OPENREWARD_URL"] = os.environ["OPENREWARD_URL"]
            if ctx.secrets:
                mcp_env["OPENREWARD_SESSION_SECRETS"] = json.dumps(ctx.secrets)

            # Rewards sidecar JSONL
            if log_dir:
                rewards_path = log_dir / f"trial_{trial_id}_rewards.jsonl"
                mcp_env["OPENREWARD_REWARDS_FILE"] = str(rewards_path)

            # Compute which env tools to exclude from MCP.
            # By default, when bash is present, other filesystem tools are excluded.
            exclude_tools = _compute_codex_excluded_tools(
                env_tool_names, ctx.use_all_filesystem_tools,
            )
            if exclude_tools:
                mcp_env["OPENREWARD_EXCLUDE_TOOLS"] = ",".join(exclude_tools)
                print(
                    f"[codex] Excluding env tools from MCP: {', '.join(sorted(exclude_tools))}",
                    file=sys.stderr,
                )

            # Resolve model and optional model_provider config
            model_name, provider_config = _resolve_model_codex(ctx.model, ctx.provider_url)

            # Build system prompt: upstream Codex prompt + MCP tools section
            mcp_section = _build_codex_mcp_prompt(env_tool_names, exclude_tools)
            system_prompt = _UPSTREAM_SYSTEM_PROMPT + mcp_section

            # Codex exec has no --system-prompt flag; prepend to user prompt
            full_prompt = f"{system_prompt}\n\n---\n\n{ctx.prompt_text}"

            # Build command. We bypass Codex's own approval + sandbox because
            # every side-effect happens in the OpenReward environment via MCP.
            # Without this, Codex cancels every MCP tool call in exec mode
            # since there's no interactive UI to approve them.
            cmd = [
                "codex", "exec",
                "--json",
                "--dangerously-bypass-approvals-and-sandbox",
                "--skip-git-repo-check",
                "--model", model_name,
                "-C", str(tmppath),
            ]

            # MCP server config via dotted-path -c flags
            cmd.extend(["-c", f"mcp_servers.openreward.command={json.dumps(sys.executable)}"])
            cmd.extend(["-c", f"mcp_servers.openreward.args={json.dumps(['-m', 'firehorse.mcp'])}"])
            for key, value in mcp_env.items():
                cmd.extend(["-c", f"mcp_servers.openreward.env.{key}={json.dumps(value)}"])

            # Pass reasoning effort via config override
            if ctx.effort:
                cmd.extend(["-c", f"model_reasoning_effort={json.dumps(ctx.effort)}"])


            # Configure a non-default model_provider when needed (OpenRouter
            # or --provider-url). Codex will pick up the bearer token from
            # env_key itself, so no local auth proxy is required.
            if provider_config:
                for key, value in provider_config.items():
                    cmd.extend([
                        "-c",
                        f"model_providers.fh.{key}={json.dumps(value)}",
                    ])
                cmd.extend(["-c", 'model_provider="fh"'])

            cmd.append(full_prompt)

            proc_env = {**os.environ}

            print(f"[codex] Launching with model={model_name}", file=sys.stderr)

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=proc_env,
                limit=1024 * 1024,  # 1MB line limit for large stderr output
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
                        f"[codex] Rollout: https://openreward.ai/rollout/{main_rollout.event_id}",
                        file=sys.stderr,
                    )
                except Exception as e:
                    print(f"[codex] Failed to create rollout: {e}", file=sys.stderr)

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
                    SystemMessage(content=system_prompt),
                    rollout_info=RolloutInfo(task_index=ctx.task_index, harness="codex"),
                )
                main_rollout.log(UserMessage(content=ctx.prompt_text))

            # --- Read stdout and stderr concurrently ---
            turns_used = 0
            stdout_lines: list[str] = []
            token_info: dict | None = None

            async def read_stdout():
                nonlocal turns_used, token_info
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
                        _log_codex_event_to_rollout(event, main_rollout)

                    # Track token usage from token_count events
                    # Support both v0.39 (nested msg) and v0.118 (flat) formats
                    msg = event.get("msg", event)
                    if isinstance(msg, dict):
                        msg_type = msg.get("type")
                        if msg_type == "token_count":
                            token_info = msg.get("info", {})
                        # Count MCP tool calls and shell commands as turns
                        if msg_type in ("mcp_tool_call_begin", "exec_command_begin"):
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

                # Extract token usage from last token_count event
                input_tokens = None
                output_tokens = None
                if token_info:
                    total_usage = token_info.get("total_token_usage", {})
                    if isinstance(total_usage, dict):
                        input_tokens = total_usage.get("input_tokens")
                        output_tokens = total_usage.get("output_tokens")

                if main_log:
                    summary_event = {
                        "type": "openreward_summary",
                        "task_spec": ctx.task_spec,
                        "env": ctx.env_name,
                        "model": ctx.model,
                        "bridge_result": result_data,
                        "usage": {
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                            "duration_ms": duration_ms,
                        },
                    }
                    main_log.write(json.dumps(summary_event) + "\n")
                    main_log.close()

                # Write per-trial result.json
                if log_dir:
                    trial_result = {
                        "task_id": trial_id,
                        "task_spec": ctx.task_spec,
                        "environment": ctx.env_name,
                        "agent": "codex",
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
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    duration_ms=duration_ms,
                )

            # No result file
            if stdout_lines:
                print(f"[codex] stdout ({len(stdout_lines)} lines), last 3:", file=sys.stderr)
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
