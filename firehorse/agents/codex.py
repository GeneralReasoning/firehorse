from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

# asyncio.create_subprocess_exec on Windows uses CreateProcess directly and
# does not honor PATHEXT, so a bare "codex" fails when only codex.cmd is on
# PATH (the typical npm-global install). Resolve once via shutil.which.
_CODEX_BIN = shutil.which("codex") or "codex"

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


def _parse_reward_tag(content_str: str) -> tuple[float | None, bool]:
    """Extract reward and finished flag from an ``[OR_REWARD:{...}]`` tag."""
    m = re.search(r'\[OR_REWARD:(\{[^}]+\})\]', content_str)
    if not m:
        return None, False
    try:
        rd = json.loads(m.group(1))
        return rd.get("r"), rd.get("f", False)
    except (json.JSONDecodeError, AttributeError):
        return None, False


def _log_codex_event_to_rollout(event: dict, rollout: Any) -> None:
    """Parse a Codex stream-json event and log it to an OpenReward rollout.

    Handles three Codex CLI output schemas:

    1. **v0.39 (nested)**: ``{"id":"0","msg":{"type":"agent_message",...}}``
    2. **v0.118 (flat)**: ``{"type":"agent_message",...}``
    3. **item-based (current CLI)**: ``{"type":"item.completed","item":{"type":"agent_message",...}}``
       Also: ``turn.completed`` for token usage.
    """
    # --- Item-based format (current Codex CLI) ---
    event_type = event.get("type", "")

    if event_type in ("item.completed", "item.started"):
        item = event.get("item")
        if not isinstance(item, dict):
            return
        item_type = item.get("type", "")

        if item_type == "agent_message" and event_type == "item.completed":
            text = item.get("text", "")
            if text:
                rollout.log(AssistantMessage(content=text))

        elif item_type == "agent_reasoning" and event_type == "item.completed":
            text = item.get("text", "")
            summary = item.get("summary", "")
            if text or summary:
                rollout.log(ReasoningItem(content=text, summary=summary))

        elif item_type == "mcp_tool_call":
            if event_type == "item.started":
                # Tool call initiation
                rollout.log(ToolCall(
                    name=item.get("tool", ""),
                    content=json.dumps(item.get("arguments", {})),
                    call_id=item.get("id", ""),
                ))
            elif event_type == "item.completed":
                # Tool call result
                result_data = item.get("result", "")
                if isinstance(result_data, dict) and "Ok" in result_data:
                    result_data = result_data["Ok"]
                content_str = _extract_mcp_text(result_data)
                reward, is_finished = _parse_reward_tag(content_str)
                rollout.log(
                    ToolResult(
                        content=strip_or_reward_marker(content_str),
                        call_id=item.get("id", ""),
                    ),
                    reward=reward,
                    is_finished=is_finished,
                )
        return

    # --- Legacy v0.39 (nested msg) / v0.118 (flat) formats ---
    msg = event.get("msg")
    if isinstance(msg, dict):
        msg_type = msg.get("type")
    elif event_type not in ("turn.started", "turn.completed", "thread.started"):
        msg = event
        msg_type = event_type
    else:
        return

    if msg_type == "agent_message":
        text = msg.get("message", "")
        if text:
            rollout.log(AssistantMessage(content=text))

    elif msg_type == "agent_reasoning":
        text = msg.get("text", "")
        summary = msg.get("summary", "")
        if text or summary:
            rollout.log(ReasoningItem(content=text, summary=summary))

    elif msg_type == "mcp_tool_call_begin":
        invocation = msg.get("invocation", {})
        if not isinstance(invocation, dict):
            return
        rollout.log(ToolCall(
            name=invocation.get("tool", invocation.get("tool_name", invocation.get("name", ""))),
            content=json.dumps(invocation.get("arguments", invocation.get("input", {}))),
            call_id=msg.get("call_id", invocation.get("call_id", "")),
        ))

    elif msg_type == "mcp_tool_call_end":
        result_data = msg.get("result", "")
        if isinstance(result_data, dict) and "Ok" in result_data:
            result_data = result_data["Ok"]
        content_str = _extract_mcp_text(result_data)
        reward, is_finished = _parse_reward_tag(content_str)

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

    elif msg_type == "exec_command_begin":
        command = msg.get("command", [])
        cmd_str = " ".join(command) if isinstance(command, list) else str(command)
        rollout.log(ToolCall(
            name="shell",
            content=json.dumps({"command": cmd_str, "cwd": msg.get("cwd", "")}),
            call_id=msg.get("call_id", ""),
        ))

    elif msg_type == "exec_command_end":
        output = msg.get("aggregated_output", msg.get("stdout", ""))
        content_str = str(output)
        rollout.log(ToolResult(
            content=content_str,
            call_id=msg.get("call_id", ""),
        ))

class CodexAgent(BaseAgent):

    @property
    def name(self) -> str:
        return "codex"

    async def setup(self) -> None:
        proc = await asyncio.create_subprocess_exec(
            _CODEX_BIN, "--version",
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
            # Forward replay / resume signals to the MCP bridge subprocess.
            # `firehorse resume` sets these on the firehorse process so the
            # bridge (which runs in a fresh subprocess started by codex)
            # needs them explicitly injected via codex's
            # `-c mcp_servers.openreward.env.<KEY>=<VALUE>` config flags.
            # Without this forwarding the bridge never sees the manifest
            # path and silently skips replay — the agent then sees a
            # blank GW1 env instead of the rebuilt state.
            for _k in (
                "OPENREWARD_REPLAY_PATH",
                "OPENREWARD_REPLAY_ONLY",
                "OPENREWARD_REPLAY_PROGRESS_FILE",
            ):
                _v = os.environ.get(_k)
                if _v:
                    mcp_env[_k] = _v

            # Rewards sidecar JSONL — must be absolute since the MCP server
            # runs with a different CWD (codex sets `-C tmppath`).
            if log_dir:
                rewards_path = (log_dir / f"trial_{trial_id}_rewards.jsonl").resolve()
                rewards_path.parent.mkdir(parents=True, exist_ok=True)
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

            # Build command. We split the old
            # --dangerously-bypass-approvals-and-sandbox flag into its two
            # components so the shell IS sandboxed but the approval gate
            # is still bypassed:
            #
            #   --sandbox workspace-write   → codex's built-in shell can
            #                                 only read/write within the
            #                                 per-trial tmppath (-C).
            #                                 Without this, when MCP
            #                                 tools/list timed out the
            #                                 agent fell back to codex's
            #                                 host shell and read prior
            #                                 rollouts off ~/.codex/...
            #   approval_policy=never       → codex executes without
            #                                 prompting for approval.
            #                                 In `codex exec` there is no
            #                                 interactive UI; without
            #                                 this, every MCP tool call
            #                                 was auto-cancelled and the
            #                                 agent's tool calls returned
            #                                 None.
            # NOTE: tried splitting this into `--sandbox workspace-write`
            # + `approval_policy="never"` + `default_tools_approval_mode="never"`
            # + `mcp_tool_call_approval="never"` to sandbox codex's shell
            # without losing MCP. Every combination still produced
            # "user cancelled MCP tool call" in `codex exec` mode, so
            # we're back on the bypass flag. Host-shell leak (codex
            # reading ~/.codex/sessions on MCP failure) is a known risk
            # to fix separately — likely by running codex inside WSL
            # or a container, not via codex's own sandbox modes.
            cmd = [
                _CODEX_BIN, "exec",
                "--json",
                "--dangerously-bypass-approvals-and-sandbox",
                "--skip-git-repo-check",
                "--model", model_name,
                "-C", str(tmppath),
            ]
            # NOTE: we used to support `codex exec resume <thread_id>` here,
            # but codex resume does NOT re-bind MCP servers from -c
            # config flags. The agent's resumed conversation remembers
            # `mcp__openreward__*` tool names but the new codex process
            # has no MCP server registered to dispatch them, so every
            # tool call returns "unsupported call".
            #
            # `firehorse resume <dir>` now skips codex resume entirely.
            # It just relies on OPENREWARD_REPLAY_PATH (handled in the
            # MCP bridge) to fast-replay every prior tool call against
            # a fresh OR session so the env reaches the same state. The
            # agent starts fresh with full MCP tools — it has no memory
            # of the prior model reasoning, but `view_current_squad` /
            # the env's own state-inspection tools give it everything
            # it needs to continue.
            _resume_thread = None  # no longer used; kept for code below

            # MCP server config via dotted-path -c flags
            cmd.extend(["-c", f"mcp_servers.openreward.command={json.dumps(sys.executable)}"])
            cmd.extend(["-c", f"mcp_servers.openreward.args={json.dumps(['-m', 'firehorse.mcp'])}"])
            for key, value in mcp_env.items():
                cmd.extend(["-c", f"mcp_servers.openreward.env.{key}={json.dumps(value)}"])

            # Codex's defaults are 30 s startup and 60-120 s per tool call.
            # Long-horizon envs (e.g. FPL's `next_gameweek` runs a full
            # gameweek simulation; MarketMakerTimed's `run_strategy` plays
            # 15 min of L2 tick data) regularly exceed those, and codex
            # then returns `timed out awaiting tools/call`, dropping the
            # env's reward and finished signals on the floor. Bump both.
            # Honor operator overrides via env vars.
            # startup_timeout_sec is the deadline for the WHOLE
            # session-create chain (OR routing + env __init__ +
            # sandbox.start + tools/list). A bad-case cold pod with
            # OR-side ping latency can push to ~80-100s; bump default to
            # 300s so we don't tip past it on a slow day. tool_timeout_sec
            # covers individual tool calls (e.g. FPL's `next_gameweek`).
            startup_to = int(os.environ.get("OPENREWARD_MCP_STARTUP_SEC", "300"))
            tool_to = int(os.environ.get("OPENREWARD_MCP_TOOL_SEC", "600"))
            # Replay mode: the bridge's initialize() will fast-replay
            # every prior tool call before tools/list answers, which on
            # long rollouts can take 1-5 minutes on top of the normal
            # cold-start. Bump startup_timeout to 1200s ONLY when
            # OPENREWARD_REPLAY_PATH is set so non-resume runs keep the
            # tighter 300s ceiling (which surfaces real env-pod bugs
            # faster). Operator env-var override always wins.
            if os.environ.get("OPENREWARD_REPLAY_PATH") and startup_to < 1200:
                startup_to = 1200
                print(
                    f"[codex] replay mode detected — bumping "
                    f"mcp_servers.openreward.startup_timeout_sec={startup_to}",
                    file=sys.stderr,
                )
            cmd.extend(["-c", f"mcp_servers.openreward.startup_timeout_sec={startup_to}"])
            cmd.extend(["-c", f"mcp_servers.openreward.tool_timeout_sec={tool_to}"])

            # Pass reasoning effort via config override.
            # Codex 0.129+ renamed the top level from "max" to "xhigh"; firehorse
            # still accepts "max" as the canonical name, so translate here.
            if ctx.effort:
                effort = "xhigh" if ctx.effort == "max" else ctx.effort
                cmd.extend(["-c", f"model_reasoning_effort={json.dumps(effort)}"])


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

                # Non-OpenAI providers don't support Codex-specific tool types
                # (namespace, MCP elicitation, tool_suggest, etc.) that Codex
                # sends in the Responses API request.  Disable the features
                # that emit these unsupported tool shapes.
                cmd.extend([
                    "-c", "features.tool_call_mcp_elicitation=false",
                    "-c", "features.tool_suggest=false",
                ])

            # Pass the prompt via stdin (codex reads it when the positional arg
            # is "-"). Avoids Windows' ~32KB CreateProcess command-line limit,
            # which truncates large prompts mid-text and leaves the agent with
            # only part of the system prompt.
            # In resume mode we still need a `-` positional and SOMETHING on
            # stdin — `codex exec resume <id>` with no PROMPT just exits 1
            # with no output. We send a minimal "Continue." nudge; the agent
            # already has the full system + env prompt in its on-disk session.
            cmd.append("-")

            proc_env = {**os.environ}

            # Resume-on-capacity: on the first attempt we run the full
            # cmd with the prompt. On subsequent attempts we relaunch with
            # `codex exec resume --last` (no prompt) and rely on codex's
            # on-disk session state to pick up exactly where we left off.
            # NOTE: `codex exec resume` does NOT accept -C/--cd; CWD is
            # inherited from the prior session. Including -C makes codex
            # exit 2 with "unexpected argument '-C' found".
            resume_cmd = [
                _CODEX_BIN, "exec", "resume", "--last",
                "--json",
                "--dangerously-bypass-approvals-and-sandbox",
                "--skip-git-repo-check",
            ]
            # Re-attach the same MCP/effort/provider -c overrides so the
            # resumed run uses identical config. Skip the prompt-positional
            # ("-") since `resume` doesn't read stdin.
            resume_cmd.extend([a for a in cmd if a == "-c"]
                              + [c for i, c in enumerate(cmd)
                                 if i > 0 and cmd[i - 1] == "-c"])
            # Cap retries; back off exponentially starting at 30 s.
            max_capacity_retries = int(os.environ.get("OPENREWARD_CODEX_CAPACITY_RETRIES", "5"))
            backoff_base_s = float(os.environ.get("OPENREWARD_CODEX_CAPACITY_BACKOFF_BASE", "30"))

            print(f"[codex] Launching with model={model_name}", file=sys.stderr)

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=proc_env,
                limit=1024 * 1024,  # 1MB line limit for large stderr output
            )

            # Write prompt to stdin and close so codex sees EOF.
            # Resume mode sends a minimal "Continue." instead of the full
            # system+env prompt (which is already in the on-disk session).
            assert proc.stdin is not None
            stdin_prompt = "Continue." if _resume_thread else full_prompt
            proc.stdin.write(stdin_prompt.encode("utf-8"))
            await proc.stdin.drain()
            proc.stdin.close()

            # --- JSONL logging ---
            main_log = None
            if log_dir:
                main_log = open(log_dir / f"trial_{trial_id}.jsonl", "w")

            # --- Rollout logging ---
            main_rollout = None
            if ctx.logging and ctx.rollout_client:
                try:
                    model_short = ctx.model.split("/")[-1]
                    from firehorse.rollout_replay import resume_metadata
                    main_rollout = ctx.rollout_client.rollout.create(
                        run_name=ctx.run_name,
                        rollout_name=f"codex_{model_short}_{trial_id}",
                        environment=ctx.env_name,
                        variant=ctx.variant,
                        split=ctx.split,
                        task_spec=ctx.task_spec,
                        metadata={
                            "effort": ctx.effort,
                            "model": ctx.model,
                            "agent": "codex",
                            **resume_metadata(),
                        },
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

                # Resume mode: replay the dead session's messages into
                # the new rollout so the openreward.ai view of the
                # resumed run lines up with the original (no-op when
                # OPENREWARD_REPLAY_ROLLOUT_ID isn't set).
                try:
                    from firehorse.rollout_replay import maybe_replay_into
                    maybe_replay_into(main_rollout)
                except Exception as _e:
                    print(
                        f"[codex] rollout-message replay failed: "
                        f"{type(_e).__name__}: {_e}",
                        file=sys.stderr,
                    )

            # --- Read stdout and stderr concurrently ---
            turns_used = 0
            stdout_lines: list[str] = []
            token_info: dict | None = None
            run_started = time.monotonic()
            # Set to True when codex emits an `at_capacity` error.
            # Used to trigger a resume-with-backoff after proc.wait().
            capacity_hit = False
            capacity_attempt = 0

            def _heartbeat(tool_name: str) -> None:
                elapsed = time.monotonic() - run_started
                short = tool_name.replace("mcp__openreward__", "") or "?"
                print(
                    f"[codex][{trial_id}] turn {turns_used} ({elapsed:5.0f}s) "
                    f"→ {short}",
                    file=sys.stderr,
                    flush=True,
                )

            async def read_stdout():
                nonlocal turns_used, token_info, capacity_hit
                assert proc.stdout is not None
                async for line in proc.stdout:
                    line_str = line.decode(errors="replace").strip()
                    if not line_str:
                        continue
                    stdout_lines.append(line_str)
                    # Detect ONLY transient OpenAI errors that retry would
                    # actually fix. "Quota exceeded" / "insufficient_quota"
                    # mean the project-level credit pool is empty — no
                    # amount of backoff fixes that, so we let codex exit
                    # and surface the error. "Selected model is at
                    # capacity" and "rate_limit_exceeded" are transient
                    # (model overload / per-minute rate cap) and resolve
                    # in seconds-to-minutes — those we retry.
                    _ll = line_str.lower()
                    if ("selected model is at capacity" in _ll
                            or "model is at capacity" in _ll
                            or "rate_limit_exceeded" in _ll):
                        capacity_hit = True
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

                    # Track token usage and count turns.
                    # Current Codex CLI uses item-based events; legacy used flat/nested.
                    evt_type = event.get("type", "")

                    # Token usage: current CLI emits turn.completed with usage dict
                    if evt_type == "turn.completed":
                        usage = event.get("usage")
                        if isinstance(usage, dict):
                            token_info = usage

                    # Count MCP tool calls as turns (item.started with mcp_tool_call)
                    if evt_type == "item.started":
                        item = event.get("item", {})
                        if isinstance(item, dict) and item.get("type") == "mcp_tool_call":
                            turns_used += 1
                            _heartbeat(item.get("tool", ""))

                    # Legacy format support (v0.39 nested msg / v0.118 flat)
                    msg = event.get("msg", event)
                    if isinstance(msg, dict):
                        msg_type = msg.get("type")
                        if msg_type == "token_count":
                            token_info = msg.get("info", {})
                        if msg_type == "mcp_tool_call_begin":
                            turns_used += 1
                            invocation = msg.get("invocation", {})
                            tool_name = (
                                invocation.get("tool")
                                or invocation.get("tool_name")
                                or invocation.get("name")
                                or ""
                            ) if isinstance(invocation, dict) else ""
                            _heartbeat(tool_name)
                        elif msg_type == "exec_command_begin":
                            turns_used += 1
                            _heartbeat("shell")

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
                # Transient-failure retry. If codex died because of a
                # transient capacity / rate-limit error, just relaunch
                # the EXACT same command up to N times with backoff.
                # This is NOT the resume path — there's no codex
                # exec resume here, no param changes. The fresh codex
                # process starts from the original prompt; the OR
                # session is still alive within TTL so the env state
                # the agent has built up is preserved.
                while capacity_hit and capacity_attempt < max_capacity_retries:
                    capacity_attempt += 1
                    delay = backoff_base_s * (2 ** (capacity_attempt - 1))
                    print(
                        f"[codex][{trial_id}] transient error — sleeping {delay:.0f}s "
                        f"then relaunching (attempt {capacity_attempt}/{max_capacity_retries})",
                        file=sys.stderr,
                        flush=True,
                    )
                    await asyncio.sleep(delay)
                    capacity_hit = False
                    proc = await asyncio.create_subprocess_exec(
                        *cmd,
                        stdin=asyncio.subprocess.PIPE,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        env=proc_env,
                        limit=1024 * 1024,
                    )
                    # Same stdin handling as the initial launch.
                    assert proc.stdin is not None
                    retry_prompt = "Continue." if _resume_thread else full_prompt
                    proc.stdin.write(retry_prompt.encode("utf-8"))
                    await proc.stdin.drain()
                    proc.stdin.close()
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

                # Extract token usage from last token/turn event.
                # turn.completed has {input_tokens, output_tokens} directly;
                # legacy token_count has {total_token_usage: {input_tokens, ...}}.
                input_tokens = None
                output_tokens = None
                if token_info:
                    if "input_tokens" in token_info:
                        input_tokens = token_info.get("input_tokens")
                        output_tokens = token_info.get("output_tokens")
                    else:
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
