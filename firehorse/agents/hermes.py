"""Hermes Agent — subprocess-based runner using the Hermes CLI.

Launches ``hermes chat -q <prompt> -Q`` as a subprocess with an MCP bridge
exposing the OpenReward session as tools. Unlike claude-code/gemini, the
Hermes CLI in quiet mode does not stream per-turn events to stdout, so we
reconstruct the rollout from the MCP bridge's ``OPENREWARD_TOOLCALLS_FILE``
sidecar after the run finishes.
"""
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

import yaml

from openreward import (
    AssistantMessage,
    ReasoningItem,
    ToolCall,
    ToolResult,
    UserMessage,
)
from openreward.models import RolloutInfo

from firehorse.agents.base import BaseAgent, AgentResult, TrialContext
from firehorse.mcp.convert import parse_or_reward_marker, strip_or_reward_marker

# asyncio.create_subprocess_exec on Windows uses CreateProcess directly and
# does not honor PATHEXT, so a bare "hermes" fails when only hermes.cmd is on
# PATH. Resolve once via shutil.which.
_HERMES_BIN = shutil.which("hermes") or "hermes"

# Env tools the agent's built-in planning tools replace.
ALWAYS_USE_BUILTIN = {"todo_write", "todowrite"}

# Hermes prefixes MCP tools as "mcp_<server>_<tool>" (non-alphanum → "_").
# See hermes tools/mcp_tool.py: sanitize_mcp_name_component + prefixed_name.
_MCP_SERVER_NAME = "openreward"
_MCP_TOOL_PREFIX = f"mcp_{_MCP_SERVER_NAME}_"

# Built-in Hermes toolsets to disable — filesystem/shell access goes through
# the OpenReward sandbox tools instead. Other toolsets (web, search, todo,
# memory, skills, delegation, ...) remain available.
_DISABLED_TOOLSETS = ["terminal", "file"]

# Submission tool names — used to remind the agent to call them before finishing.
SUBMISSION_TOOL_NAMES = {"submit", "answer", "submit_answer"}

_SUBPROCESS_LINE_LIMIT = 10 * 1024 * 1024


def _resolve_model_hermes(
    model: str,
    provider_url: str | None,
) -> tuple[list[str], dict[str, str]]:
    """Return (extra cli args, extra env vars) for the chosen model.

    Hermes accepts either ``--model <vendor>/<name>`` (provider auto-detected)
    or ``--model <bare> --provider <provider>``. The ``provider`` is a Hermes
    ProviderConfig id — built-ins include ``anthropic``, ``openai-api``,
    ``openrouter``, ``google``, etc. See hermes_cli/auth.py PROVIDER_REGISTRY.

    A bare model with ``provider_url`` set is routed through the ``openai-api``
    provider (any OpenAI-compatible endpoint). We set ``OPENAI_API_KEY`` and
    ``OPENAI_BASE_URL`` so the SDK targets the custom host.
    """
    args = ["--model", model]
    env: dict[str, str] = {}
    if "/" not in model:
        # Bare model with a custom URL → OpenAI-compatible endpoint.
        if provider_url:
            args.extend(["--provider", "openai-api"])
            env["OPENAI_BASE_URL"] = provider_url.rstrip("/")
        else:
            # Bare name without a URL: let hermes auto-detect via its
            # catalog / model aliases.
            pass
    return args, env


def _build_mcp_tool_prompt(env_tool_names: list[str]) -> str:
    """Build the MCP-tools section appended to the user prompt.

    Hermes exposes MCP tools as ``mcp_openreward_<name>``. Tell the model
    which tools are available and that built-in terminal/file tools are
    disabled.
    """
    available = [n for n in env_tool_names if n.lower() not in ALWAYS_USE_BUILTIN]
    if not available:
        return ""

    lines = [
        "",
        "# OpenReward Environment Tools",
        "",
        "You are solving a task in an OpenReward environment. The environment "
        "provides tools via an MCP server named 'openreward'. Hermes exposes "
        "each MCP tool with the prefix `mcp_openreward_`. Use ONLY these tools "
        "for environment interactions — Hermes's built-in terminal and file "
        "toolsets are disabled.",
        "",
        "Available MCP tools:",
    ]
    for name in available:
        lines.append(f"- `{_MCP_TOOL_PREFIX}{name}`")
    lines.append("")
    lines.append(
        "When a tool result contains [EPISODE COMPLETE], stop working "
        "immediately — the task is done. Do not make any more tool calls "
        "after seeing [EPISODE COMPLETE]."
    )
    return "\n".join(lines)


def _build_submission_reminder(env_tool_names: list[str]) -> str:
    submission_tools = [
        f"`{_MCP_TOOL_PREFIX}{n}`"
        for n in env_tool_names
        if n.lower() in SUBMISSION_TOOL_NAMES
    ]
    if not submission_tools:
        return ""
    return (
        "Submission Reminder:\n"
        f"This environment has a submission tool: {', '.join(submission_tools)}. "
        f"You MUST call this tool to submit your final answer before you finish. "
        f"If you stop without calling it, your work will not be scored."
    )


def _sanitize_prompt(text: str) -> str:
    if text.startswith("-"):
        return " " + text
    return text


def _build_hermes_config(mcp_env: dict[str, str]) -> dict[str, Any]:
    """Build the ~/.hermes/config.yaml content for an isolated run."""
    return {
        "mcp_servers": {
            _MCP_SERVER_NAME: {
                "command": sys.executable,
                "args": ["-m", "firehorse.mcp"],
                "env": mcp_env,
                "enabled": True,
            }
        },
        "agent": {
            "disabled_toolsets": list(_DISABLED_TOOLSETS),
        },
    }


def _replay_toolcalls_fallback(
    toolcalls_path: Path,
    rollout: Any,
) -> int:
    """Last-resort rollout replay using only the MCP bridge's toolcalls JSONL.

    Used when ``hermes sessions export`` fails — gives an incomplete trace
    (no agent built-in tool calls, no reasoning, no inter-turn assistant
    text) but at least preserves the environment-side interactions.
    """
    if not toolcalls_path.exists():
        return 0
    count = 0
    with toolcalls_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                evt = json.loads(line)
            except json.JSONDecodeError:
                continue
            call_id = evt.get("call_id", "")
            tool = evt.get("tool", "")
            args = evt.get("arguments") or {}
            result = evt.get("result", "")
            reward = evt.get("reward")
            finished = bool(evt.get("finished", False))
            try:
                rollout.log(ToolCall(
                    name=f"{_MCP_TOOL_PREFIX}{tool}",
                    content=json.dumps(args),
                    call_id=call_id,
                ))
                rollout.log(
                    ToolResult(
                        content=strip_or_reward_marker(str(result)),
                        call_id=call_id,
                    ),
                    reward=reward,
                    is_finished=finished,
                )
                count += 1
            except Exception as e:
                print(
                    f"[hermes] fallback rollout.log failed for {tool}: "
                    f"{type(e).__name__}: {e}",
                    file=sys.stderr,
                )
    return count


async def _export_hermes_session(
    hermes_home: Path,
    session_id: str,
) -> dict | None:
    """Run ``hermes sessions export`` against the given HERMES_HOME and return
    the parsed JSON object, or None on failure.

    Hermes' ``-Q`` mode emits no per-turn events to stdout. The full
    transcript — system prompt, user/assistant/tool messages, reasoning
    blocks, tool calls and their arguments — is persisted to
    ``$HERMES_HOME/state.db`` and only surfaced via this command. We call it
    before the agent's tempdir cleanup so the SQLite store still exists.
    """
    proc = await asyncio.create_subprocess_exec(
        _HERMES_BIN, "sessions", "export", "-", "--session-id", session_id,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env={**os.environ, "HERMES_HOME": str(hermes_home)},
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        print(
            f"[hermes] sessions export failed (exit {proc.returncode}): "
            f"{stderr.decode(errors='replace').strip()[:300]}",
            file=sys.stderr,
        )
        return None
    text = stdout.decode(errors="replace").strip()
    if not text:
        return None
    # Hermes exports as a single JSON object per session, one line in
    # JSONL parlance. Robust to multi-session exports (we only requested
    # one): take the first non-blank line.
    line = text.splitlines()[0]
    try:
        return json.loads(line)
    except json.JSONDecodeError as e:
        print(f"[hermes] sessions export parse failed: {e}", file=sys.stderr)
        return None


def _replay_hermes_session_to_rollout(
    session: dict,
    rollout: Any,
    skip_first_user: bool = True,
) -> int:
    """Replay a hermes sessions-export object into the OpenReward rollout.

    Hermes' export schema (one JSON object per session) has a ``messages`` list
    where each entry has:
      - ``role``: "user" | "assistant" | "tool"
      - ``content``: text
      - ``reasoning`` / ``reasoning_content``: thinking blocks (assistant only)
      - ``tool_calls``: [{call_id, function: {name, arguments}}] (assistant)
      - ``tool_call_id`` / ``tool_name`` (tool role)

    Walks them in order and logs:
      - assistant ``reasoning`` → ``ReasoningItem``
      - assistant ``content`` → ``AssistantMessage`` (when non-empty)
      - assistant ``tool_calls[*]`` → ``ToolCall``
      - tool role → ``ToolResult`` (with ``[OR_REWARD:{...}]`` parsed for
        ``reward`` + ``is_finished``)

    ``skip_first_user`` defaults True — firehorse already logged the prompt
    as the initial ``UserMessage``; the export's first user message is the
    same prompt and would be a duplicate.

    Returns the number of items logged.
    """
    messages = session.get("messages") or []
    count = 0
    first_user_seen = False
    for msg in messages:
        role = msg.get("role")
        if role == "user":
            if skip_first_user and not first_user_seen:
                first_user_seen = True
                continue
            content = msg.get("content") or ""
            if content:
                try:
                    rollout.log(UserMessage(content=content))
                    count += 1
                except Exception as e:
                    print(f"[hermes] rollout UserMessage failed: {e}", file=sys.stderr)
        elif role == "assistant":
            reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
            if reasoning:
                try:
                    rollout.log(ReasoningItem(content=reasoning))
                    count += 1
                except Exception as e:
                    print(f"[hermes] rollout ReasoningItem failed: {e}", file=sys.stderr)
            content = msg.get("content") or ""
            if content:
                try:
                    rollout.log(AssistantMessage(content=content))
                    count += 1
                except Exception as e:
                    print(f"[hermes] rollout AssistantMessage failed: {e}", file=sys.stderr)
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function") or {}
                name = fn.get("name") or tc.get("name") or ""
                args = fn.get("arguments")
                if args is None:
                    args = "{}"
                elif not isinstance(args, str):
                    args = json.dumps(args)
                call_id = tc.get("call_id") or tc.get("id") or ""
                try:
                    rollout.log(ToolCall(name=name, content=args, call_id=call_id))
                    count += 1
                except Exception as e:
                    print(
                        f"[hermes] rollout ToolCall failed for {name}: {e}",
                        file=sys.stderr,
                    )
        elif role == "tool":
            content_str = msg.get("content") or ""
            if not isinstance(content_str, str):
                content_str = json.dumps(content_str)
            # Match the *last* [OR_REWARD:{...}] tag; the bridge appends its
            # authoritative marker after stripping env-supplied ones.
            reward, finished = parse_or_reward_marker(content_str)
            try:
                rollout.log(
                    ToolResult(
                        content=strip_or_reward_marker(content_str),
                        call_id=msg.get("tool_call_id") or "",
                    ),
                    reward=reward,
                    is_finished=finished,
                )
                count += 1
            except Exception as e:
                print(
                    f"[hermes] rollout ToolResult failed for "
                    f"{msg.get('tool_name', '?')}: {e}",
                    file=sys.stderr,
                )
    return count


class HermesAgent(BaseAgent):

    @property
    def name(self) -> str:
        return "hermes"

    async def setup(self) -> None:
        proc = await asyncio.create_subprocess_exec(
            _HERMES_BIN, "--version",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(
                "Hermes Agent CLI not found. Install via: "
                "uv tool install hermes-agent  (or see "
                "https://github.com/NousResearch/hermes-agent)"
            )
        version = stdout.decode().strip().splitlines()[0] if stdout else ""
        print(f"Hermes Agent version: {version}", file=sys.stderr)

    async def run(self, ctx: TrialContext) -> AgentResult:
        start_time = time.monotonic()

        with tempfile.TemporaryDirectory(prefix="orwd-hermes-trial-") as tmpdir:
            tmppath = Path(tmpdir)
            result_file = tmppath / "result.json"
            toolcalls_file = tmppath / "toolcalls.jsonl"
            trial_id = ctx.task_spec.get(
                "id", ctx.task_spec.get(
                    "index",
                    ctx.task_index if ctx.task_index is not None else "unknown",
                ),
            )
            log_dir = Path(ctx.output_dir) if ctx.output_dir else None

            env_tool_names = [t.name for t in ctx.tools]

            # Build MCP env vars (same shape as claude-code/gemini).
            session_task = ctx.session.task
            mcp_env: dict[str, str] = {
                "OPENREWARD_API_KEY": os.environ.get("OPENREWARD_API_KEY", ""),
                "OPENREWARD_ENV_NAME": ctx.env_name,
                "OPENREWARD_TASK_SPEC": json.dumps(dict(ctx.task_spec)),
                "OPENREWARD_TASK_SERVER_NAME": session_task.server_name,
                "OPENREWARD_TASK_ENV_NAME": session_task.environment_name,
                "OPENREWARD_TASK_NAMESPACE": session_task.namespace or "",
                "OPENREWARD_RESULT_FILE": str(result_file),
                "OPENREWARD_TOOLCALLS_FILE": str(toolcalls_file),
            }
            if ctx.toolset_name:
                mcp_env["OPENREWARD_TOOLSET_NAME"] = ctx.toolset_name
            else:
                mcp_env["OPENREWARD_TOOL_DESCRIPTIONS"] = "env"
            if os.environ.get("OPENREWARD_URL"):
                mcp_env["OPENREWARD_URL"] = os.environ["OPENREWARD_URL"]
            if ctx.secrets:
                mcp_env["OPENREWARD_SESSION_SECRETS"] = json.dumps(ctx.secrets)

            if log_dir:
                rewards_path = log_dir / f"trial_{trial_id}_rewards.jsonl"
                mcp_env["OPENREWARD_REWARDS_FILE"] = str(rewards_path)

            exclude_tools = [n for n in env_tool_names if n.lower() in ALWAYS_USE_BUILTIN]
            if exclude_tools:
                mcp_env["OPENREWARD_EXCLUDE_TOOLS"] = ",".join(exclude_tools)

            # Isolated HERMES_HOME with our config.yaml.
            hermes_home = tmppath / ".hermes"
            hermes_home.mkdir()
            hermes_config = _build_hermes_config(mcp_env)
            (hermes_home / "config.yaml").write_text(
                yaml.safe_dump(hermes_config, sort_keys=False)
            )

            # Resolve model and provider routing.
            model_args, extra_env = _resolve_model_hermes(ctx.model, ctx.provider_url)

            # Build prompt: MCP tool list + termination + submission reminder.
            mcp_section = _build_mcp_tool_prompt(env_tool_names)
            termination = (
                "Termination Instructions:\n"
                "When a tool result contains [EPISODE COMPLETE], stop working "
                "immediately — the task is done. Do not make any more tool calls "
                "after seeing [EPISODE COMPLETE]."
            )
            full_prompt = f"{ctx.prompt_text}\n\n{termination}"
            if mcp_section:
                full_prompt = f"{full_prompt}\n{mcp_section}"
            submission_reminder = _build_submission_reminder(env_tool_names)
            if submission_reminder:
                full_prompt = f"{full_prompt}\n\n{submission_reminder}"
            full_prompt = _sanitize_prompt(full_prompt)

            cmd = [
                _HERMES_BIN, "chat",
                "-q", full_prompt,
                "-Q",          # quiet / programmatic mode
                "--yolo",      # skip approval prompts
                "--ignore-rules",  # skip AGENTS.md / .cursorrules / memory injection
                *model_args,
            ]
            if ctx.max_turns:
                cmd.extend(["--max-turns", str(ctx.max_turns)])

            proc_env = {
                **os.environ,
                "HERMES_HOME": str(hermes_home),
                **extra_env,
            }
            # Same MCP_TIMEOUT guard as claude-code — the python-based MCP
            # bridge takes a few seconds to import openreward + connect on cold
            # start, especially on Windows.
            proc_env.setdefault("MCP_TIMEOUT", "120000")

            print(
                f"[hermes] Launching with model={ctx.model} "
                f"(HERMES_HOME={hermes_home})",
                file=sys.stderr,
            )

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=proc_env,
                limit=_SUBPROCESS_LINE_LIMIT,
            )

            main_log = None
            if log_dir:
                main_log = open(log_dir / f"trial_{trial_id}.jsonl", "w")

            main_rollout = None
            if ctx.logging and ctx.rollout_client:
                try:
                    model_short = ctx.model.split("/")[-1]
                    from firehorse.rollout_replay import resume_metadata
                    main_rollout = ctx.rollout_client.rollout.create(
                        run_name=ctx.run_name,
                        rollout_name=f"hermes_{model_short}_{trial_id}",
                        environment=ctx.env_name,
                        variant=ctx.variant,
                        split=ctx.split,
                        task_spec=ctx.task_spec,
                        metadata={
                            "effort": ctx.effort,
                            "model": ctx.model,
                            "agent": "hermes",
                            **resume_metadata(),
                        },
                    )
                    print(
                        f"[hermes] Rollout: "
                        f"https://openreward.ai/rollout/{main_rollout.event_id}",
                        file=sys.stderr,
                    )
                except Exception as e:
                    print(f"[hermes] Failed to create rollout: {e}", file=sys.stderr)

            prompt_event = {
                "type": "openreward_prompt",
                "system_prompt": "default",
                "environment_prompt": full_prompt,
            }
            if main_log:
                main_log.write(json.dumps(prompt_event) + "\n")

            if main_rollout:
                main_rollout.log(
                    UserMessage(content=full_prompt),
                    rollout_info=RolloutInfo(task_index=ctx.task_index, harness="hermes"),
                )
                try:
                    from firehorse.rollout_replay import maybe_replay_into
                    maybe_replay_into(main_rollout)
                except Exception as _e:
                    print(
                        f"[hermes] rollout-message replay failed: "
                        f"{type(_e).__name__}: {_e}",
                        file=sys.stderr,
                    )

            stdout_chunks: list[str] = []
            stderr_chunks: list[str] = []
            run_started = time.monotonic()
            heartbeat_stop = asyncio.Event()

            async def read_stdout() -> None:
                assert proc.stdout is not None
                async for line in proc.stdout:
                    text = line.decode(errors="replace")
                    stdout_chunks.append(text)
                    if main_log:
                        main_log.write(strip_or_reward_marker(text.rstrip("\n")) + "\n")

            async def read_stderr() -> None:
                assert proc.stderr is not None
                async for line in proc.stderr:
                    text = line.decode(errors="replace")
                    stderr_chunks.append(text)
                    s = text.strip()
                    if s and (
                        "[openreward-bridge]" in s
                        or "Error" in s
                        or "error" in s
                        or s.startswith("session_id:")
                    ):
                        print(f"  {s}", file=sys.stderr)

            async def heartbeat() -> None:
                """Tail toolcalls_file and emit one heartbeat per MCP call.

                Hermes's `-Q` mode emits nothing during the run, so without
                this the trial looks hung. The MCP bridge writes a JSONL line
                per call — that's our signal.
                """
                seen = 0
                while not heartbeat_stop.is_set():
                    try:
                        await asyncio.wait_for(heartbeat_stop.wait(), timeout=5.0)
                        return
                    except asyncio.TimeoutError:
                        pass
                    if not toolcalls_file.exists():
                        continue
                    try:
                        with toolcalls_file.open() as f:
                            lines = f.readlines()
                    except OSError:
                        continue
                    while seen < len(lines):
                        try:
                            evt = json.loads(lines[seen])
                        except json.JSONDecodeError:
                            seen += 1
                            continue
                        seen += 1
                        elapsed = time.monotonic() - run_started
                        print(
                            f"[hermes][{trial_id}] turn {seen} "
                            f"({elapsed:5.0f}s) → {evt.get('tool', '?')}",
                            file=sys.stderr,
                            flush=True,
                        )

            heartbeat_task = asyncio.create_task(heartbeat())
            try:
                await asyncio.gather(read_stdout(), read_stderr())
                await proc.wait()
            except Exception as e:
                if proc.returncode is None:
                    proc.kill()
                    await proc.wait()
                heartbeat_stop.set()
                await heartbeat_task
                return AgentResult(success=False, error=str(e))
            finally:
                heartbeat_stop.set()
                try:
                    await heartbeat_task
                except Exception:
                    pass

                duration_ms = int((time.monotonic() - start_time) * 1000)

                result_data = None
                if result_file.exists():
                    try:
                        result_data = json.loads(result_file.read_text())
                    except json.JSONDecodeError:
                        pass

                final_text = "".join(stdout_chunks).strip()

                # Parse session_id off stderr (hermes -Q prints
                # "session_id: <id>" before exit). Used to export the SQLite
                # transcript while HERMES_HOME still exists.
                session_id: str | None = None
                for line in reversed(stderr_chunks):
                    m = re.search(r"session_id:\s*([A-Za-z0-9_]+)", line)
                    if m:
                        session_id = m.group(1)
                        break

                # Export hermes' full session transcript (assistant text,
                # reasoning, every tool call incl. built-ins) and replay it
                # into the OpenReward rollout. Falls back to the bridge's
                # MCP-only toolcalls log if the export fails for any reason.
                replayed = 0
                session_export: dict | None = None
                if session_id:
                    try:
                        session_export = await _export_hermes_session(
                            hermes_home, session_id,
                        )
                    except Exception as e:
                        print(
                            f"[hermes] sessions export raised: "
                            f"{type(e).__name__}: {e}",
                            file=sys.stderr,
                        )
                    if session_export is not None and log_dir:
                        export_path = log_dir / f"trial_{trial_id}_hermes_session.json"
                        try:
                            export_path.write_text(
                                json.dumps(session_export, indent=2)
                            )
                        except OSError as e:
                            print(
                                f"[hermes] failed to persist session export: {e}",
                                file=sys.stderr,
                            )

                if main_rollout:
                    if session_export is not None:
                        replayed = _replay_hermes_session_to_rollout(
                            session_export, main_rollout, skip_first_user=True,
                        )
                    else:
                        # Best-effort fallback: at least replay the MCP
                        # tool calls + final assistant text we captured.
                        print(
                            "[hermes] session export unavailable; falling back "
                            "to MCP-only rollout replay",
                            file=sys.stderr,
                        )
                        replayed = _replay_toolcalls_fallback(
                            toolcalls_file, main_rollout,
                        )
                        if final_text:
                            try:
                                main_rollout.log(AssistantMessage(content=final_text))
                            except Exception as e:
                                print(
                                    f"[hermes] rollout assistant-message log "
                                    f"failed: {type(e).__name__}: {e}",
                                    file=sys.stderr,
                                )

                if main_log:
                    summary_event = {
                        "type": "openreward_summary",
                        "task_spec": ctx.task_spec,
                        "env": ctx.env_name,
                        "model": ctx.model,
                        "bridge_result": result_data,
                        "stdout_final": final_text,
                        "session_id": session_id,
                        "rollout_items_replayed": replayed,
                        "usage": {"duration_ms": duration_ms},
                    }
                    main_log.write(json.dumps(summary_event) + "\n")
                    main_log.close()

                if log_dir:
                    trial_result = {
                        "task_id": trial_id,
                        "task_spec": ctx.task_spec,
                        "environment": ctx.env_name,
                        "agent": "hermes",
                        "model": ctx.model,
                        "split": ctx.split,
                        "final_reward": (
                            result_data.get("last_reward") if result_data else None
                        ),
                        "finished": (
                            result_data.get("finished", False) if result_data else False
                        ),
                        "total_reward": (
                            result_data.get("total_reward") if result_data else None
                        ),
                        "tool_calls": (
                            result_data.get("calls") if result_data else replayed
                        ),
                        "duration_seconds": duration_ms / 1000,
                        "error": None,
                        "rollout_url": (
                            f"https://openreward.ai/rollout/{main_rollout.event_id}"
                            if main_rollout else None
                        ),
                    }
                    result_json_path = log_dir / f"trial_{trial_id}_result.json"
                    result_json_path.write_text(json.dumps(trial_result, indent=2))

            if result_data is not None:
                return AgentResult(
                    success=True,
                    reward=result_data.get("last_reward"),
                    finished=result_data.get("finished", False),
                    turns_used=result_data.get("calls", replayed),
                    duration_ms=duration_ms,
                )

            stderr_tail = "".join(stderr_chunks).strip().splitlines()[-5:]
            return AgentResult(
                success=False,
                error=(
                    f"No result file produced. Exit code: {proc.returncode}. "
                    f"stderr tail: {' | '.join(stderr_tail)}"
                ),
                turns_used=replayed,
                duration_ms=duration_ms,
            )
