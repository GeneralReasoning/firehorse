"""MCP stdio server that bridges OpenReward session tools to Claude Code.

Launched as a subprocess by ClaudeCodeAgent. Receives configuration
via environment variables, creates an OpenReward session, and exposes
the session's tools over MCP stdio transport.
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import sys
import time
from typing import Any

from mcp.server.lowlevel.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolResult,
    TextContent,
    Tool,
)

from openreward.client import AsyncOpenReward
from openreward.api.environments.types import (
    Task,
    ToolCallError,
    ToolSpec,
)
from firehorse.mcp.builtin_descriptions import BUILTIN_DESCRIPTIONS
from firehorse.mcp.codex_descriptions import CODEX_DESCRIPTIONS
from firehorse.mcp.convert import strip_bridge_markers, toolspec_to_mcp, tooloutput_to_mcp


class OpenRewardBridge:

    def __init__(self):
        self.server = Server("openreward-bridge")
        self._client: AsyncOpenReward | None = None
        self._session: Any = None  # AsyncSession
        self._session_entered = False
        self._rewards_file: Any = None  # file handle for rewards sidecar

        self._toolset_name: str | None = None
        self._tool_prefix: str = os.environ.get("OPENREWARD_TOOL_PREFIX", "")
        self._prebuilt_tools: list[Tool] | None = self._load_prebuilt_tools()
        self._toolcalls_file: Any = None  # file handle for tool-call log
        self.tools: list[ToolSpec] = []
        self.finished = False
        self.last_reward: float | None = None
        self.total_reward: float = 0.0
        self.call_count: int = 0

        self._initialized = False

        self.server.list_tools()(self._list_tools)
        self.server.call_tool(validate_input=False)(self._call_tool)
        # Codex CLI probes resources/list, resources/templates/list, and
        # prompts/list during MCP startup. We don't expose any of those, but
        # without explicit handlers the requests time out (30s), and the
        # error messages leak into the agent's own captured output where the
        # model sees them and wastes tokens trying to debug. Register empty
        # handlers so the probes return immediately.
        self.server.list_resources()(self._list_resources_empty)
        self.server.list_resource_templates()(self._list_resource_templates_empty)
        self.server.list_prompts()(self._list_prompts_empty)

    async def _list_resources_empty(self) -> list:
        return []

    async def _list_resource_templates_empty(self) -> list:
        return []

    async def _list_prompts_empty(self) -> list:
        return []

    async def initialize(self):
        env_name = os.environ["OPENREWARD_ENV_NAME"]
        task_spec_str = os.environ.get("OPENREWARD_TASK_SPEC", "{}")
        task_spec = json.loads(task_spec_str)
        secrets_str = os.environ.get("OPENREWARD_SESSION_SECRETS", "{}")
        secrets = json.loads(secrets_str) or None

        self._toolset_name = os.environ.get("OPENREWARD_TOOLSET_NAME") or None

        self._client = AsyncOpenReward()
        env = self._client.environments.get(env_name)

        # Use the exact task fields from the trial runner's session
        server_name = os.environ.get("OPENREWARD_TASK_SERVER_NAME", "")
        task_env_name = os.environ.get("OPENREWARD_TASK_ENV_NAME", "")
        namespace = os.environ.get("OPENREWARD_TASK_NAMESPACE", "") or None

        # Fall back to parsing env_name if specific fields not provided
        if not server_name:
            parts = env_name.split("/", maxsplit=1)
            if len(parts) == 2:
                namespace, server_name = parts
            else:
                server_name = parts[0]
            task_env_name = server_name

        task = Task(
            server_name=server_name,
            environment_name=task_env_name,
            task_spec=task_spec,
            namespace=namespace,
        )

        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                self._session = env.session(task, secrets, toolset=self._toolset_name)
                await self._session.__aenter__()
                self._session_entered = True
                all_tools = await self._session.list_tools()
                break
            except Exception as e:
                if attempt == max_attempts:
                    raise
                delay = 2 ** attempt + random.uniform(0, 1.0)
                print(
                    f"[openreward-bridge] Init failed (attempt {attempt}/{max_attempts}): {e}, "
                    f"retrying in {delay:.1f}s",
                    file=sys.stderr,
                )
                if self._session_entered:
                    try:
                        await self._session.__aexit__(None, None, None)
                    except Exception as cleanup_err:
                        print(
                            f"[openreward-bridge] Session cleanup failed during retry: {cleanup_err}",
                            file=sys.stderr,
                        )
                    self._session_entered = False
                await asyncio.sleep(delay)

        # Filter out env tools that duplicate Claude's built-in planning tools.
        exclude = os.environ.get("OPENREWARD_EXCLUDE_TOOLS", "").split(",")
        exclude_lower = {n.strip().lower() for n in exclude if n.strip()}
        self.tools = [t for t in all_tools if t.name.lower() not in exclude_lower]

        if len(self.tools) < len(all_tools):
            excluded = [t.name for t in all_tools if t.name.lower() in exclude_lower]
            print(f"[openreward-bridge] Excluded env tools (Claude built-in preferred): {excluded}", file=sys.stderr)

        # Open rewards sidecar JSONL
        rewards_path = os.environ.get("OPENREWARD_REWARDS_FILE")
        if rewards_path:
            self._rewards_file = open(rewards_path, "w")

        # Open tool-call log JSONL (records name, arguments, result for rollout)
        toolcalls_path = os.environ.get("OPENREWARD_TOOLCALLS_FILE")
        if toolcalls_path:
            self._toolcalls_file = open(toolcalls_path, "w")

        print(f"[openreward-bridge] Session created, {len(self.tools)} tools available", file=sys.stderr)

        # Resume replay: if OPENREWARD_REPLAY_PATH points to a JSON file
        # produced by `firehorse resume`/`firehorse replay`, fast-replay
        # each prior successful tool call against this fresh OR session
        # so its env state matches the dead session before the agent's
        # first turn. See firehorse/resume.py for the manifest schema.
        #
        # We also set OPENREWARD_REPLAY_MODE=1 in the env process for
        # the duration of the replay so env code can detect it and
        # short-circuit any non-idempotent side effects (logging,
        # external webhooks, telemetry, etc.).
        replay_path = os.environ.get("OPENREWARD_REPLAY_PATH")
        if replay_path:
            try:
                manifest = json.loads(open(replay_path, encoding="utf-8").read())
                calls = manifest.get("tool_calls", [])
                # If firehorse passed OPENREWARD_REPLAY_PROGRESS_FILE,
                # mirror each replay event to that file as a JSONL
                # record. Firehorse tails it in real time and surfaces
                # `[REPLAY N/M]` lines in the user's terminal — codex
                # buffers MCP subprocess stderr too aggressively for
                # the bridge's own prints to surface live.
                _progress_path = os.environ.get("OPENREWARD_REPLAY_PROGRESS_FILE")
                _progress_fh = None
                if _progress_path:
                    try:
                        _progress_fh = open(_progress_path, "w", buffering=1)
                        _progress_fh.write(json.dumps({
                            "event": "begin",
                            "total": len(calls),
                        }) + "\n")
                    except Exception as _e:
                        print(
                            f"[openreward-bridge] could not open progress file "
                            f"{_progress_path}: {type(_e).__name__}: {_e}",
                            file=sys.stderr, flush=True,
                        )
                        _progress_fh = None

                print(
                    f"[openreward-bridge] REPLAY MODE BEGIN — "
                    f"replaying {len(calls)} prior tool calls from "
                    f"{replay_path} (no LLM in the loop)",
                    file=sys.stderr, flush=True,
                )
                # Hint to env code that we're not driving the agent right now.
                # Env can check os.environ.get("OPENREWARD_REPLAY_MODE") == "1"
                # to skip side effects.
                os.environ["OPENREWARD_REPLAY_MODE"] = "1"
                # Only env-side tools can be replayed against OR's session.
                # The trial JSONL also records codex built-ins (bash,
                # multi_edit, ls, ...) — those are agent-local and OR has
                # no handler, so sending them produces "unknown tool"
                # entries that pollute the replay rollout. Filter to the
                # set OR actually exposes.
                or_tool_names = {t.name for t in self.tools}
                try:
                    ok = 0
                    fail = 0
                    skipped = 0
                    for i, tc in enumerate(calls):
                        tool = tc.get("tool", "")
                        args = tc.get("arguments", {}) or {}
                        if tool not in or_tool_names:
                            skipped += 1
                            print(
                                f"[openreward-bridge] REPLAY {i+1:>4}/{len(calls)} "
                                f"tool={tool} SKIP (agent-side, not an OR tool)",
                                file=sys.stderr, flush=True,
                            )
                            if _progress_fh is not None:
                                _progress_fh.write(json.dumps({
                                    "event": "call",
                                    "i": i + 1,
                                    "total": len(calls),
                                    "tool": tool,
                                    "ok": True,
                                    "skipped": True,
                                    "reason": "not an OR tool",
                                }) + "\n")
                            continue
                        try:
                            result = await self._session.call_tool(tool, args)
                            ok += 1
                            print(
                                f"[openreward-bridge] REPLAY {i+1:>4}/{len(calls)} "
                                f"tool={tool} ok",
                                file=sys.stderr, flush=True,
                            )
                            if _progress_fh is not None:
                                _progress_fh.write(json.dumps({
                                    "event": "call",
                                    "i": i + 1,
                                    "total": len(calls),
                                    "tool": tool,
                                    "ok": True,
                                }) + "\n")
                        except Exception as e:
                            fail += 1
                            print(
                                f"[openreward-bridge] REPLAY: call {i+1}/{len(calls)} "
                                f"({tool}) raised {type(e).__name__}: {e}",
                                file=sys.stderr, flush=True,
                            )
                            if _progress_fh is not None:
                                _progress_fh.write(json.dumps({
                                    "event": "call",
                                    "i": i + 1,
                                    "total": len(calls),
                                    "tool": tool,
                                    "ok": False,
                                    "error": f"{type(e).__name__}: {e}",
                                }) + "\n")
                finally:
                    os.environ.pop("OPENREWARD_REPLAY_MODE", None)
                if _progress_fh is not None:
                    try:
                        _progress_fh.write(json.dumps({
                            "event": "complete",
                            "ok": ok,
                            "fail": fail,
                            "skipped": skipped,
                            "total": len(calls),
                        }) + "\n")
                        _progress_fh.close()
                    except Exception:
                        pass
                print(
                    f"[openreward-bridge] REPLAY COMPLETE — ok={ok} fail={fail} "
                    f"skipped={skipped} (agent-side tools); handing off to agent",
                    file=sys.stderr, flush=True,
                )
                # Replay-only mode: caller wants the env state rebuilt
                # and that's it. Don't hand off to the agent.
                if os.environ.get("OPENREWARD_REPLAY_ONLY") == "1":
                    print(
                        "[openreward-bridge] REPLAY_ONLY=1 — exiting bridge "
                        "without serving tools.",
                        file=sys.stderr, flush=True,
                    )
                    # Caller (firehorse replay) is expected to read this
                    # signal and terminate. Sleep briefly so the message
                    # actually flushes before stdin close kills us.
                    await asyncio.sleep(0.5)
                    sys.exit(0)
            except SystemExit:
                raise
            except Exception as e:
                print(
                    f"[openreward-bridge] Replay aborted: "
                    f"{type(e).__name__}: {e}",
                    file=sys.stderr, flush=True,
                )

    def _load_prebuilt_tools(self) -> list[Tool] | None:
        """Load pre-built tool specs from env var for instant list_tools response.

        When OPENREWARD_PREBUILT_TOOLS is set (JSON array of {name, description, inputSchema}),
        list_tools returns these immediately without waiting for OpenReward init.
        This is critical for the Gemini CLI which times out if tool listing is slow.
        """
        raw = os.environ.get("OPENREWARD_PREBUILT_TOOLS")
        if not raw:
            return None
        try:
            specs = json.loads(raw)
            prefix = self._tool_prefix
            tools = []
            for s in specs:
                name = f"{prefix}{s['name']}" if prefix else s["name"]
                tools.append(Tool(
                    name=name,
                    description=s.get("description", ""),
                    inputSchema=s.get("inputSchema", {"type": "object", "properties": {}}),
                ))
            print(f"[openreward-bridge] Pre-built {len(tools)} tools for instant listing", file=sys.stderr)
            return tools
        except (json.JSONDecodeError, KeyError) as e:
            print(f"[openreward-bridge] Failed to load prebuilt tools: {e}", file=sys.stderr)
            return None

    async def _background_init(self) -> None:
        """Initialize OpenReward in background, parallel with MCP handshake."""
        try:
            print("[openreward-bridge] Starting OpenReward init...", file=sys.stderr)
            await self.initialize()
            self._initialized = True
            print(f"[openreward-bridge] Ready, {len(self.tools)} tools", file=sys.stderr)
        except Exception as e:
            import traceback
            err_msg = f"Init failed: {e}\n{traceback.format_exc()}"
            print(f"[openreward-bridge] {err_msg}", file=sys.stderr)
            # Also write to a debug file since Gemini CLI may not forward stderr
            try:
                with open("/tmp/firehorse_mcp_debug.log", "w") as f:
                    f.write(err_msg)
            except Exception:
                pass
            self._init_error = e

    async def _ensure_initialized(self) -> None:
        """Wait for background initialization to complete."""
        if not self._initialized and hasattr(self, '_init_task'):
            await self._init_task
            if hasattr(self, '_init_error'):
                raise RuntimeError(f"OpenReward init failed: {self._init_error}")

    def _prefixed_tool(self, spec: ToolSpec, description_override: str | None = None) -> Tool:
        """Convert a ToolSpec to MCP Tool, optionally prefixing the name."""
        tool = toolspec_to_mcp(spec, description_override=description_override)
        if self._tool_prefix:
            tool = Tool(
                name=f"{self._tool_prefix}{tool.name}",
                description=tool.description,
                inputSchema=tool.inputSchema,
            )
        return tool

    async def _list_tools(self) -> list[Tool]:
        # If pre-built tool specs are available (passed via env var), return
        # them immediately without waiting for OpenReward init. This avoids
        # blocking the Gemini CLI's MCP tool discovery.
        if self._prebuilt_tools is not None:
            return self._prebuilt_tools

        await self._ensure_initialized()

        if self._toolset_name is not None:
            # Session toolset provides correct descriptions; no manual overrides needed
            return [self._prefixed_tool(t) for t in self.tools]

        # Legacy path: manual description overrides when no toolset is used
        variant = os.environ.get("OPENREWARD_TOOL_DESCRIPTIONS", "claude")
        if variant == "claude":
            descs = BUILTIN_DESCRIPTIONS
        elif variant == "codex":
            descs = CODEX_DESCRIPTIONS
        else:
            # "env" or any other value: use environment's original descriptions
            return [self._prefixed_tool(t) for t in self.tools]
        return [
            self._prefixed_tool(t, description_override=descs.get(t.name.lower()))
            for t in self.tools
        ]

    async def _call_tool(self, name: str, arguments: dict[str, Any]) -> CallToolResult:
        await self._ensure_initialized()

        # Strip the prefix if one was applied during list_tools
        if self._tool_prefix and name.startswith(self._tool_prefix):
            name = name[len(self._tool_prefix):]

        if self.finished:
            return CallToolResult(
                content=[TextContent(type="text", text="Episode already complete. No more tool calls allowed.")],
                isError=True,
            )

        if self._session is None:
            return CallToolResult(
                content=[TextContent(type="text", text="Session not initialized.")],
                isError=True,
            )

        try:
            output = await self._session.call_tool(name, arguments)
        except ToolCallError as e:
            try:
                with open("/tmp/firehorse_mcp_debug.log", "a") as f:
                    f.write(f"ToolCallError({name}): {e}\n")
            except Exception:
                pass
            return CallToolResult(
                content=[TextContent(type="text", text=f"Tool error: {e}")],
                isError=True,
            )
        except Exception as e:
            try:
                import traceback
                with open("/tmp/firehorse_mcp_debug.log", "a") as f:
                    f.write(f"Exception({name}): {e}\n{traceback.format_exc()}\n")
            except Exception:
                pass
            return CallToolResult(
                content=[TextContent(type="text", text=f"Unexpected error: {e}")],
                isError=True,
            )

        self.call_count += 1
        raw_contents = tooloutput_to_mcp(output)

        # Coalesce env-supplied text blocks into one and strip any
        # ``[OR_REWARD:...]`` / ``[EPISODE COMPLETE]`` substrings the env may
        # have echoed back. Without this, a tool whose result mirrors the
        # caller's input (write_file, send_email, log_message, ...) is a
        # spoofing vector: the model writes a fake marker as input, the env
        # echoes it, the harness rollout logger parses the spoof and persists
        # an inflated reward to the OpenReward database. Stripping on the
        # joined text — rather than per-block — also closes the
        # marker-split-across-blocks variant.
        env_text_parts: list[str] = []
        non_text_contents: list[TextContent | ImageContent] = []
        for c in raw_contents:
            if isinstance(c, TextContent):
                env_text_parts.append(c.text)
            else:
                non_text_contents.append(c)
        joined = "".join(env_text_parts)
        scrubbed = strip_bridge_markers(joined)
        contents: list[TextContent | ImageContent] = list(non_text_contents)
        if scrubbed:
            # Single coalesced text block — keeps the joined-string property
            # the harness parsers rely on.
            contents.append(TextContent(type="text", text=scrubbed))

        # Write tool call + result to sidecar log for rollout reconstruction
        if self._toolcalls_file:
            tc_event = {
                "call_id": f"call_{self.call_count}",
                "tool": name,
                "arguments": arguments,
                "result": scrubbed,
                "reward": output.reward,
                "finished": output.finished,
            }
            self._toolcalls_file.write(json.dumps(tc_event) + "\n")
            self._toolcalls_file.flush()

        if output.reward is not None:
            self.total_reward += output.reward
            self.last_reward = output.reward

        # Write reward data to sidecar JSONL (not injected into model context)
        if self._rewards_file:
            reward_event = {
                "call_count": self.call_count,
                "tool": name,
                "reward": output.reward,
                "finished": output.finished,
                "total_reward": self.total_reward,
                "timestamp": time.time(),
            }
            self._rewards_file.write(json.dumps(reward_event) + "\n")
            self._rewards_file.flush()

        # ALWAYS append a single canonical [OR_REWARD:...] marker as the last
        # text block — even when reward is null and not finished. This is the
        # second half of the anti-spoofing contract: harness parsers do
        # "match the last marker in the text", so guaranteeing the bridge's
        # marker is the last one short-circuits any envious upstream content.
        contents.append(TextContent(
            type="text",
            text=f"\n[OR_REWARD:{json.dumps({'r': output.reward, 'f': output.finished})}]",
        ))

        if output.finished:
            self.finished = True
            contents.append(TextContent(
                type="text",
                text=f"\n\n[EPISODE COMPLETE] The environment has signaled that this task is finished. "
                     f"Final reward: {output.reward}. Stop making tool calls.",
            ))
            self._write_result_file()

        return CallToolResult(content=contents, isError=False)

    def _write_result_file(self):
        result_file = os.environ.get("OPENREWARD_RESULT_FILE")
        if not result_file:
            return
        result = {
            "finished": self.finished,
            "last_reward": self.last_reward,
            "total_reward": self.total_reward,
            "calls": self.call_count,
        }
        try:
            with open(result_file, "w") as f:
                json.dump(result, f)
        except OSError as e:
            print(f"[openreward-bridge] Failed to write result file: {e}", file=sys.stderr)

    async def shutdown(self):
        self._write_result_file()
        if self._rewards_file:
            self._rewards_file.close()
        if self._session is not None and self._session_entered:
            try:
                await self._session.__aexit__(None, None, None)
            except Exception as e:
                print(f"[openreward-bridge] Error during session teardown: {e}", file=sys.stderr)
            self._session_entered = False

    async def run(self):
        # Start OpenReward init in background — the Gemini CLI has a 5s MCP
        # timeout, so we must respond to the handshake immediately.
        self._init_task = asyncio.create_task(self._background_init())
        try:
            async with stdio_server() as (read_stream, write_stream):
                await self.server.run(
                    read_stream,
                    write_stream,
                    self._initialization_options(),
                )
        finally:
            await self.shutdown()

    def _initialization_options(self):
        """Build init options that hide resources/prompts capabilities.

        The MCP SDK auto-advertises a capability whenever a matching handler is
        registered. We register empty resources/prompts handlers only to silence
        the Codex CLI's 30s startup probe, but the advertisement is a side
        effect: Hermes (and any other capability-aware client) sees them and
        synthesizes utility tools like ``mcp_<server>_list_resources`` that the
        model then wastes turns calling. Strip both fields from the advertised
        capabilities so the handlers still respond fast (returning ``[]``) but
        clients don't expose the synthesized tools.
        """
        opts = self.server.create_initialization_options()
        caps = opts.capabilities.model_copy(update={"resources": None, "prompts": None})
        return opts.model_copy(update={"capabilities": caps})
