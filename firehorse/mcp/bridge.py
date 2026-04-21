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
from firehorse.mcp.convert import toolspec_to_mcp, tooloutput_to_mcp


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
        contents = tooloutput_to_mcp(output)

        # Write tool call + result to sidecar log for rollout reconstruction
        if self._toolcalls_file:
            result_text = "\n".join(
                c.text for c in contents if hasattr(c, "text")
            )
            tc_event = {
                "call_id": f"call_{self.call_count}",
                "tool": name,
                "arguments": arguments,
                "result": result_text,
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

        # Tag reward/finished so the rollout logger can parse it
        if output.reward is not None or output.finished:
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
                    self.server.create_initialization_options(),
                )
        finally:
            await self.shutdown()
