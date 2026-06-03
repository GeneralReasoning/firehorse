from __future__ import annotations

import asyncio
import random
import sys
import time
from typing import Any, TYPE_CHECKING

import aiohttp

from firehorse.agents.base import AgentResult, TrialContext
from firehorse.config import TrialConfig
from firehorse.results import TrialResult

if TYPE_CHECKING:
    from openreward.api.environments.client import AsyncEnvironment
    from openreward.api.environments.types import Task
    from firehorse.agents.base import BaseAgent


def _is_toolset_rejection(exc: BaseException) -> bool:
    """Detect a session-open failure caused by an unsupported toolset.

    OpenReward rejects toolsets in two layers:
      - client-side ``ValueError`` from ``_validate_toolset_name`` when the name
        isn't a known built-in. Our agent→toolset map only emits known names,
        so this branch is mostly defensive.
      - server-side HTTP 400 (``aiohttp.ClientResponseError``) when the env's
        ``Toolset.__init__`` raises — most commonly because the env doesn't
        declare ``self.sandbox`` and the requested toolset requires one.

    Both paths put the substring ``"toolset"`` (or ``"sandbox"``) in their
    message. Matching on that keeps the check tight without exporting a new
    exception type from the SDK.
    """
    if isinstance(exc, aiohttp.ClientResponseError):
        if exc.status != 400:
            return False
        msg = (exc.message or "").lower()
        return "toolset" in msg or "sandbox" in msg
    if isinstance(exc, ValueError):
        return "toolset" in str(exc).lower()
    return False

_MCP_RETRY_MAX = 8
_MCP_RETRY_BASE_DELAY = 2.0  # seconds, exponential backoff: 2s, 4s, 8s


def _is_mcp_failure(result: AgentResult) -> bool:
    """Check if an agent result indicates a transient MCP connection failure."""
    if result.success or result.error is None:
        return False
    return "MCP" in result.error and "failed to connect" in result.error


async def run_trial(
    env: AsyncEnvironment,
    task: Task,
    agent: BaseAgent,
    config: TrialConfig,
    rollout_client: Any = None,
) -> TrialResult:
    start = time.monotonic()

    # Map agent names to SDK toolset names. These toolsets provide optimized
    # tool descriptions for each agent type. A non-None config.toolset always
    # wins (incl. the empty string, which means "no toolset" — use the env's
    # own tools directly, for envs without a sandbox).
    _AGENT_TO_TOOLSET = {
        "claude-code": "claude-code",
        "codex": "codex",
        "gemini": "gemini-cli",
        "hermes": "hermes",
    }
    if config.toolset is None:
        toolset_name = _AGENT_TO_TOOLSET.get(agent.name)
    else:
        toolset_name = config.toolset or None

    try:
        session_secrets = config.secrets or None
        # Open the session with the resolved toolset, falling back to no
        # toolset if the env doesn't support the one we asked for. The
        # ``__aenter__`` is where the server-side validation fires (Task
        # creation + ``Toolset(env)`` instantiation), so the try/except has
        # to wrap entry, not construction.
        session_cm = env.session(
            task, secrets=session_secrets, toolset=toolset_name,
        )
        try:
            session = await session_cm.__aenter__()
        except BaseException as enter_exc:
            if toolset_name is None or not _is_toolset_rejection(enter_exc):
                raise
            print(
                f"[trial] toolset {toolset_name!r} not supported by env "
                f"{config.env!r} ({enter_exc}); retrying with no toolset",
                file=sys.stderr,
            )
            toolset_name = None
            session_cm = env.session(
                task, secrets=session_secrets, toolset=None,
            )
            session = await session_cm.__aenter__()
        try:
            prompt_blocks = await session.get_prompt()
            tools = await session.list_tools()

            parts = []
            for block in prompt_blocks:
                if hasattr(block, "text"):
                    parts.append(block.text)
            prompt_text = "\n\n".join(parts)

            ctx = TrialContext(
                prompt_text=prompt_text,
                tools=tools,
                session=session,
                model=config.model,
                env_name=config.env,
                task_spec=config.task_spec,
                run_name=config.run_name,
                split=config.split,
                variant=config.variant,
                task_index=config.task_index,
                max_turns=config.max_turns,
                provider_url=config.provider_url,
                disable_builtin_tools=config.disable_builtin_tools,
                secrets=config.secrets,
                output_dir=config.output_dir,
                effort=config.effort,
                logging=config.logging,
                use_builtin_descriptions=config.use_builtin_descriptions,
                use_all_filesystem_tools=config.use_all_filesystem_tools,
                plan_mode=config.plan_mode,
                toolset_name=toolset_name,
                rollout_client=rollout_client,
            )

            result = await agent.run(ctx)

            attempt = 1
            while _is_mcp_failure(result) and attempt < _MCP_RETRY_MAX:
                delay = _MCP_RETRY_BASE_DELAY * (2 ** (attempt - 1)) + random.uniform(0, 2.0)
                print(
                    f"[trial] MCP connection failed (attempt {attempt}/{_MCP_RETRY_MAX}), "
                    f"retrying in {delay:.0f}s...",
                    file=sys.stderr,
                )
                await asyncio.sleep(delay)
                attempt += 1
                result = await agent.run(ctx)
        finally:
            await session_cm.__aexit__(None, None, None)

        duration = time.monotonic() - start
        return TrialResult(
            task_index=config.task_index,
            task_spec=config.task_spec,
            success=result.success,
            reward=result.reward,
            finished=result.finished,
            turns_used=result.turns_used,
            error=result.error,
            duration_seconds=duration,
            cost_usd=result.cost_usd,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
        )
    except Exception as e:
        duration = time.monotonic() - start
        return TrialResult(
            task_index=config.task_index,
            task_spec=config.task_spec,
            success=False,
            error=str(e),
            duration_seconds=duration,
        )
