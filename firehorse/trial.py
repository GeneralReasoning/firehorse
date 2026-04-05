from __future__ import annotations

import asyncio
import random
import sys
import time
from typing import Any, TYPE_CHECKING

from firehorse.agents.base import AgentResult, TrialContext
from firehorse.config import TrialConfig
from firehorse.results import TrialResult

if TYPE_CHECKING:
    from openreward.api.environments.client import AsyncEnvironment
    from openreward.api.environments.types import Task
    from firehorse.agents.base import BaseAgent

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

    try:
        session_secrets = config.secrets or None
        async with env.session(task, secrets=session_secrets) as session:
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
