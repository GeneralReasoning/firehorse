from __future__ import annotations

import asyncio
import importlib.metadata
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from openreward.client import AsyncOpenReward
from openreward.models import RunInfo
from firehorse.agents import get_agent
from firehorse.agents.claude_code import ENV_TO_BUILTIN
from firehorse.config import RunConfig, TrialConfig
from firehorse.results import RunSummary, TrialResult
from firehorse.trial import run_trial

if TYPE_CHECKING:
    from openreward.api.environments.types import ToolSpec


def _print_banner(
    config: RunConfig,
    n_tasks: int,
    tools: list[ToolSpec],
) -> None:
    variant_suffix = f" variant={config.variant}" if config.variant else ""
    print(f"\nOpenReward Run: {config.env}{variant_suffix} ({config.split} split, {n_tasks} tasks)", file=sys.stderr)
    print(f"Agent: {config.agent} | Model: {config.model}", file=sys.stderr)
    print(f"Concurrency: {config.n_concurrent}", file=sys.stderr)

    replaced_builtins = []
    if tools:
        print(f"\nEnvironment tools (via MCP):", file=sys.stderr)
        for t in tools:
            builtin = ENV_TO_BUILTIN.get(t.name.lower())
            override_note = ""
            if builtin:
                override_note = f" (replaces built-in {builtin})"
                replaced_builtins.append(builtin)
            desc = t.description or ""
            if len(desc) > 80:
                desc = desc[:77] + "..."
            print(f"  - {t.name}{override_note}: {desc}", file=sys.stderr)

    if replaced_builtins:
        print(f"\nDisabled built-ins (replaced by env): {', '.join(sorted(replaced_builtins))}", file=sys.stderr)

    print(f"Remaining Claude built-in tools: available (Task, TodoWrite, etc.)", file=sys.stderr)
    print(f"\nStarting trials...\n", file=sys.stderr)


def _get_firehorse_version() -> str:
    try:
        return importlib.metadata.version("firehorse")
    except importlib.metadata.PackageNotFoundError:
        try:
            return "dev+" + subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=os.path.dirname(__file__),
                stderr=subprocess.DEVNULL,
            ).decode().strip()
        except Exception:
            return "dev"


async def run_evaluation(config: RunConfig) -> RunSummary:
    # OpenClaw does not support Anthropic's native API endpoint — Anthropic
    # models must go through OpenRouter when using the openclaw harness.
    if config.agent == "openclaw" and config.model.startswith("anthropic/"):
        print(
            f"Error: OpenClaw harness does not support the Anthropic API endpoint "
            f"(got model {config.model!r}). Route Claude models through OpenRouter, "
            f"e.g. --model openrouter/anthropic/claude-sonnet-4-6.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    # Validate provider credentials early
    if config.model.startswith("openrouter/") and not os.environ.get("OPENROUTER_API_KEY"):
        print("Error: OPENROUTER_API_KEY environment variable required for openrouter/ models", file=sys.stderr)
        raise SystemExit(1)
    if config.model.startswith("anthropic/") and not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable required for anthropic/ models", file=sys.stderr)
        raise SystemExit(1)
    if config.model.startswith("openai/") and not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable required for openai/ models", file=sys.stderr)
        raise SystemExit(1)
    if config.model.startswith("google/") and not os.environ.get("GOOGLE_API_KEY") and not os.environ.get("GEMINI_API_KEY"):
        print("Error: GOOGLE_API_KEY or GEMINI_API_KEY environment variable required for google/ models", file=sys.stderr)
        raise SystemExit(1)
    
    run_info = RunInfo(
        model_name=config.model,
        run_type="eval",
        framework="firehorse",
        framework_version=_get_firehorse_version(),
    )
    client = AsyncOpenReward(run_info=run_info)
    env = client.environments.get(config.env, variant=config.variant)

    # Count tasks (index-based API — list_tasks is unsupported on some envs)
    print(f"Listing tasks for {config.env} ({config.split} split)...", file=sys.stderr)
    total_available = await env.num_tasks(config.split)
    print(f"Found {total_available} tasks", file=sys.stderr)
    start = config.skip_tasks or 0
    stop = total_available if config.max_tasks is None else min(start + config.max_tasks, total_available)
    indices = list(range(start, stop))

    if not indices:
        print(f"No tasks found for split {config.split!r}", file=sys.stderr)
        return RunSummary(
            run_name=config.effective_run_name(),
            environment=config.env,
            agent=config.agent,
            model=config.model,
            split=config.split,
            total_tasks=0,
            completed=0,
            failed=0,
            avg_reward=None,
        )

    # Auto-detect required secrets from env vars
    secrets = dict(config.secrets)
    required = await env.list_required_secrets()
    for key in required:
        if key not in secrets:
            # Try uppercase env var (e.g. openai_api_key -> OPENAI_API_KEY)
            env_var = key.upper()
            val = os.environ.get(env_var, "")
            if val:
                secrets[key] = val
                print(f"Auto-detected secret: {key} (from ${env_var})", file=sys.stderr)
            else:
                print(f"Warning: required secret '{key}' not found in env vars", file=sys.stderr)

    # Fetch tools for the banner
    tools = await env.list_tools()
    _print_banner(config, len(indices), tools)

    # Setup agent
    agent = get_agent(config.agent)
    await agent.setup()

    run_name = config.effective_run_name()

    # Create output directory for JSONL logs
    output_dir = config.output_dir
    if output_dir is None:
        output_dir = run_name
    os.makedirs(output_dir, exist_ok=True)
    print(f"Trajectory logs: {os.path.abspath(output_dir)}/", file=sys.stderr)

    semaphore = asyncio.Semaphore(config.n_concurrent)
    completed_count = 0
    total = len(indices)

    async def run_with_semaphore(abs_index: int, local_idx: int) -> TrialResult:
        nonlocal completed_count
        async with semaphore:
            # Stagger trial starts to avoid thundering herd on MCP/API connections
            if config.n_concurrent > 1:
                jitter = random.uniform(0, config.n_concurrent * 1.0)
                await asyncio.sleep(jitter)
            task = await env.get_task(config.split, abs_index)
            trial_config = TrialConfig(
                task_index=local_idx,
                task_spec=dict(task.task_spec),
                run_name=run_name,
                env=config.env,
                split=config.split,
                model=config.model,
                variant=config.variant,
                max_turns=config.max_turns,
                provider_url=config.provider_url,
                disable_builtin_tools=config.disable_builtin_tools,
                secrets=secrets,
                output_dir=output_dir,
                effort=config.effort,
                logging=config.logging,
                use_builtin_descriptions=config.use_builtin_descriptions,
                use_all_filesystem_tools=config.use_all_filesystem_tools,
                plan_mode=config.plan_mode,
            )
            result = await run_trial(
                env, task, agent, trial_config,
                rollout_client=client if config.logging else None,
            )
            completed_count += 1
            reward_str = f"{result.reward:.3f}" if result.reward is not None else "N/A"
            if result.error:
                status = "ERROR"
            elif result.finished:
                status = "done"
            elif result.success:
                status = "timeout"  # Agent ran but env didn't signal finished
            else:
                status = "FAIL"
            print(
                f"[{completed_count}/{total}] task={local_idx} reward={reward_str} {status} "
                f"({result.duration_seconds:.1f}s)",
                file=sys.stderr,
            )
            if result.error:
                print(f"  Error: {result.error[:500]}", file=sys.stderr)
            return result

    results = await asyncio.gather(
        *[run_with_semaphore(abs_i, i) for i, abs_i in enumerate(indices)],
        return_exceptions=True,
    )

    # Flush pending rollout uploads before the event loop shuts down.
    if config.logging:
        client.rollout.close()

    summary = RunSummary.from_results(
        results,
        run_name=run_name,
        environment=config.env,
        agent=config.agent,
        model=config.model,
        split=config.split,
    )
    summary.print_report()

    # Write aggregate run_result.json
    summary.write_json(Path(output_dir) / "run_result.json")
    print(f"Results: {os.path.abspath(output_dir)}/run_result.json", file=sys.stderr)

    return summary
