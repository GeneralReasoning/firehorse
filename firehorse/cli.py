"""Firehorse CLI — agent evaluation harness for OpenReward environments."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys

from firehorse.config import RunConfig
from firehorse.orchestrator import run_evaluation


async def command_run(
    env: str,
    agent: str,
    model: str,
    n_concurrent: int = 1,
    split: str = "test",
    max_tasks: int | None = None,
    skip_tasks: int = 0,
    plan_mode: bool = False,
    run_name: str | None = None,
    max_turns: int | None = None,
    provider_url: str | None = None,
    disable_builtin_tools: str | None = None,
    secrets: dict[str, str] | None = None,
    output_dir: str | None = None,
    effort: str = "high",
    logging: bool = True,
    use_env_descriptions: bool = False,
    use_all_filesystem_tools: bool = False,
    toolset: str | None = None,
) -> int:
    disabled = []
    if disable_builtin_tools:
        disabled = [t.strip() for t in disable_builtin_tools.split(",") if t.strip()]

    config = RunConfig(
        env=env,
        agent=agent,
        model=model,
        n_concurrent=n_concurrent,
        split=split,
        max_tasks=max_tasks,
        skip_tasks=skip_tasks,
        plan_mode=plan_mode,
        run_name=run_name,
        max_turns=max_turns,
        provider_url=provider_url,
        disable_builtin_tools=disabled,
        secrets=secrets or {},
        output_dir=output_dir,
        effort=effort,
        logging=logging,
        use_builtin_descriptions=not use_env_descriptions,
        use_all_filesystem_tools=use_all_filesystem_tools,
        toolset_name=toolset,
    )

    summary = await run_evaluation(config)
    return 0 if summary.failed == 0 else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="firehorse",
        description="Agent evaluation harness for OpenReward environments",
    )
    parser.add_argument("--env", required=True, help="Environment name (e.g. GeneralReasoning/portfolio)")
    parser.add_argument("--agent", default="claude-code", help="Agent type (default: claude-code)")
    parser.add_argument("--model", required=True, help="Model identifier (e.g. anthropic/claude-opus-4-6)")
    parser.add_argument("--n-concurrent", type=int, default=1, help="Max parallel trials (default: 1)")
    parser.add_argument("--split", default="test", help="Which split to evaluate (default: test)")
    parser.add_argument("--max-tasks", type=int, default=None, help="Limit number of tasks")
    parser.add_argument("--skip-tasks", type=int, default=0, help="Skip first N tasks (default: 0)")
    parser.add_argument("--run-name", default=None, help="Name for this run")
    parser.add_argument("--max-turns", type=int, default=None, help="Max tool call turns per trial")
    parser.add_argument(
        "--effort", default="high", choices=["low", "medium", "high", "max"],
        help="Claude thinking effort level (default: high)",
    )
    parser.add_argument(
        "--provider-url", default=None,
        help="Custom API base URL for non-Anthropic models (e.g. https://openrouter.ai/api/v1)",
    )
    parser.add_argument(
        "--disable-builtin-tools", default=None,
        help="Comma-separated list of Claude built-in tools to disable",
    )
    parser.add_argument(
        "--plan-mode", action="store_true", default=False,
        help="Enable Claude Code plan mode (claude-code agent only)",
    )
    parser.add_argument(
        "--no-logging", action="store_true", default=False,
        help="Disable OpenReward rollout logging (default: logging enabled)",
    )
    parser.add_argument(
        "--secret", action="append", default=[], metavar="KEY=VALUE",
        help="Session secret (can be repeated, e.g. --secret openai_api_key=sk-...)",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Directory to write per-trial JSONL trajectory logs",
    )
    parser.add_argument(
        "--use-env-descriptions", action="store_true", default=False,
        help="Use environment tool descriptions instead of Claude Code built-in descriptions (default: use Claude Code descriptions)",
    )
    parser.add_argument(
        "--use-all-filesystem-tools", action="store_true", default=False,
        help="Codex: expose all filesystem tools via MCP (default: only bash, others filtered)",
    )
    parser.add_argument(
        "--toolset", default=None,
        help="SDK toolset name (e.g. hermes-sandboxed, openclaw-sandboxed). "
             "Defaults to agent name for claude-code/codex; omitted for hermes/openclaw (full mode).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    secrets = {}
    for s in (args.secret or []):
        if "=" not in s:
            print(f"Invalid --secret format: {s!r} (expected KEY=VALUE)", file=sys.stderr)
            return 1
        k, v = s.split("=", 1)
        secrets[k] = v

    try:
        exit_code = asyncio.run(command_run(
            env=args.env,
            agent=args.agent,
            model=args.model,
            n_concurrent=args.n_concurrent,
            split=args.split,
            max_tasks=args.max_tasks,
            skip_tasks=args.skip_tasks,
            plan_mode=args.plan_mode,
            run_name=args.run_name,
            max_turns=args.max_turns,
            provider_url=args.provider_url,
            disable_builtin_tools=args.disable_builtin_tools,
            secrets=secrets,
            output_dir=args.output_dir,
            effort=args.effort,
            logging=not args.no_logging,
            use_env_descriptions=args.use_env_descriptions,
            use_all_filesystem_tools=args.use_all_filesystem_tools,
            toolset=args.toolset,
        ))
    except KeyboardInterrupt:
        print("\nInterrupted", file=sys.stderr)
        return 130

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
